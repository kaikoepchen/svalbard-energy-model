"""Energy system optimization model for Isfjord Radio, Svalbard.

This module defines a PyPSA network representing the energy system at
Isfjord Radio, a remote Arctic research station on Svalbard (78 deg N).
The model covers both electricity and heating demand, and optimizes
the capacity mix of renewable generation, storage, and backup diesel
generators under CO2 emission constraints.

The energy system includes:
- Wind turbines (SWP and IceWind types, rated for Arctic conditions)
- Solar PV (rooftop and ground-mounted park installations)
- Battery storage (lithium-ion)
- Hydrogen storage (electrolysis + fuel cell with waste heat recovery)
- Diesel generators (GenSets) producing both electricity and heat
- Heat pumps (geothermal and electric boiler)
- Hot water thermal storage

The station experiences extreme seasonal variation in solar irradiance
(polar night from October to February, midnight sun from April to August)
making wind and storage particularly critical for year-round supply.

Based on the PyPSA-Longyearbyen model by Koen van Greevenbroek and Lars Klein.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml

from src.utilities import load_parameters

# Project root directory (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data():
    """Load configuration, cost parameters, and time series data for Isfjord Radio.

    Reads the model configuration from config.yml, loads technology cost
    parameters from the CSV, and assembles hourly demand, solar capacity
    factors, and wind capacity factors into a single DataFrame.

    Returns
    -------
    config : dict
        Model configuration (snapshots, costs, system options).
    parameters : pd.DataFrame
        Processed technology cost parameters.
    data : pd.DataFrame
        Hourly time series with columns for AC load, heat load,
        PV roof/park capacity factors, and wind capacity factors.
    """
    config_path = PROJECT_ROOT / "config.yml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    parameters = load_parameters(
        PROJECT_ROOT / "data" / "parameters.csv",
        config["costs"],
        config["storage"],
        Nyears=1.0,
    )

    # Hourly electricity and heat demand profiles
    data = pd.read_csv(
        PROJECT_ROOT / "data" / "isfjord_load.csv",
        index_col=0,
        parse_dates=True,
    )

    # Hourly solar PV capacity factors (rooftop and park installations)
    data_pv = pd.read_csv(
        PROJECT_ROOT / "data" / "isfjord_solar.csv",
        index_col=0,
        parse_dates=True,
    )
    data.loc[:, "PV roof"] = data_pv["roof"]
    data.loc[:, "PV park"] = data_pv["park"]

    # Hourly wind capacity factors for two turbine types
    # SWP: standard wind power turbine
    # IceWind: ice-resistant turbine designed for Arctic conditions
    data_wind = pd.read_csv(
        PROJECT_ROOT / "data" / "isfjord_wind.csv",
        index_col=0,
        parse_dates=True,
    )
    data.loc[:, "Wind SWP"] = data_wind["SWP"]
    data.loc[:, "Wind IceWind"] = data_wind["IceWind"]

    return config, parameters, data


def create_network(config, parameters, data, co2_limit, diesel_price):
    """Create the PyPSA network with all extendable components.

    Builds the base energy system network for Isfjord Radio with
    optimizable (extendable) capacities for all generation, storage,
    and conversion technologies.

    Parameters
    ----------
    config : dict
        Model configuration.
    parameters : pd.DataFrame
        Technology cost parameters.
    data : pd.DataFrame
        Hourly time series data.
    co2_limit : float
        Maximum annual CO2 emissions in tonnes.
    diesel_price : float
        Diesel fuel cost in EUR/kWh.

    Returns
    -------
    pypsa.Network
        Network with all components added, ready for optimization.
    """
    # --- System parameters ---
    wind_turbine_size = 25          # kW rated power per turbine module
    co2_per_kwh_diesel = 0.00026   # tonnes CO2 per kWh of diesel burned
    battery_charge_eff = 0.98      # round-trip charge efficiency
    boiler_efficiency = 0.9        # electric boiler thermal efficiency
    genset_elec_eff = 0.4          # diesel generator electrical efficiency
    genset_heat_eff = 0.2          # diesel generator waste heat recovery
    solar_roof_capital = 259.39    # annualised rooftop PV cost (EUR/kW)

    # Fuel cell waste heat recovery: captures 80% of rejected heat
    fuel_cell_heat_eff = (1 - parameters.at["fuel cell", "efficiency"]) * 0.8

    # --- Network setup ---
    n = pypsa.Network()
    snapshots = pd.date_range(
        start=config["snapshots"]["start"],
        end=config["snapshots"]["end"],
        freq="1h",
    )
    n.set_snapshots(snapshots)

    # --- Energy carriers ---
    n.add("Carrier", ["AC", "heat", "hydrogen"])
    n.add("Carrier", "hydrogen", co2_emissions=0)
    n.add("Carrier", "solar", co2_emissions=0)
    n.add("Carrier", "wind", co2_emissions=0)
    n.add("Carrier", "heat pump", co2_emissions=0)
    n.add("Carrier", "diesel", co2_emissions=co2_per_kwh_diesel)

    # Global CO2 constraint on primary energy consumption
    n.add(
        "GlobalConstraint",
        "CO2_limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_limit,
    )

    # --- Buses ---
    n.add("Bus", "AC bus", carrier="AC")
    if config["system"]["heat_enabled"]:
        n.add("Bus", "heat bus", carrier="heat")
    n.add("Bus", "diesel bus", carrier="diesel")

    # --- Demand loads ---
    n.add("Load", "electricity demand", bus="AC bus", p_set=data["AC load"])
    if config["system"]["heat_enabled"]:
        n.add("Load", "heat demand", bus="heat bus", p_set=data["heat load"])

    # --- Wind generation ---
    # SWP (standard wind power) turbines
    n.add(
        "Generator",
        name="SWP turbine",
        bus="AC bus",
        p_nom_max=parameters.at["onwind", "p_nom_max"],
        p_nom_extendable=True,
        p_nom_mod=wind_turbine_size,
        p_max_pu=data["Wind SWP"],
        capital_cost=parameters.at["onwind", "capital_cost"],
        carrier="wind",
    )

    # IceWind turbines (designed for icing conditions in Arctic environments)
    n.add(
        "Generator",
        name="IceWind turbine",
        bus="AC bus",
        p_nom_max=parameters.at["onwind", "p_nom_max"],
        p_nom_extendable=True,
        p_nom_mod=wind_turbine_size,
        p_max_pu=data["Wind IceWind"],
        capital_cost=parameters.at["Ice_wind", "capital_cost"],
        carrier="wind",
    )

    # --- Diesel import and generation ---
    n.add(
        "Generator",
        name="Diesel_import",
        bus="diesel bus",
        p_nom_extendable=True,
        capital_cost=parameters.at["Diesel", "capital_cost"],
        marginal_cost=diesel_price,
        carrier="diesel",
    )

    # Diesel GenSet: converts diesel to electricity + recoverable waste heat
    n.add(
        "Link",
        "GenSet",
        bus0="diesel bus",
        bus1="AC bus",
        bus2="heat bus",
        efficiency=genset_elec_eff,
        efficiency2=genset_heat_eff,
        p_nom_extendable=True,
    )

    # Heat vent: allows dumping excess heat at zero cost (Arctic ventilation)
    n.add(
        "Generator",
        name="heat_vent",
        bus="heat bus",
        p_nom_max=0,
        p_nom_min=-1000,
        p_nom_extendable=True,
        capital_cost=0,
        marginal_cost=0,
    )

    # --- Solar PV ---
    n.add(
        "Generator",
        "PV park",
        bus="AC bus",
        p_nom_max=parameters.at["solar", "p_nom_max"],
        p_nom_extendable=True,
        p_max_pu=data["PV park"],
        capital_cost=parameters.at["solar_park", "capital_cost"],
        carrier="solar",
    )

    n.add(
        "Generator",
        "PV roof",
        bus="AC bus",
        p_nom_max=parameters.at["solar", "p_nom_max"],
        p_nom_extendable=True,
        p_max_pu=data["PV roof"],
        capital_cost=solar_roof_capital,
        carrier="solar",
    )

    # --- Load curtailment (penalty generators for feasibility) ---
    n.add(
        "Generator",
        "load curtailment AC",
        bus="AC bus",
        p_nom=1000000,
        marginal_cost=100000,
    )
    if config["system"]["heat_enabled"]:
        n.add(
            "Generator",
            "load curtailment heat",
            bus="heat bus",
            p_nom=10000,
            marginal_cost=100000,
        )

    # --- Battery storage ---
    n.add("Bus", "battery bus")
    n.add(
        "Store",
        "batteries",
        bus="battery bus",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=parameters.at["battery", "capital_cost"],
    )
    n.add(
        "Link",
        "charge",
        bus0="AC bus",
        bus1="battery bus",
        efficiency=battery_charge_eff,
        p_nom_extendable=True,
    )
    n.add(
        "Link",
        "discharge",
        bus0="battery bus",
        bus1="AC bus",
        p_nom_extendable=True,
    )

    # --- Hydrogen storage (long-duration, seasonal storage) ---
    n.add("Bus", "hydrogen bus", carrier="hydrogen")
    n.add(
        "Store",
        "hydrogen storage",
        bus="hydrogen bus",
        e_cyclic=True,
        standing_loss=parameters.at["hydrogen storage", "standing loss"],
        e_nom_extendable=True,
        capital_cost=parameters.at["hydrogen storage", "capital_cost"],
    )
    # Electrolysis: electricity -> hydrogen
    n.add(
        "Link",
        "electrolysis",
        bus0="AC bus",
        bus1="hydrogen bus",
        efficiency=parameters.at["electrolysis", "efficiency"],
        p_nom_extendable=True,
        capital_cost=parameters.at["electrolysis", "capital_cost"],
        carrier="hydrogen",
    )
    # Fuel cell: hydrogen -> electricity + waste heat recovery
    n.add(
        "Link",
        "fuel cell",
        bus0="hydrogen bus",
        bus1="AC bus",
        bus2="heat bus",
        efficiency=parameters.at["fuel cell", "efficiency"],
        efficiency2=fuel_cell_heat_eff,
        p_nom_extendable=True,
        capital_cost=parameters.at["fuel cell", "capital_cost"],
    )

    # --- Heating systems ---
    if config["system"]["heat_enabled"]:
        # Electric boiler for direct heat production
        n.add(
            "Link",
            "Boiler",
            bus0="AC bus",
            bus1="heat bus",
            p_nom_extendable=True,
            efficiency=boiler_efficiency,
            capital_cost=parameters.at["hot storage thermal generator", "capital_cost"],
        )

        # Geothermal heat pump (COP ~4 in Arctic ground conditions)
        n.add(
            "Link",
            "Geothermal heat pump",
            bus0="AC bus",
            bus1="heat bus",
            p_nom_extendable=True,
            efficiency=4,
            capital_cost=parameters.at["geothermal", "capital_cost"],
        )

        if config["storage"]["thermal_enabled"]:
            # Hot water thermal storage for short-term heat buffering
            n.add(
                "Store",
                "hot water storage",
                bus="heat bus",
                e_cyclic=True,
                standing_loss=parameters.at["hot water storage", "standing loss"],
                e_nom_extendable=True,
                capital_cost=parameters.at["hot water storage", "capital_cost"],
            )

    return n


def add_existing_infrastructure(n, config, parameters, data, diesel_price):
    """Add existing (non-extendable) infrastructure at Isfjord Radio.

    Represents the already-installed equipment at the station with fixed
    capacities that cannot be expanded by the optimizer. This includes
    the current diesel generators, rooftop and park solar PV, batteries,
    electric boiler, and thermal storage.

    Parameters
    ----------
    n : pypsa.Network
        Network to add existing components to.
    config : dict
        Model configuration.
    parameters : pd.DataFrame
        Technology cost parameters.
    data : pd.DataFrame
        Hourly time series data.
    diesel_price : float
        Diesel fuel cost in EUR/kWh.

    Returns
    -------
    pypsa.Network
        Network with existing infrastructure added.
    """
    boiler_efficiency = 0.9
    solar_roof_capital = 259.39

    # Existing diesel generator capacity (350 kW)
    n.add(
        "Generator",
        name="Diesel_import-existing",
        bus="diesel bus",
        p_nom=350,
        p_nom_extendable=False,
        capital_cost=parameters.at["Diesel", "capital_cost"],
        marginal_cost=diesel_price,
        carrier="diesel",
    )

    # Existing rooftop solar PV (98 kWp installed)
    n.add(
        "Generator",
        "PV roof - existing",
        bus="AC bus",
        p_nom_max=parameters.at["solar", "p_nom_max"],
        p_nom=98,
        p_nom_extendable=False,
        p_max_pu=data["PV roof"],
        capital_cost=solar_roof_capital,
        carrier="solar",
    )

    # Existing ground-mounted solar park (200 kWp installed)
    n.add(
        "Generator",
        "PV park - existing",
        bus="AC bus",
        p_nom_max=parameters.at["solar", "p_nom_max"],
        p_nom=200,
        p_nom_extendable=False,
        p_max_pu=data["PV park"],
        capital_cost=parameters.at["solar_park", "capital_cost"],
        carrier="solar",
    )

    # Existing battery storage (405 kWh)
    n.add(
        "Store",
        "batteries-existing",
        bus="battery bus",
        e_nom_extendable=False,
        e_nom=405,
        e_cyclic=True,
        capital_cost=parameters.at["battery", "capital_cost"],
    )

    if config["system"]["heat_enabled"]:
        # Existing electric boiler (128 kW thermal output)
        n.add(
            "Link",
            "Boiler - existing",
            bus0="AC bus",
            bus1="heat bus",
            p_nom=128,
            p_nom_extendable=False,
            efficiency=boiler_efficiency,
            capital_cost=parameters.at["hot storage thermal generator", "capital_cost"],
        )

        if config["storage"]["thermal_enabled"]:
            # Existing hot water storage tank (235 kWh thermal)
            n.add(
                "Store",
                "hot water storage - existing",
                bus="heat bus",
                e_nom=235,
                e_cyclic=True,
                standing_loss=parameters.at["hot water storage", "standing loss"],
                e_nom_extendable=False,
                capital_cost=parameters.at["hot water storage", "capital_cost"],
            )

    return n


def optimize_network(n, config):
    """Solve the capacity expansion and dispatch optimization.

    Optionally resamples time series to a coarser resolution before
    solving to reduce computation time.

    Parameters
    ----------
    n : pypsa.Network
        Fully configured network to optimize.
    config : dict
        Configuration with solver settings and time resolution.

    Returns
    -------
    pypsa.Network
        Solved network with optimal capacities and dispatch.
    """
    offset = config["snapshots"]["resolution"]
    if offset != "1h":
        snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
        n.set_snapshots(snapshot_weightings.index)
        for c in n.iterate_components():
            pnl = getattr(n, c.list_name + "_t")
            for k, df in c.pnl.items():
                if not df.empty:
                    pnl[k] = df.resample(offset).mean()

    n.optimize(solver_name=config["solving"]["solver"])

    return n


def run_model(co2_limit, diesel_price):
    """Run the full Isfjord Radio energy system optimization.

    This is the main entry point. It loads data, builds the network
    with both extendable and existing components, and solves the
    optimization problem.

    Parameters
    ----------
    co2_limit : float
        Maximum annual CO2 emissions in tonnes.
    diesel_price : float
        Diesel fuel cost in EUR/kWh (typically 0.1-0.4).

    Returns
    -------
    pypsa.Network
        Solved network with optimal capacities and hourly dispatch.

    Example
    -------
    >>> n = run_model(co2_limit=50, diesel_price=0.2)
    >>> print(n.generators.p_nom_opt)
    """
    config, parameters, data = load_data()
    n = create_network(config, parameters, data, co2_limit, diesel_price)
    n = add_existing_infrastructure(n, config, parameters, data, diesel_price)
    n = optimize_network(n, config)
    return n


if __name__ == "__main__":
    n = run_model(co2_limit=1000, diesel_price=0.2)
