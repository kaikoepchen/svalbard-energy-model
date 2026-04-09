"""Utility functions for cost calculations in the PyPSA energy system model.

Adapted from PyPSA-Longyearbyen by Koen van Greevenbroek and Lars Klein.
Original source: PyPSA-Eur (https://github.com/PyPSA/pypsa-eur)
"""

import math

import pandas as pd


def annuity(n, r=0):
    """Calculate the annuity factor for an asset.

    Parameters
    ----------
    n : float or pd.Series
        Lifetime in years.
    r : float or pd.Series
        Discount rate (e.g. 0.06 for 6%).

    Returns
    -------
    float or pd.Series
        Annuity factor such that annuity(20, 0.05) * 20 ~ 1.6.
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def load_parameters(parameters_file, cost_config, storage_config, Nyears=1.0):
    """Load and process technology cost and efficiency parameters.

    Reads a CSV of technology parameters, applies unit conversions,
    calculates annualised capital costs and marginal costs, and computes
    combined storage costs for battery and hydrogen storage systems.

    Parameters
    ----------
    parameters_file : str or Path
        Path to a CSV file with columns: technology, year, parameter, value, unit, source.
    cost_config : dict
        Cost assumptions including 'USD_to_EUR', 'year', and 'discountrate'.
    storage_config : dict
        Storage configuration with 'max_hours' for batteries and H2.
    Nyears : float
        Number of years modelled (default 1.0).

    Returns
    -------
    pd.DataFrame
        Processed parameters indexed by technology with columns for
        capital_cost, marginal_cost, efficiency, etc.
    """
    costs = pd.read_csv(parameters_file, index_col=list(range(3))).sort_index()

    # Convert units: kW -> MW and USD -> EUR
    costs.loc[costs.unit.str.contains("/MW"), "value"] *= 1e-3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= cost_config["USD_to_EUR"]

    # Convert standing losses from %/day to hourly loss fractions
    costs.loc[costs.unit.str.contains("%/day"), "value"] = costs.loc[
        costs.unit.str.contains("%/day"), "value"
    ].apply(lambda x: math.pow(1 + x / 100.0, 1.0 / 24) - 1)

    # Filter to the target cost year and pivot to wide format
    costs = (
        costs.loc[pd.IndexSlice[:, cost_config["year"], :], "value"]
        .unstack(level=2)
        .groupby("technology")
        .sum(min_count=1)
    )

    # Add default columns
    costs["CO2 intensity"] = None
    costs["discount rate"] = None
    costs["fuel"] = None

    costs = costs.fillna(
        {
            "CO2 intensity": 0,
            "FOM": 0,
            "VOM": 0,
            "discount rate": cost_config["discountrate"],
            "efficiency": 1,
            "fuel": 0,
            "investment": 0,
            "lifetime": 25,
        }
    )

    # Annualised capital cost = (annuity + FOM fraction) * investment
    costs["capital_cost"] = (
        (annuity(costs["lifetime"], costs["discount rate"]) + costs["FOM"] / 100.0)
        * costs["investment"]
        * Nyears
    )

    # Marginal cost = variable O&M + fuel cost adjusted for efficiency
    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    # Compute combined storage costs for StorageUnit components
    def costs_for_storage(store, link1, link2=None, max_hours=1.0):
        capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
        if link2 is not None:
            capital_cost += link2["capital_cost"]
        return pd.Series(
            dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0)
        )

    max_hours = storage_config["max_hours"]
    costs.loc["battery"] = costs_for_storage(
        costs.loc["battery storage"],
        costs.loc["battery inverter"],
        max_hours=max_hours["battery"],
    )
    costs.loc["H2"] = costs_for_storage(
        costs.loc["hydrogen storage"],
        costs.loc["fuel cell"],
        costs.loc["electrolysis"],
        max_hours=max_hours["H2"],
    )

    return costs
