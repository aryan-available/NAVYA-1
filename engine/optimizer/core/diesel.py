"""Diesel Generator Modeling and Logic for NAVYA-1.

This module provides physical and operational calculations for backup diesel generators:
- DieselState canonical representation
- Fuel consumption calculations
- CO2 combustion emissions calculation
- Fuel cost calculations
- Remaining diesel runway hours estimation
- Availability and dispatch viability verification

Single source of truth: NAVYA-1 Architecture Specification.
"""

from dataclasses import dataclass
import math


@dataclass
class DieselState:
    """Canonical representation of diesel generator parameters."""
    available: bool
    maxOutputKW: float
    fuelRemainingLiters: float
    fuelPricePerLiter: float
    litersPerKWh: float = 0.27
    co2KgPerLiter: float = 2.68


def calculate_fuel_consumption(
    diesel_kw: float,
    dt_hours: float = 1.0,
    liters_per_kwh: float = 0.27,
) -> float:
    """Calculate fuel consumed (liters) for a given diesel generation level and duration.

    Formula:
      Fuel_Liters = max(0, diesel_kw) * dt_hours * liters_per_kwh
    """
    if diesel_kw <= 0.0 or dt_hours <= 0.0 or liters_per_kwh <= 0.0:
        return 0.0
    return diesel_kw * dt_hours * liters_per_kwh


def calculate_co2_emissions(
    fuel_liters: float,
    co2_kg_per_liter: float = 2.68,
) -> float:
    """Calculate CO2 emissions (kg) resulting from burning diesel fuel.

    Formula:
      CO2_Kg = max(0, fuel_liters) * co2_kg_per_liter
    """
    if fuel_liters <= 0.0 or co2_kg_per_liter <= 0.0:
        return 0.0
    return fuel_liters * co2_kg_per_liter


def calculate_fuel_cost(
    fuel_liters: float,
    fuel_price_per_liter: float,
) -> float:
    """Calculate total financial cost ($) for diesel fuel consumed.

    Formula:
      Cost = max(0, fuel_liters) * fuel_price_per_liter
    """
    if fuel_liters <= 0.0 or fuel_price_per_liter <= 0.0:
        return 0.0
    return fuel_liters * fuel_price_per_liter


def calculate_runway_hours(
    fuel_remaining_liters: float,
    load_kw: float,
    liters_per_kwh: float = 0.27,
) -> float:
    """Estimate remaining operating runway hours under a specified generation load.

    Formula:
      Hourly_Consumption = load_kw * liters_per_kwh
      Runway_Hours = fuel_remaining_liters / Hourly_Consumption
    """
    if fuel_remaining_liters <= 0.0:
        return 0.0
    if load_kw <= 0.0 or liters_per_kwh <= 0.0:
        return float("inf")

    hourly_liters = load_kw * liters_per_kwh
    return fuel_remaining_liters / hourly_liters


def is_diesel_available_for_dispatch(state: DieselState) -> bool:
    """Check if the diesel generator is operational and has available fuel.

    Returns True if generator is marked available, has remaining fuel, and has rated capacity > 0.
    """
    return bool(
        state.available
        and state.fuelRemainingLiters > 0.0
        and state.maxOutputKW > 0.0
    )
