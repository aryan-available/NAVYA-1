"""Optimization Objective Module for NAVYA-1.

This module defines the multi-objective formulation for the microgrid optimizer:
- Operating cost (diesel fuel cost and maintenance)
- CO2 emissions (carbon impact from diesel combustion)
- Diesel fuel consumption
- Battery degradation (cycling wear and throughput penalty)
- Unserved energy / reliability (extreme penalty to strictly prioritize reliability)
- Renewable curtailment (minor penalty to maximize green energy utilization)

Single source of truth: NAVYA-1 Architecture Specification.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import pulp


# Standard physical constants for microgrid operations
DEFAULT_DIESEL_LITERS_PER_KWH = 0.27    # Average specific fuel consumption (~0.25 - 0.30 L/kWh)
DEFAULT_CO2_KG_PER_LITER = 2.68        # Standard diesel combustion emissions (2.68 kg CO2/L)
DEFAULT_UNSERVED_ENERGY_PENALTY = 10000.0  # Very strong penalty ($/kWh) for reliability
DEFAULT_BATTERY_DEG_COST_PER_KWH = 0.015   # Battery degradation wear cost ($/kWh throughput)
DEFAULT_CURTAILMENT_PENALTY = 0.001       # Minor penalty ($/kWh) to encourage green usage
DEFAULT_CO2_WEIGHT = 0.05                 # Weight for CO2 cost ($/kg CO2)


@dataclass
class ObjectiveParameters:
    """Parameters governing the optimization objective function."""
    fuel_price_per_liter: float = 1.50
    liters_per_kwh: float = DEFAULT_DIESEL_LITERS_PER_KWH
    co2_kg_per_liter: float = DEFAULT_CO2_KG_PER_LITER
    co2_cost_weight: float = DEFAULT_CO2_WEIGHT
    battery_deg_cost_per_kwh: float = DEFAULT_BATTERY_DEG_COST_PER_KWH
    unserved_energy_penalty: float = DEFAULT_UNSERVED_ENERGY_PENALTY
    curtailment_penalty: float = DEFAULT_CURTAILMENT_PENALTY


def build_objective_expression(
    diesel_vars: List[pulp.LpVariable],
    battery_charge_vars: List[pulp.LpVariable],
    battery_discharge_vars: List[pulp.LpVariable],
    unserved_load_vars: List[pulp.LpVariable],
    curtailed_vars: List[pulp.LpVariable],
    params: ObjectiveParameters,
    dt_hours: float = 1.0,
) -> pulp.LpAffineExpression:
    """Build the PuLP linear expression representing the total cost to minimize.

    Objective:
      Min Z = sum_t [
          C_fuel(t) + C_co2(t) + C_deg(t) + C_unserved(t) + C_curtail(t)
      ]

    Where for each timestep t:
      - Fuel cost: diesel_fuel_price_per_liter * liters_per_kwh * P_diesel(t) * dt
      - CO2 cost: co2_cost_weight * co2_kg_per_liter * liters_per_kwh * P_diesel(t) * dt
      - Battery degradation: battery_deg_cost_per_kwh * (P_charge(t) + P_discharge(t)) * dt
      - Unserved energy penalty: unserved_energy_penalty * P_unserved(t) * dt
      - Curtailment penalty: curtailment_penalty * P_curtailed(t) * dt
    """
    cost_terms = []

    # Combined diesel unit cost ($/kWh of diesel generation)
    diesel_fuel_cost_per_kwh = params.fuel_price_per_liter * params.liters_per_kwh
    diesel_co2_cost_per_kwh = params.co2_cost_weight * params.co2_kg_per_liter * params.liters_per_kwh
    total_diesel_rate_per_kwh = diesel_fuel_cost_per_kwh + diesel_co2_cost_per_kwh

    n_steps = len(diesel_vars)
    for t in range(n_steps):
        # 1. Diesel generation (fuel cost + CO2 emission impact)
        cost_terms.append(total_diesel_rate_per_kwh * dt_hours * diesel_vars[t])

        # 2. Battery degradation throughput (charge + discharge wear)
        cost_terms.append(params.battery_deg_cost_per_kwh * dt_hours * battery_charge_vars[t])
        cost_terms.append(params.battery_deg_cost_per_kwh * dt_hours * battery_discharge_vars[t])

        # 3. Extreme penalty for unserved energy to safeguard reliability
        cost_terms.append(params.unserved_energy_penalty * dt_hours * unserved_load_vars[t])

        # 4. Curtailment penalty (encourage renewable utilization)
        cost_terms.append(params.curtailment_penalty * dt_hours * curtailed_vars[t])

    return pulp.lpSum(cost_terms)


def compute_objective_metrics(
    diesel_kwh_list: List[float],
    unserved_kwh_list: List[float],
    total_renewable_generated_kwh: float,
    total_renewable_available_kwh: float,
    params: ObjectiveParameters,
    battery_throughput_kwh: float = 0.0,
) -> Dict[str, float]:
    """Calculate the canonical outcome metrics from solved values.

    Returns:
      {
        "totalCost": float,
        "totalCO2Kg": float,
        "dieselLiters": float,
        "renewableUtilizationPercent": float,
        "unservedEnergyKWh": float
      }
    """
    total_diesel_kwh = sum(diesel_kwh_list)
    total_unserved_kwh = sum(unserved_kwh_list)

    total_diesel_liters = total_diesel_kwh * params.liters_per_kwh
    fuel_cost = total_diesel_liters * params.fuel_price_per_liter
    degradation_cost = battery_throughput_kwh * params.battery_deg_cost_per_kwh
    unserved_cost = total_unserved_kwh * params.unserved_energy_penalty
    total_cost = fuel_cost + degradation_cost + unserved_cost

    total_co2_kg = total_diesel_liters * params.co2_kg_per_liter

    if total_renewable_available_kwh > 1e-6:
        renewable_utilization = (total_renewable_generated_kwh / total_renewable_available_kwh) * 100.0
        renewable_utilization = min(100.0, max(0.0, renewable_utilization))
    else:
        renewable_utilization = 100.0 if total_renewable_generated_kwh >= 0 else 0.0

    return {
        "totalCost": round(total_cost, 2),
        "totalCO2Kg": round(total_co2_kg, 2),
        "dieselLiters": round(total_diesel_liters, 2),
        "renewableUtilizationPercent": round(renewable_utilization, 2),
        "unservedEnergyKWh": round(total_unserved_kwh, 2),
    }
