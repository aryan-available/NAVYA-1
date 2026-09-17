"""Optimization Constraints Module for NAVYA-1.

This module implements all physical and operational constraints for the microgrid optimizer:
- Solar generation limits
- Wind generation limits
- Renewable curtailment accounting
- Power balance (instantaneous generation equals consumption + charging)
- Demand balance (served load + unserved load equals total demand)
- Battery SOC dynamics, minimum/maximum capacity, and charge/discharge limits
- Diesel maximum output limits and cumulative fuel availability constraints
- Critical-load protection (strictly enforced whenever physically feasible)

Single source of truth: NAVYA-1 Architecture Specification.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pulp


@dataclass
class OptimizationInputs:
    """Canonical inputs for optimization over a forecast horizon."""
    timestamps: List[str]
    demand_kw: List[float]
    solar_available_kw: List[float]
    wind_available_kw: List[float]
    battery_soc_percent: float
    battery_capacity_kwh: float
    battery_max_charge_kw: float
    battery_max_discharge_kw: float
    battery_min_soc_percent: float
    diesel_available: bool
    diesel_max_output_kw: float
    diesel_fuel_remaining_liters: float
    diesel_fuel_price_per_liter: float
    critical_load_kw: Optional[List[float]] = None
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    diesel_liters_per_kwh: float = 0.27
    dt_hours: float = 1.0


@dataclass
class OptimizationVariables:
    """Container for all decision variables indexed over the time horizon."""
    solar_kw: List[pulp.LpVariable] = field(default_factory=list)
    wind_kw: List[pulp.LpVariable] = field(default_factory=list)
    battery_charge_kw: List[pulp.LpVariable] = field(default_factory=list)
    battery_discharge_kw: List[pulp.LpVariable] = field(default_factory=list)
    battery_energy_kwh: List[pulp.LpVariable] = field(default_factory=list)
    diesel_kw: List[pulp.LpVariable] = field(default_factory=list)
    load_served_kw: List[pulp.LpVariable] = field(default_factory=list)
    unserved_load_kw: List[pulp.LpVariable] = field(default_factory=list)
    curtailed_kw: List[pulp.LpVariable] = field(default_factory=list)
    critical_deficit_kw: List[pulp.LpVariable] = field(default_factory=list)


def create_optimization_variables(inputs: OptimizationInputs) -> OptimizationVariables:
    """Instantiate decision variables for each time step in the horizon."""
    n_steps = len(inputs.timestamps)
    vars = OptimizationVariables()

    min_energy_kwh = (inputs.battery_min_soc_percent / 100.0) * inputs.battery_capacity_kwh
    max_energy_kwh = inputs.battery_capacity_kwh

    for t in range(n_steps):
        s_avail = max(0.0, inputs.solar_available_kw[t])
        w_avail = max(0.0, inputs.wind_available_kw[t])
        dem = max(0.0, inputs.demand_kw[t])

        vars.solar_kw.append(
            pulp.LpVariable(f"solar_{t}", lowBound=0.0, upBound=s_avail)
        )
        vars.wind_kw.append(
            pulp.LpVariable(f"wind_{t}", lowBound=0.0, upBound=w_avail)
        )
        vars.battery_charge_kw.append(
            pulp.LpVariable(f"bat_ch_{t}", lowBound=0.0, upBound=inputs.battery_max_charge_kw)
        )
        vars.battery_discharge_kw.append(
            pulp.LpVariable(f"bat_dis_{t}", lowBound=0.0, upBound=inputs.battery_max_discharge_kw)
        )
        vars.battery_energy_kwh.append(
            pulp.LpVariable(f"bat_e_{t}", lowBound=min_energy_kwh, upBound=max_energy_kwh)
        )

        max_diesel = inputs.diesel_max_output_kw if inputs.diesel_available else 0.0
        vars.diesel_kw.append(
            pulp.LpVariable(f"diesel_{t}", lowBound=0.0, upBound=max_diesel)
        )
        vars.load_served_kw.append(
            pulp.LpVariable(f"load_served_{t}", lowBound=0.0, upBound=dem)
        )
        vars.unserved_load_kw.append(
            pulp.LpVariable(f"unserved_{t}", lowBound=0.0, upBound=dem)
        )
        vars.curtailed_kw.append(
            pulp.LpVariable(f"curtailed_{t}", lowBound=0.0, upBound=s_avail + w_avail)
        )
        vars.critical_deficit_kw.append(
            pulp.LpVariable(f"crit_def_{t}", lowBound=0.0, upBound=dem)
        )

    return vars


def add_generation_limits(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Enforce physical generation upper bounds."""
    for t in range(len(inputs.timestamps)):
        prob += (
            vars.solar_kw[t] <= inputs.solar_available_kw[t],
            f"SolarGenLimit_{t}",
        )
        prob += (
            vars.wind_kw[t] <= inputs.wind_available_kw[t],
            f"WindGenLimit_{t}",
        )


def add_curtailment_constraints(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Ensure curtailed renewable energy tracks unused solar and wind generation."""
    for t in range(len(inputs.timestamps)):
        total_avail = inputs.solar_available_kw[t] + inputs.wind_available_kw[t]
        used_renewables = vars.solar_kw[t] + vars.wind_kw[t]
        prob += (
            vars.curtailed_kw[t] == total_avail - used_renewables,
            f"CurtailmentDef_{t}",
        )


def add_power_balance_constraints(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Enforce microgrid power balance at every timestep:
    Generation + Battery Discharge = Served Load + Battery Charge.
    """
    for t in range(len(inputs.timestamps)):
        total_supply = (
            vars.solar_kw[t]
            + vars.wind_kw[t]
            + vars.battery_discharge_kw[t]
            + vars.diesel_kw[t]
        )
        total_demand = vars.load_served_kw[t] + vars.battery_charge_kw[t]
        prob += (total_supply == total_demand, f"PowerBalance_{t}")


def add_demand_balance_constraints(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Enforce served load + unserved load = total demand."""
    for t in range(len(inputs.timestamps)):
        prob += (
            vars.load_served_kw[t] + vars.unserved_load_kw[t] == inputs.demand_kw[t],
            f"DemandBalance_{t}",
        )


def add_battery_constraints(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Enforce battery SOC tracking and dynamics across time steps:
    E(t) = E(t-1) + [eta_ch * P_ch(t) - P_dis(t) / eta_dis] * dt.
    """
    initial_energy_kwh = (inputs.battery_soc_percent / 100.0) * inputs.battery_capacity_kwh
    eta_ch = inputs.battery_charge_efficiency
    eta_dis = inputs.battery_discharge_efficiency
    dt = inputs.dt_hours

    for t in range(len(inputs.timestamps)):
        prev_energy = initial_energy_kwh if t == 0 else vars.battery_energy_kwh[t - 1]
        energy_change = (eta_ch * vars.battery_charge_kw[t] - (1.0 / eta_dis) * vars.battery_discharge_kw[t]) * dt
        prob += (
            vars.battery_energy_kwh[t] == prev_energy + energy_change,
            f"BatteryEnergyDynamics_{t}",
        )


def add_diesel_constraints(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Enforce diesel operating availability, maximum power, and fuel inventory limits:
    sum_t (liters_per_kwh * P_diesel(t) * dt) <= fuelRemainingLiters.
    """
    dt = inputs.dt_hours
    max_output = inputs.diesel_max_output_kw if inputs.diesel_available else 0.0

    for t in range(len(inputs.timestamps)):
        prob += (vars.diesel_kw[t] <= max_output, f"DieselMaxOutput_{t}")

    # Cumulative fuel consumption must not exceed available fuel
    total_fuel_consumed = pulp.lpSum(
        inputs.diesel_liters_per_kwh * vars.diesel_kw[t] * dt
        for t in range(len(inputs.timestamps))
    )
    prob += (
        total_fuel_consumed <= inputs.diesel_fuel_remaining_liters,
        "DieselCumulativeFuelLimit",
    )


def add_critical_load_constraints(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Enforce critical load protection.

    Critical load must be served whenever physically feasible. If physical capacity
    is strictly insufficient under extreme shortfall conditions, the critical deficit
    variable absorbs the deficit without breaking model feasibility, while guaranteeing
    that non-critical loads are shed first.
    """
    critical_loads = inputs.critical_load_kw or [0.0] * len(inputs.timestamps)
    max_diesel = inputs.diesel_max_output_kw if inputs.diesel_available else 0.0

    # Calculate actual deliverable battery discharge energy above minSocPercent
    usable_soc_fraction = max(
        0.0, (inputs.battery_soc_percent - inputs.battery_min_soc_percent) / 100.0
    )
    available_battery_energy_kwh = (
        usable_soc_fraction * inputs.battery_capacity_kwh * inputs.battery_discharge_efficiency
    )

    for t in range(len(inputs.timestamps)):
        crit_kw = min(critical_loads[t], inputs.demand_kw[t])
        prob += (
            vars.load_served_kw[t] >= crit_kw - vars.critical_deficit_kw[t],
            f"CriticalLoadServed_{t}",
        )
        prob += (
            vars.critical_deficit_kw[t] <= vars.unserved_load_kw[t],
            f"CriticalDeficitUpperBound_{t}",
        )

        # Battery can only deliver power if it has usable energy above minSocPercent
        max_battery_discharge_possible = 0.0
        if available_battery_energy_kwh > 1e-4 and inputs.dt_hours > 0:
            max_battery_discharge_possible = min(
                inputs.battery_max_discharge_kw,
                available_battery_energy_kwh / inputs.dt_hours,
            )

        total_physical_capacity = (
            inputs.solar_available_kw[t]
            + inputs.wind_available_kw[t]
            + max_battery_discharge_possible
            + max_diesel
        )

        # Strictly enforce zero critical deficit only when physical resources are sufficient
        if total_physical_capacity >= crit_kw:
            prob += (vars.critical_deficit_kw[t] == 0.0, f"CriticalLoadGuaranteed_{t}")
            # Track battery energy consumption allocated towards critical demand
            renewable_and_diesel = (
                inputs.solar_available_kw[t]
                + inputs.wind_available_kw[t]
                + max_diesel
            )
            battery_needed = max(0.0, crit_kw - renewable_and_diesel)
            available_battery_energy_kwh = max(
                0.0, available_battery_energy_kwh - (battery_needed * inputs.dt_hours)
            )
        else:
            available_battery_energy_kwh = 0.0



def apply_all_constraints(
    prob: pulp.LpProblem,
    vars: OptimizationVariables,
    inputs: OptimizationInputs,
) -> None:
    """Apply the full suite of NAVYA optimization constraints to the PuLP problem."""
    add_generation_limits(prob, vars, inputs)
    add_curtailment_constraints(prob, vars, inputs)
    add_power_balance_constraints(prob, vars, inputs)
    add_demand_balance_constraints(prob, vars, inputs)
    add_battery_constraints(prob, vars, inputs)
    add_diesel_constraints(prob, vars, inputs)
    add_critical_load_constraints(prob, vars, inputs)
