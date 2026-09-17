"""Pytest Suite for Optimization Constraints (engine/optimizer/tests/test_constraints.py).

Tests:
1. Generation limits and curtailment (solar, wind upper bounds, curtailment equality)
2. Power balance equality at every timestep (generation + discharge = load served + charge)
3. Demand balance at every timestep (load served + unserved load = total demand)
4. Battery energy and power limits (min SOC, max capacity, charge/discharge rates)
5. Diesel cumulative fuel cutoff and availability flag
6. Critical-load protection under partial and severe deficit conditions

Single source of truth: NAVYA-1 Architecture Specification.
"""

import pytest
import pulp

from engine.optimizer.core.constraints import (
    OptimizationInputs,
    create_optimization_variables,
    apply_all_constraints,
)
from engine.optimizer.core.objective import ObjectiveParameters, build_objective_expression


def _solve_test_lp(inputs: OptimizationInputs):
    """Helper to construct and solve an LP problem with the full constraint set."""
    prob = pulp.LpProblem("TestConstraints", pulp.LpMinimize)
    vars = create_optimization_variables(inputs)
    apply_all_constraints(prob, vars, inputs)
    obj_params = ObjectiveParameters(fuel_price_per_liter=inputs.diesel_fuel_price_per_liter)
    prob += build_objective_expression(
        diesel_vars=vars.diesel_kw,
        battery_charge_vars=vars.battery_charge_kw,
        battery_discharge_vars=vars.battery_discharge_kw,
        unserved_load_vars=vars.unserved_load_kw,
        curtailed_vars=vars.curtailed_kw,
        params=obj_params,
        dt_hours=inputs.dt_hours,
    )
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)
    return pulp.LpStatus[status], vars, prob


def test_generation_limits_and_curtailment():
    """Verify that solar and wind generation never exceed availability, and curtailment tracks unused energy."""
    inputs = OptimizationInputs(
        timestamps=["2026-09-17T12:00:00Z"],
        demand_kw=[30.0],
        solar_available_kw=[100.0],
        wind_available_kw=[50.0],
        battery_soc_percent=100.0,  # Full battery, cannot absorb excess
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=30.0,
        battery_max_discharge_kw=30.0,
        battery_min_soc_percent=20.0,
        diesel_available=True,
        diesel_max_output_kw=50.0,
        diesel_fuel_remaining_liters=100.0,
        diesel_fuel_price_per_liter=1.50,
    )

    status, vars, _ = _solve_test_lp(inputs)
    assert status == "Optimal"

    s_val = pulp.value(vars.solar_kw[0])
    w_val = pulp.value(vars.wind_kw[0])
    c_val = pulp.value(vars.curtailed_kw[0])

    assert s_val <= 100.0 + 1e-4
    assert w_val <= 50.0 + 1e-4
    assert s_val >= 0.0
    assert w_val >= 0.0

    # Total generation + curtailment must equal total available renewables
    assert abs((s_val + w_val + c_val) - 150.0) < 1e-4
    assert c_val > 0.0  # Excess must be curtailed since battery is full and demand is only 30kW


def test_power_balance():
    """Verify that at every timestep: solar + wind + batteryDischarge + diesel = loadServed + batteryCharge."""
    inputs = OptimizationInputs(
        timestamps=["2026-09-17T12:00:00Z", "2026-09-17T13:00:00Z", "2026-09-17T14:00:00Z"],
        demand_kw=[50.0, 70.0, 40.0],
        solar_available_kw=[40.0, 10.0, 0.0],
        wind_available_kw=[20.0, 10.0, 15.0],
        battery_soc_percent=50.0,
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=30.0,
        battery_max_discharge_kw=30.0,
        battery_min_soc_percent=20.0,
        diesel_available=True,
        diesel_max_output_kw=50.0,
        diesel_fuel_remaining_liters=100.0,
        diesel_fuel_price_per_liter=1.50,
    )

    status, vars, _ = _solve_test_lp(inputs)
    assert status == "Optimal"

    for t in range(3):
        supply = (
            pulp.value(vars.solar_kw[t])
            + pulp.value(vars.wind_kw[t])
            + pulp.value(vars.battery_discharge_kw[t])
            + pulp.value(vars.diesel_kw[t])
        )
        consumption = pulp.value(vars.load_served_kw[t]) + pulp.value(vars.battery_charge_kw[t])
        assert abs(supply - consumption) < 1e-4, f"Power balance violated at t={t}: {supply} != {consumption}"


def test_demand_balance():
    """Verify that at every timestep: loadServed + unserved = demand.totalKW."""
    inputs = OptimizationInputs(
        timestamps=["2026-09-17T18:00:00Z", "2026-09-17T19:00:00Z"],
        demand_kw=[60.0, 80.0],
        solar_available_kw=[0.0, 0.0],
        wind_available_kw=[5.0, 5.0],
        battery_soc_percent=20.0,  # Min SOC reached
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=30.0,
        battery_max_discharge_kw=30.0,
        battery_min_soc_percent=20.0,
        diesel_available=True,
        diesel_max_output_kw=40.0,
        diesel_fuel_remaining_liters=100.0,
        diesel_fuel_price_per_liter=1.50,
    )

    status, vars, _ = _solve_test_lp(inputs)
    assert status == "Optimal"

    for t in range(2):
        served = pulp.value(vars.load_served_kw[t])
        unserved = pulp.value(vars.unserved_load_kw[t])
        total_demand = inputs.demand_kw[t]
        assert abs((served + unserved) - total_demand) < 1e-4, f"Demand balance violated at t={t}"


def test_battery_energy_and_power_limits():
    """Verify battery SOC never falls below minSocPercent, never exceeds capacity, and obeys charge/discharge limits."""
    inputs = OptimizationInputs(
        timestamps=["2026-09-17T01:00:00Z", "2026-09-17T02:00:00Z"],
        demand_kw=[50.0, 50.0],
        solar_available_kw=[0.0, 0.0],
        wind_available_kw=[0.0, 0.0],
        battery_soc_percent=25.0,  # 25 kWh stored
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=30.0,
        battery_max_discharge_kw=30.0,
        battery_min_soc_percent=20.0,  # 20 kWh minimum
        diesel_available=False,        # Cannot use diesel
        diesel_max_output_kw=0.0,
        diesel_fuel_remaining_liters=0.0,
        diesel_fuel_price_per_liter=1.50,
    )

    status, vars, _ = _solve_test_lp(inputs)
    assert status == "Optimal"

    for t in range(2):
        e_val = pulp.value(vars.battery_energy_kwh[t])
        dis_val = pulp.value(vars.battery_discharge_kw[t])
        ch_val = pulp.value(vars.battery_charge_kw[t])

        # Battery energy must stay within [20.0, 100.0]
        assert e_val >= 20.0 - 1e-4, f"Battery fell below min SOC: {e_val} < 20.0"
        assert e_val <= 100.0 + 1e-4, f"Battery exceeded capacity: {e_val} > 100.0"

        # Charge / discharge power limits
        assert dis_val <= 30.0 + 1e-4
        assert ch_val <= 30.0 + 1e-4


def test_diesel_cumulative_fuel_cutoff_and_unavailability():
    """Verify cumulative fuel cap limits diesel generation, and unavailable diesel produces zero output."""
    # Test 1: Unavailable diesel
    inputs_unavail = OptimizationInputs(
        timestamps=["2026-09-17T12:00:00Z"],
        demand_kw=[50.0],
        solar_available_kw=[0.0],
        wind_available_kw=[0.0],
        battery_soc_percent=20.0,
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=20.0,
        battery_max_discharge_kw=20.0,
        battery_min_soc_percent=20.0,
        diesel_available=False,
        diesel_max_output_kw=50.0,
        diesel_fuel_remaining_liters=200.0,
        diesel_fuel_price_per_liter=1.50,
    )
    status, vars, _ = _solve_test_lp(inputs_unavail)
    assert status == "Optimal"
    assert pulp.value(vars.diesel_kw[0]) == 0.0

    # Test 2: Cumulative fuel limitation
    # Fuel remaining = 13.5 L -> At 0.27 L/kWh, can generate at most 13.5 / 0.27 = 50.0 kWh
    inputs_fuel_cap = OptimizationInputs(
        timestamps=["2026-09-17T12:00:00Z", "2026-09-17T13:00:00Z"],
        demand_kw=[50.0, 50.0],  # Demands 100 kWh total
        solar_available_kw=[0.0, 0.0],
        wind_available_kw=[0.0, 0.0],
        battery_soc_percent=20.0,
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=20.0,
        battery_max_discharge_kw=20.0,
        battery_min_soc_percent=20.0,
        diesel_available=True,
        diesel_max_output_kw=50.0,
        diesel_fuel_remaining_liters=13.5,  # Exactly 50 kWh worth of fuel
        diesel_fuel_price_per_liter=1.50,
        diesel_liters_per_kwh=0.27,
    )
    status2, vars2, _ = _solve_test_lp(inputs_fuel_cap)
    assert status2 == "Optimal"
    total_diesel_gen = pulp.value(vars2.diesel_kw[0]) + pulp.value(vars2.diesel_kw[1])
    assert total_diesel_gen <= 50.0 + 1e-4
    total_fuel_used = total_diesel_gen * 0.27
    assert total_fuel_used <= 13.5 + 1e-4


def test_critical_load_protection():
    """Verify critical load is protected first during shortages, and model remains feasible if capacity is physically insufficient."""
    # Case A: Partial shortage where generation is sufficient to cover critical load (30 kW) but not total demand (70 kW)
    inputs_partial = OptimizationInputs(
        timestamps=["2026-09-17T12:00:00Z"],
        demand_kw=[70.0],
        solar_available_kw=[0.0],
        wind_available_kw=[0.0],
        battery_soc_percent=20.0,
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=20.0,
        battery_max_discharge_kw=20.0,
        battery_min_soc_percent=20.0,
        diesel_available=True,
        diesel_max_output_kw=40.0,  # Total supply = 40 kW
        diesel_fuel_remaining_liters=100.0,
        diesel_fuel_price_per_liter=1.50,
        critical_load_kw=[30.0],    # Critical demand = 30 kW
    )
    status_a, vars_a, _ = _solve_test_lp(inputs_partial)
    assert status_a == "Optimal"
    served_a = pulp.value(vars_a.load_served_kw[0])
    unserved_a = pulp.value(vars_a.unserved_load_kw[0])
    # 40 kW served >= 30 kW critical load (critical load fully protected)
    assert served_a >= 30.0 - 1e-4
    assert unserved_a == pytest.approx(30.0, 0.1)  # Only non-critical load (40/50) shed

    # Case B: Extreme shortage where total physical capacity (20 kW) is less than critical load (30 kW)
    # Model must still solve to Optimal without raising infeasibility
    inputs_extreme = OptimizationInputs(
        timestamps=["2026-09-17T12:00:00Z"],
        demand_kw=[70.0],
        solar_available_kw=[0.0],
        wind_available_kw=[0.0],
        battery_soc_percent=20.0,
        battery_capacity_kwh=100.0,
        battery_max_charge_kw=20.0,
        battery_max_discharge_kw=20.0,
        battery_min_soc_percent=20.0,
        diesel_available=True,
        diesel_max_output_kw=20.0,  # Max physical power is 20 kW < 30 kW critical load
        diesel_fuel_remaining_liters=100.0,
        diesel_fuel_price_per_liter=1.50,
        critical_load_kw=[30.0],
    )
    status_b, vars_b, _ = _solve_test_lp(inputs_extreme)
    assert status_b == "Optimal"
    served_b = pulp.value(vars_b.load_served_kw[0])
    assert served_b == pytest.approx(20.0, 0.1)  # All available physical capacity is dispatched
