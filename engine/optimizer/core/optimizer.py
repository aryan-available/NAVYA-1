"""Central Mathematical Optimization Engine for NAVYA-1.

This module orchestrates the LP/MILP optimization:
- Parses canonical microgrid state and forecast inputs
- Constructs optimization constraints and multi-objective function
- Solves the mathematical model using PuLP / CBC solver
- Handles solver statuses (OPTIMAL, INFEASIBLE, UNBOUNDED)
- Produces the canonical dispatch schedule and outcome metrics contract

Single source of truth: NAVYA-1 Architecture Specification.
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import uuid
import pulp

from engine.optimizer.core.objective import ObjectiveParameters, build_objective_expression
from engine.optimizer.core.constraints import (
    OptimizationInputs,
    create_optimization_variables,
    apply_all_constraints,
)
from engine.optimizer.core.dispatch import (
    build_dispatch_schedule,
    calculate_schedule_metrics,
    format_canonical_result,
    OptimizationMetrics,
)


def parse_optimization_request(request_data: Dict[str, Any]) -> OptimizationInputs:
    """Parse a canonical optimization request payload into typed OptimizationInputs."""
    community_id = request_data.get("communityId", "community-001")

    # Extract battery parameters
    batt_data = request_data.get("battery", {})
    soc_percent = float(
        batt_data.get("socPercent", request_data.get("battery.socPercent", 50.0))
    )
    capacity_kwh = float(
        batt_data.get("capacityKWh", request_data.get("battery.capacityKWh", 100.0))
    )
    max_charge_kw = float(
        batt_data.get("maxChargeKW", request_data.get("battery.maxChargeKW", 50.0))
    )
    max_discharge_kw = float(
        batt_data.get("maxDischargeKW", request_data.get("battery.maxDischargeKW", 50.0))
    )
    min_soc_percent = float(
        batt_data.get("minSocPercent", request_data.get("battery.minSocPercent", 20.0))
    )
    charge_eff = float(batt_data.get("chargeEfficiency", 0.95))
    discharge_eff = float(batt_data.get("dischargeEfficiency", 0.95))

    # Extract diesel parameters
    diesel_data = request_data.get("diesel", {})
    diesel_avail = bool(
        diesel_data.get("available", request_data.get("diesel.available", True))
    )
    diesel_max_kw = float(
        diesel_data.get("maxOutputKW", request_data.get("diesel.maxOutputKW", 50.0))
    )
    diesel_fuel = float(
        diesel_data.get("fuelRemainingLiters", request_data.get("diesel.fuelRemainingLiters", 500.0))
    )
    diesel_price = float(
        diesel_data.get("fuelPricePerLiter", request_data.get("diesel.fuelPricePerLiter", 1.50))
    )
    liters_per_kwh = float(diesel_data.get("litersPerKWh", 0.27))

    # Extract forecast timeline points
    raw_points = []
    if "forecast" in request_data and isinstance(request_data["forecast"], dict):
        raw_points = request_data["forecast"].get("points", [])
    elif "points" in request_data and isinstance(request_data["points"], list):
        raw_points = request_data["points"]
    elif "schedule" in request_data and isinstance(request_data["schedule"], list):
        raw_points = request_data["schedule"]

    timestamps: List[str] = []
    demand_kw: List[float] = []
    solar_kw: List[float] = []
    wind_kw: List[float] = []
    critical_kw: List[float] = []

    if raw_points:
        for pt in raw_points:
            timestamps.append(str(pt.get("timestamp", datetime.now(timezone.utc).isoformat())))
            dem = float(pt.get("demandKW", pt.get("demand", {}).get("totalKW", 0.0)))
            s = float(pt.get("solarAvailableKW", pt.get("solar", {}).get("availableKW", 0.0)))
            w = float(pt.get("windAvailableKW", pt.get("wind", {}).get("availableKW", 0.0)))
            c = float(pt.get("criticalLoadKW", dem * 0.40))  # Default 40% critical load if not specified

            demand_kw.append(dem)
            solar_kw.append(s)
            wind_kw.append(w)
            critical_kw.append(c)
    else:
        # Fallback to single instant point from top-level state
        ts = str(request_data.get("timestamp", datetime.now(timezone.utc).isoformat()))
        dem = float(request_data.get("demand", {}).get("totalKW", request_data.get("demand.totalKW", 50.0)))
        s = float(request_data.get("solar", {}).get("availableKW", request_data.get("solar.availableKW", 20.0)))
        w = float(request_data.get("wind", {}).get("availableKW", request_data.get("wind.availableKW", 10.0)))
        c = float(request_data.get("criticalLoadKW", dem * 0.40))

        timestamps.append(ts)
        demand_kw.append(dem)
        solar_kw.append(s)
        wind_kw.append(w)
        critical_kw.append(c)

    dt_hours = float(request_data.get("intervalMinutes", 60)) / 60.0

    return OptimizationInputs(
        timestamps=timestamps,
        demand_kw=demand_kw,
        solar_available_kw=solar_kw,
        wind_available_kw=wind_kw,
        battery_soc_percent=soc_percent,
        battery_capacity_kwh=capacity_kwh,
        battery_max_charge_kw=max_charge_kw,
        battery_max_discharge_kw=max_discharge_kw,
        battery_min_soc_percent=min_soc_percent,
        diesel_available=diesel_avail,
        diesel_max_output_kw=diesel_max_kw,
        diesel_fuel_remaining_liters=diesel_fuel,
        diesel_fuel_price_per_liter=diesel_price,
        critical_load_kw=critical_kw,
        battery_charge_efficiency=charge_eff,
        battery_discharge_efficiency=discharge_eff,
        diesel_liters_per_kwh=liters_per_kwh,
        dt_hours=dt_hours,
    )


def run_optimization(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute mathematical microgrid dispatch optimization and return canonical result."""
    run_id = str(request_data.get("runId") or f"run-{uuid.uuid4().hex[:8]}")
    community_id = str(request_data.get("communityId", "community-001"))
    created_at = str(request_data.get("createdAt") or datetime.now(timezone.utc).isoformat())

    # 1. Parse inputs
    inputs = parse_optimization_request(request_data)

    # 2. Build optimization problem
    prob = pulp.LpProblem(f"NAVYA_Opt_{run_id}", pulp.LpMinimize)
    vars = create_optimization_variables(inputs)
    apply_all_constraints(prob, vars, inputs)

    obj_params = ObjectiveParameters(
        fuel_price_per_liter=inputs.diesel_fuel_price_per_liter,
        liters_per_kwh=inputs.diesel_liters_per_kwh,
    )
    prob += build_objective_expression(
        diesel_vars=vars.diesel_kw,
        battery_charge_vars=vars.battery_charge_kw,
        battery_discharge_vars=vars.battery_discharge_kw,
        unserved_load_vars=vars.unserved_load_kw,
        curtailed_vars=vars.curtailed_kw,
        params=obj_params,
        dt_hours=inputs.dt_hours,
    )

    # 3. Solve model using PuLP / CBC
    solver = pulp.PULP_CBC_CMD(msg=0)
    solver_status = prob.solve(solver)
    status_str = pulp.LpStatus.get(solver_status, "UNKNOWN").upper()

    # 4. Handle solver statuses
    if status_str != "OPTIMAL":
        empty_metrics = OptimizationMetrics(
            totalCost=0.0,
            totalCO2Kg=0.0,
            dieselLiters=0.0,
            renewableUtilizationPercent=0.0,
            unservedEnergyKWh=sum(inputs.demand_kw) * inputs.dt_hours,
        )
        return format_canonical_result(
            run_id=run_id,
            community_id=community_id,
            created_at=created_at,
            status=status_str,
            schedule=[],
            metrics=empty_metrics,
            reason_codes=[f"SOLVER_STATUS_{status_str}"],
        )

    # 5. Extract schedule and metrics
    schedule = build_dispatch_schedule(vars, inputs.timestamps)
    metrics = calculate_schedule_metrics(schedule, inputs)

    # 6. Determine canonical reason codes
    reason_codes: List[str] = []
    if any(item["curtailedRenewableKW"] > 0.05 for item in schedule):
        reason_codes.append("RENEWABLE_CURTAILMENT_ACTIVE")
    if any(item["dieselKW"] > 0.05 for item in schedule):
        reason_codes.append("DIESEL_DISPATCHED")
    if metrics.unservedEnergyKWh > 0.05:
        reason_codes.append("UNSERVED_ENERGY_SHORTFALL")

    return format_canonical_result(
        run_id=run_id,
        community_id=community_id,
        created_at=created_at,
        status="OPTIMAL",
        schedule=schedule,
        metrics=metrics,
        reason_codes=reason_codes,
    )
