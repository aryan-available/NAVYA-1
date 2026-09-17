"""Dispatch Output and Schedule Construction Module for NAVYA-1.

This module converts solved optimization variables into the canonical schedule
and outcome contracts specified by the NAVYA-1 Architecture:
- DispatchInterval representation
- OptimizationMetrics calculation
- Schedule extraction from solved decision variables
- Canonical output packaging

Single source of truth: NAVYA-1 Architecture Specification.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import pulp

from engine.optimizer.core.constraints import OptimizationVariables, OptimizationInputs
from engine.optimizer.core.diesel import calculate_fuel_consumption, calculate_co2_emissions, calculate_fuel_cost
from engine.optimizer.core.battery import calculate_battery_throughput


@dataclass
class DispatchInterval:
    """Canonical dispatch interval representation."""
    timestamp: str
    solarKW: float
    windKW: float
    batteryChargeKW: float
    batteryDischargeKW: float
    dieselKW: float
    loadServedKW: float
    curtailedRenewableKW: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "solarKW": round(self.solarKW, 2),
            "windKW": round(self.windKW, 2),
            "batteryChargeKW": round(self.batteryChargeKW, 2),
            "batteryDischargeKW": round(self.batteryDischargeKW, 2),
            "dieselKW": round(self.dieselKW, 2),
            "loadServedKW": round(self.loadServedKW, 2),
            "curtailedRenewableKW": round(self.curtailedRenewableKW, 2),
        }


@dataclass
class OptimizationMetrics:
    """Canonical summary metrics of an optimization execution."""
    totalCost: float
    totalCO2Kg: float
    dieselLiters: float
    renewableUtilizationPercent: float
    unservedEnergyKWh: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "totalCost": round(self.totalCost, 2),
            "totalCO2Kg": round(self.totalCO2Kg, 2),
            "dieselLiters": round(self.dieselLiters, 2),
            "renewableUtilizationPercent": round(self.renewableUtilizationPercent, 2),
            "unservedEnergyKWh": round(self.unservedEnergyKWh, 2),
        }


def _val(var: pulp.LpVariable) -> float:
    """Safely extract variable value, clamping numerical solver noise."""
    v = pulp.value(var)
    if v is None:
        return 0.0
    return max(0.0, float(v)) if abs(v) > 1e-6 else 0.0


def build_dispatch_schedule(
    vars: OptimizationVariables,
    timestamps: List[str],
) -> List[Dict[str, Any]]:
    """Convert solved OptimizationVariables into canonical dispatch interval dictionaries."""
    schedule: List[Dict[str, Any]] = []

    for t, ts in enumerate(timestamps):
        interval = DispatchInterval(
            timestamp=ts,
            solarKW=_val(vars.solar_kw[t]),
            windKW=_val(vars.wind_kw[t]),
            batteryChargeKW=_val(vars.battery_charge_kw[t]),
            batteryDischargeKW=_val(vars.battery_discharge_kw[t]),
            dieselKW=_val(vars.diesel_kw[t]),
            loadServedKW=_val(vars.load_served_kw[t]),
            curtailedRenewableKW=_val(vars.curtailed_kw[t]),
        )
        schedule.append(interval.to_dict())

    return schedule


def calculate_schedule_metrics(
    schedule: List[Dict[str, Any]],
    inputs: OptimizationInputs,
    battery_deg_cost_per_kwh: float = 0.015,
    unserved_energy_penalty: float = 10000.0,
) -> OptimizationMetrics:
    """Calculate the canonical outcome metrics from the dispatch schedule."""
    dt = inputs.dt_hours
    total_diesel_kwh = sum(item["dieselKW"] * dt for item in schedule)
    diesel_liters = total_diesel_kwh * inputs.diesel_liters_per_kwh
    total_co2_kg = diesel_liters * 2.68

    fuel_cost = diesel_liters * inputs.diesel_fuel_price_per_liter

    total_charge_kwh = sum(item["batteryChargeKW"] * dt for item in schedule)
    total_discharge_kwh = sum(item["batteryDischargeKW"] * dt for item in schedule)
    battery_throughput = calculate_battery_throughput(total_charge_kwh, total_discharge_kwh)
    degradation_cost = battery_throughput * battery_deg_cost_per_kwh

    total_demand_kwh = sum(dem * dt for dem in inputs.demand_kw)
    total_served_kwh = sum(item["loadServedKW"] * dt for item in schedule)
    unserved_kwh = max(0.0, total_demand_kwh - total_served_kwh)
    unserved_cost = unserved_kwh * unserved_energy_penalty

    total_cost = fuel_cost + degradation_cost + unserved_cost

    total_solar_used_kwh = sum(item["solarKW"] * dt for item in schedule)
    total_wind_used_kwh = sum(item["windKW"] * dt for item in schedule)
    total_renewable_used_kwh = total_solar_used_kwh + total_wind_used_kwh

    total_solar_avail_kwh = sum(s * dt for s in inputs.solar_available_kw)
    total_wind_avail_kwh = sum(w * dt for w in inputs.wind_available_kw)
    total_renewable_avail_kwh = total_solar_avail_kwh + total_wind_avail_kwh

    if total_renewable_avail_kwh > 1e-6:
        renewable_utilization = (total_renewable_used_kwh / total_renewable_avail_kwh) * 100.0
        renewable_utilization = min(100.0, max(0.0, renewable_utilization))
    else:
        renewable_utilization = 100.0

    return OptimizationMetrics(
        totalCost=total_cost,
        totalCO2Kg=total_co2_kg,
        dieselLiters=diesel_liters,
        renewableUtilizationPercent=renewable_utilization,
        unservedEnergyKWh=unserved_kwh,
    )


def format_canonical_result(
    run_id: str,
    community_id: str,
    created_at: str,
    status: str,
    schedule: List[Dict[str, Any]],
    metrics: OptimizationMetrics,
    reason_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format the complete canonical optimization response contract."""
    return {
        "runId": run_id,
        "communityId": community_id,
        "createdAt": created_at,
        "status": status,
        "schedule": schedule,
        "metrics": metrics.to_dict(),
        "reasonCodes": reason_codes or [],
    }
