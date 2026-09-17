"""Shortfall Ladder Module for NAVYA-1.

This module implements the deterministic 5-stage shortfall ladder:
1. EARLY_WARNING: Impending shortfall detected; reserves and generation adequate for now.
2. PRE_CHARGE_RESERVE: Imminent shortfall window; pre-charge battery from renewables.
3. RESCHEDULE_FLEXIBLE_DEMAND: Shift deferrable loads to peak solar windows.
4. DIESEL_BACKUP: Dispatch diesel generator to bridge generation deficit.
5. LOAD_SHEDDING: Controlled shedding of non-critical loads (critical loads strictly protected).

Single source of truth: NAVYA-1 Architecture Specification.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from engine.optimizer.core.battery import calculate_available_discharge_energy


class ShortfallStage(str, Enum):
    """The exact 5 shortfall ladder stages defined by NAVYA architecture."""
    NORMAL = "NORMAL"
    EARLY_WARNING = "EARLY_WARNING"
    PRE_CHARGE_RESERVE = "PRE_CHARGE_RESERVE"
    RESCHEDULE_FLEXIBLE_DEMAND = "RESCHEDULE_FLEXIBLE_DEMAND"
    DIESEL_BACKUP = "DIESEL_BACKUP"
    LOAD_SHEDDING = "LOAD_SHEDDING"


STAGE_ORDER = {
    ShortfallStage.NORMAL: 0,
    ShortfallStage.EARLY_WARNING: 1,
    ShortfallStage.PRE_CHARGE_RESERVE: 2,
    ShortfallStage.RESCHEDULE_FLEXIBLE_DEMAND: 3,
    ShortfallStage.DIESEL_BACKUP: 4,
    ShortfallStage.LOAD_SHEDDING: 5,
}


@dataclass
class LadderEvaluationResult:
    """Canonical evaluation result of the shortfall ladder."""
    communityId: str
    stage: ShortfallStage
    stageIndex: int
    shortfallKW: float
    shortfallKWh: float
    criticalLoadKW: float
    criticalLoadProtected: bool
    recommendedActions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "communityId": self.communityId,
            "stage": self.stage.value,
            "stageIndex": self.stageIndex,
            "shortfallKW": round(self.shortfallKW, 2),
            "shortfallKWh": round(self.shortfallKWh, 2),
            "criticalLoadKW": round(self.criticalLoadKW, 2),
            "criticalLoadProtected": self.criticalLoadProtected,
            "recommendedActions": self.recommendedActions,
        }


def evaluate_shortfall_ladder(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically evaluate the shortfall ladder for the current microgrid state and forecast.

    Inputs:
      - communityId: str
      - demand: float or dict or list
      - forecast: points with demandKW, solarAvailableKW, windAvailableKW
      - battery: socPercent, minSocPercent, capacityKWh, maxDischargeKW
      - diesel: available, maxOutputKW, fuelRemainingLiters

    Returns:
      LadderEvaluationResult dictionary.
    """
    community_id = payload.get("communityId", "community-001")

    # 1. Parse battery state
    batt_data = payload.get("battery", {})
    soc_pct = float(batt_data.get("socPercent", payload.get("battery.socPercent", 50.0)))
    min_soc_pct = float(batt_data.get("minSocPercent", payload.get("battery.minSocPercent", 20.0)))
    capacity_kwh = float(batt_data.get("capacityKWh", payload.get("battery.capacityKWh", 100.0)))
    max_discharge_kw = float(batt_data.get("maxDischargeKW", payload.get("battery.maxDischargeKW", 30.0)))
    discharge_eff = float(batt_data.get("dischargeEfficiency", 0.95))

    usable_batt_kwh = calculate_available_discharge_energy(
        current_soc_percent=soc_pct,
        min_soc_percent=min_soc_pct,
        capacity_kwh=capacity_kwh,
        discharge_efficiency=discharge_eff,
    )

    # 2. Parse diesel state
    diesel_data = payload.get("diesel", {})
    diesel_avail = bool(diesel_data.get("available", payload.get("diesel.available", True)))
    diesel_max_kw = float(diesel_data.get("maxOutputKW", payload.get("diesel.maxOutputKW", 50.0)))
    diesel_fuel = float(diesel_data.get("fuelRemainingLiters", payload.get("diesel.fuelRemainingLiters", 200.0)))
    liters_per_kwh = float(diesel_data.get("litersPerKWh", 0.27))

    diesel_max_effective_kw = diesel_max_kw if (diesel_avail and diesel_fuel > 0) else 0.0
    diesel_energy_available_kwh = (
        (diesel_fuel / liters_per_kwh) if (diesel_avail and liters_per_kwh > 0) else 0.0
    )

    # 3. Parse forecast points / current demand
    raw_points = []
    if "forecast" in payload and isinstance(payload["forecast"], dict):
        raw_points = payload["forecast"].get("points", [])
    elif "points" in payload and isinstance(payload["points"], list):
        raw_points = payload["points"]

    dt_hours = float(payload.get("intervalMinutes", 60)) / 60.0

    demand_series: List[float] = []
    renewables_series: List[float] = []
    critical_series: List[float] = []

    if raw_points:
        for pt in raw_points:
            dem = float(pt.get("demandKW", pt.get("demand", {}).get("totalKW", 0.0)))
            sol = float(pt.get("solarAvailableKW", pt.get("solar", {}).get("availableKW", 0.0)))
            win = float(pt.get("windAvailableKW", pt.get("wind", {}).get("availableKW", 0.0)))
            crit = float(pt.get("criticalLoadKW", dem * 0.40))
            demand_series.append(dem)
            renewables_series.append(sol + win)
            critical_series.append(crit)
    else:
        dem = float(payload.get("demand", {}).get("totalKW", payload.get("demand.totalKW", 50.0)))
        sol = float(payload.get("solar", {}).get("availableKW", payload.get("solar.availableKW", 20.0)))
        win = float(payload.get("wind", {}).get("availableKW", payload.get("wind.availableKW", 10.0)))
        crit = float(payload.get("criticalLoadKW", dem * 0.40))
        demand_series.append(dem)
        renewables_series.append(sol + win)
        critical_series.append(crit)

    n_steps = len(demand_series)
    total_horizon_hours = n_steps * dt_hours

    total_demand_kwh = sum(demand_series) * dt_hours
    total_renewables_kwh = sum(renewables_series) * dt_hours
    total_critical_kwh = sum(critical_series) * dt_hours
    peak_demand_kw = max(demand_series) if demand_series else 0.0
    peak_critical_kw = max(critical_series) if critical_series else 0.0

    # Deficit before any dispatch
    gross_energy_deficit_kwh = max(0.0, total_demand_kwh - total_renewables_kwh)

    # Net deficit after usable battery
    net_energy_deficit_kwh = max(0.0, gross_energy_deficit_kwh - usable_batt_kwh)

    # Peak instantaneous deficit
    peak_instant_deficit_kw = max(
        0.0,
        max(
            (demand_series[i] - renewables_series[i] - min(max_discharge_kw, usable_batt_kwh / dt_hours))
            for i in range(n_steps)
        )
    )

    # Flexible load portion (assumed ~20% of non-critical load can be rescheduled)
    flexible_kwh = (total_demand_kwh - total_critical_kwh) * 0.20

    # Total physical capacity across the horizon
    total_physical_energy_kwh = (
        total_renewables_kwh
        + usable_batt_kwh
        + min(diesel_max_effective_kw * total_horizon_hours, diesel_energy_available_kwh)
    )
    max_instant_power_kw = (
        max(renewables_series) + max_discharge_kw + diesel_max_effective_kw
    )

    # 4. Stage Determination
    if net_energy_deficit_kwh <= 0.0 and peak_instant_deficit_kw <= 0.0:
        stage = ShortfallStage.NORMAL
        actions = ["System operating normally. Renewables and stored reserves cover projected demand."]
        crit_protected = True

    elif total_demand_kwh > total_physical_energy_kwh or peak_demand_kw > max_instant_power_kw:
        # Physical capacity strictly insufficient -> Stage 5: LOAD_SHEDDING
        stage = ShortfallStage.LOAD_SHEDDING
        crit_protected = (
            total_physical_energy_kwh >= total_critical_kwh
            and max_instant_power_kw >= peak_critical_kw
        )
        actions = [
            "Initiate controlled load shedding for non-critical loads.",
            "Protect critical health, water, and emergency infrastructure.",
            "Alert community operators of generation shortfall.",
        ]

    elif net_energy_deficit_kwh > flexible_kwh:
        # Deficit exceeds flexible demand rescheduling -> Stage 4: DIESEL_BACKUP
        stage = ShortfallStage.DIESEL_BACKUP
        crit_protected = True
        actions = [
            "Dispatch backup diesel generator to bridge generation deficit.",
            "Monitor diesel fuel inventory and operating runtime.",
            "Protect critical loads with dedicated generator output.",
        ]

    elif gross_energy_deficit_kwh > 0.0 and net_energy_deficit_kwh <= flexible_kwh:
        # Deficit can be covered by shifting flexible loads -> Stage 3: RESCHEDULE_FLEXIBLE_DEMAND
        stage = ShortfallStage.RESCHEDULE_FLEXIBLE_DEMAND
        crit_protected = True
        actions = [
            "Reschedule flexible and deferrable loads (water pumping, EV charging).",
            "Shift discretionary consumption to peak renewable generation hours.",
            "Conserve battery energy for non-flexible and critical demand.",
        ]

    elif soc_pct < 40.0 and any(renewables_series[i] > demand_series[i] for i in range(n_steps)):
        # Shortfall risk imminent, battery needs pre-charging -> Stage 2: PRE_CHARGE_RESERVE
        stage = ShortfallStage.PRE_CHARGE_RESERVE
        crit_protected = True
        actions = [
            "Pre-charge battery storage immediately during upcoming solar/wind surplus.",
            "Raise battery reserve threshold to ensure evening headroom.",
            "Avoid discretionary battery discharge until peak shortfall window.",
        ]

    else:
        # Impending deficit identified in forecast -> Stage 1: EARLY_WARNING
        stage = ShortfallStage.EARLY_WARNING
        crit_protected = True
        actions = [
            "Issue early shortfall advisory to microgrid operations.",
            "Monitor weather forecasts and renewable generation trends.",
            "Verify diesel generator availability and fuel readiness.",
        ]

    result = LadderEvaluationResult(
        communityId=community_id,
        stage=stage,
        stageIndex=STAGE_ORDER[stage],
        shortfallKW=peak_instant_deficit_kw,
        shortfallKWh=net_energy_deficit_kwh,
        criticalLoadKW=peak_critical_kw,
        criticalLoadProtected=crit_protected,
        recommendedActions=actions,
    )
    return result.to_dict()
