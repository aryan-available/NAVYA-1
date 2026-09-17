"""FastAPI Router for NAVYA-1 Diesel Runway Forecast.

Endpoint:
  POST /api/v1/engine/runway

Single source of truth: NAVYA-1 Architecture Specification.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Body

from engine.optimizer.core.diesel import calculate_fuel_consumption, calculate_runway_hours

router = APIRouter(tags=["Runway"])


def compute_diesel_runway(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate remaining diesel runway from projected fuel, demand, renewables, and diesel usage."""
    diesel_data = payload.get("diesel", {})
    initial_fuel = float(
        diesel_data.get("fuelRemainingLiters", payload.get("diesel.fuelRemainingLiters", 0.0))
    )
    max_output_kw = float(
        diesel_data.get("maxOutputKW", payload.get("diesel.maxOutputKW", 50.0))
    )
    liters_per_kwh = float(diesel_data.get("litersPerKWh", 0.27))
    diesel_avail = bool(diesel_data.get("available", payload.get("diesel.available", True)))

    dt_hours = float(payload.get("intervalMinutes", 60)) / 60.0

    raw_points = []
    if "forecast" in payload and isinstance(payload["forecast"], dict):
        raw_points = payload["forecast"].get("points", [])
    elif "points" in payload and isinstance(payload["points"], list):
        raw_points = payload["points"]
    elif "schedule" in payload and isinstance(payload["schedule"], list):
        raw_points = payload["schedule"]

    timeline: List[Dict[str, Any]] = []
    current_fuel = initial_fuel
    exhaustion_ts: Optional[str] = None
    accumulated_burn_liters = 0.0
    total_hours_simulated = 0.0
    runway_hours_count = 0.0
    exhausted = False

    if not raw_points:
        # Single point / instant calculation
        dem = float(payload.get("demand", {}).get("totalKW", payload.get("demand.totalKW", 50.0)))
        sol = float(payload.get("solar", {}).get("availableKW", payload.get("solar.availableKW", 0.0)))
        win = float(payload.get("wind", {}).get("availableKW", payload.get("wind.availableKW", 0.0)))
        ts = str(payload.get("timestamp", datetime.now(timezone.utc).isoformat()))

        net_dem = max(0.0, dem - sol - win)
        gen_kw = min(max_output_kw, net_dem) if diesel_avail else 0.0
        burn_rate_hr = gen_kw * liters_per_kwh

        if burn_rate_hr > 0.0 and current_fuel > 0.0:
            runway_hours = current_fuel / burn_rate_hr
            try:
                base_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                exhaust_dt = base_dt + timedelta(hours=runway_hours)
                exhaustion_ts = exhaust_dt.isoformat()
            except Exception:
                exhaustion_ts = None
        elif current_fuel <= 0.0:
            runway_hours = 0.0
            exhaustion_ts = ts
        else:
            runway_hours = 999.0
            exhaustion_ts = None

        return {
            "runwayHours": round(runway_hours, 2),
            "fuelRemainingLiters": round(current_fuel, 2),
            "burnRateLitersPerHour": round(burn_rate_hr, 2),
            "exhaustionTimestamp": exhaustion_ts or "NONE",
            "timeline": [],
        }

    # Forecast timeline simulation
    for pt in raw_points:
        ts = str(pt.get("timestamp", datetime.now(timezone.utc).isoformat()))
        dem = float(pt.get("demandKW", pt.get("demand", {}).get("totalKW", 0.0)))
        sol = float(pt.get("solarAvailableKW", pt.get("solar", {}).get("availableKW", pt.get("solarKW", 0.0))))
        win = float(pt.get("windAvailableKW", pt.get("wind", {}).get("availableKW", pt.get("windKW", 0.0))))

        # If dieselKW is explicitly in point (e.g. from optimizer schedule), use it; else estimate net demand
        if "dieselKW" in pt:
            diesel_kw = float(pt["dieselKW"])
        else:
            net_dem = max(0.0, dem - sol - win)
            diesel_kw = min(max_output_kw, net_dem) if diesel_avail else 0.0

        step_burn_liters = calculate_fuel_consumption(diesel_kw, dt_hours, liters_per_kwh)

        if not exhausted:
            if current_fuel >= step_burn_liters:
                current_fuel -= step_burn_liters
                accumulated_burn_liters += step_burn_liters
                runway_hours_count += dt_hours
            else:
                fraction = current_fuel / max(1e-6, step_burn_liters)
                runway_hours_count += fraction * dt_hours
                current_fuel = 0.0
                exhausted = True
                try:
                    base_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    exhaust_dt = base_dt + timedelta(hours=fraction * dt_hours)
                    exhaustion_ts = exhaust_dt.isoformat()
                except Exception:
                    exhaustion_ts = ts
        else:
            current_fuel = 0.0

        total_hours_simulated += dt_hours
        timeline.append({
            "timestamp": ts,
            "projectedDemandKW": round(dem, 2),
            "projectedRenewablesKW": round(sol + win, 2),
            "projectedDieselKW": round(diesel_kw, 2),
            "projectedFuelBurnLiters": round(step_burn_liters, 2),
            "projectedFuelRemainingLiters": round(current_fuel, 2),
        })

    avg_burn_rate_hr = (
        (accumulated_burn_liters / total_hours_simulated) if total_hours_simulated > 0 else 0.0
    )

    if not exhausted:
        if current_fuel > 0.0 and avg_burn_rate_hr > 0.0:
            extra_hours = current_fuel / avg_burn_rate_hr
            runway_hours_count += extra_hours
            last_ts = timeline[-1]["timestamp"] if timeline else datetime.now(timezone.utc).isoformat()
            try:
                base_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                exhaust_dt = base_dt + timedelta(hours=extra_hours)
                exhaustion_ts = exhaust_dt.isoformat()
            except Exception:
                exhaustion_ts = None
        elif current_fuel <= 0.0:
            runway_hours_count = 0.0
        else:
            runway_hours_count = 999.0
            exhaustion_ts = None

    return {
        "runwayHours": round(runway_hours_count, 2),
        "fuelRemainingLiters": round(initial_fuel, 2),
        "burnRateLitersPerHour": round(avg_burn_rate_hr, 2),
        "exhaustionTimestamp": exhaustion_ts or "NONE",
        "timeline": timeline,
    }


@router.post("/api/v1/engine/runway")
@router.post("/runway")
async def get_diesel_runway(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Calculate and return projected diesel fuel runway and timeline.

    Uses projected fuel inventory, demand, renewables, and diesel consumption
    to compute runway hours and fuel exhaustion trajectory.
    """
    try:
        result = compute_diesel_runway(payload)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid runway calculation request: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Runway calculation error: {e}")
