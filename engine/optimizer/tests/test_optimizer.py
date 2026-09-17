"""Pytest Suite for Central Optimizer (engine/optimizer/tests/test_optimizer.py).

Tests:
1. Pure renewable dispatch (zero diesel, zero unserved load)
2. Battery cycling (charge during surplus, discharge during deficit, min SOC respected)
3. Diesel backup (dispatched when renewables + battery insufficient, fuel limits enforced)
4. Unserved energy penalty (unserved load only occurs when physically impossible to serve)
5. Canonical response format (exact top-level, schedule, and metrics contracts)
6. Multi-period continuous horizon behavior

Single source of truth: NAVYA-1 Architecture Specification.
"""

import pytest
from engine.optimizer.core.optimizer import run_optimization


def test_pure_renewable_dispatch():
    """Verify that when renewable availability exceeds demand, zero diesel and zero unserved energy occur."""
    payload = {
        "communityId": "community-001",
        "battery": {
            "socPercent": 50.0,
            "capacityKWh": 100.0,
            "maxChargeKW": 30.0,
            "maxDischargeKW": 30.0,
            "minSocPercent": 20.0,
        },
        "diesel": {
            "available": True,
            "maxOutputKW": 50.0,
            "fuelRemainingLiters": 100.0,
            "fuelPricePerLiter": 1.50,
        },
        "forecast": {
            "points": [
                {
                    "timestamp": "2026-09-17T12:00:00Z",
                    "solarAvailableKW": 60.0,
                    "windAvailableKW": 20.0,
                    "demandKW": 50.0,
                }
            ]
        },
    }

    result = run_optimization(payload)
    assert result["status"] == "OPTIMAL"
    schedule_item = result["schedule"][0]
    assert schedule_item["dieselKW"] == 0.0
    assert schedule_item["loadServedKW"] == 50.0
    assert result["metrics"]["unservedEnergyKWh"] == 0.0
    assert result["metrics"]["dieselLiters"] == 0.0


def test_battery_cycling():
    """Verify battery charges during renewable surplus and discharges during deficit without violating min SOC."""
    payload = {
        "communityId": "community-001",
        "battery": {
            "socPercent": 20.0,
            "capacityKWh": 100.0,
            "maxChargeKW": 25.0,
            "maxDischargeKW": 25.0,
            "minSocPercent": 20.0,
        },
        "diesel": {
            "available": True,
            "maxOutputKW": 50.0,
            "fuelRemainingLiters": 100.0,
            "fuelPricePerLiter": 1.50,
        },
        "forecast": {
            "points": [
                # Timestep 0: Surplus renewable -> battery should charge
                {
                    "timestamp": "2026-09-17T12:00:00Z",
                    "solarAvailableKW": 60.0,
                    "windAvailableKW": 0.0,
                    "demandKW": 40.0,
                },
                # Timestep 1: Deficit renewable -> battery should discharge
                {
                    "timestamp": "2026-09-17T13:00:00Z",
                    "solarAvailableKW": 0.0,
                    "windAvailableKW": 0.0,
                    "demandKW": 20.0,
                },
            ]
        },
    }

    result = run_optimization(payload)
    assert result["status"] == "OPTIMAL"
    t0 = result["schedule"][0]
    t1 = result["schedule"][1]

    assert t0["batteryChargeKW"] > 0.0
    assert t0["batteryDischargeKW"] == 0.0
    assert t1["batteryDischargeKW"] > 0.0
    assert t1["batteryChargeKW"] == 0.0
    assert result["metrics"]["unservedEnergyKWh"] == 0.0


def test_diesel_backup():
    """Verify diesel is dispatched when renewables and battery cannot meet demand, respecting fuel availability."""
    payload = {
        "communityId": "community-001",
        "battery": {
            "socPercent": 20.0,  # Already at min SOC, cannot discharge
            "capacityKWh": 100.0,
            "maxChargeKW": 30.0,
            "maxDischargeKW": 30.0,
            "minSocPercent": 20.0,
        },
        "diesel": {
            "available": True,
            "maxOutputKW": 40.0,
            "fuelRemainingLiters": 50.0,
            "fuelPricePerLiter": 1.50,
        },
        "forecast": {
            "points": [
                {
                    "timestamp": "2026-09-17T20:00:00Z",
                    "solarAvailableKW": 0.0,
                    "windAvailableKW": 0.0,
                    "demandKW": 35.0,
                }
            ]
        },
    }

    result = run_optimization(payload)
    assert result["status"] == "OPTIMAL"
    t0 = result["schedule"][0]
    assert t0["dieselKW"] == pytest.approx(35.0, 0.1)
    assert t0["loadServedKW"] == pytest.approx(35.0, 0.1)
    assert result["metrics"]["dieselLiters"] > 0.0
    assert result["metrics"]["unservedEnergyKWh"] == 0.0


def test_unserved_energy_penalty():
    """Verify that unserved energy occurs ONLY when demand exceeds all physical capacities, and diesel is prioritized over unserved energy."""
    payload = {
        "communityId": "community-001",
        "battery": {
            "socPercent": 20.0,  # Empty usable capacity
            "capacityKWh": 100.0,
            "maxChargeKW": 20.0,
            "maxDischargeKW": 20.0,
            "minSocPercent": 20.0,
        },
        "diesel": {
            "available": True,
            "maxOutputKW": 30.0,
            "fuelRemainingLiters": 100.0,
            "fuelPricePerLiter": 1.50,
        },
        "forecast": {
            "points": [
                {
                    "timestamp": "2026-09-17T21:00:00Z",
                    "solarAvailableKW": 0.0,
                    "windAvailableKW": 0.0,
                    "demandKW": 50.0,  # Exceeds max diesel (30kW) -> 20kW must be unserved
                }
            ]
        },
    }

    result = run_optimization(payload)
    assert result["status"] == "OPTIMAL"
    t0 = result["schedule"][0]
    # Diesel should run at max 30 kW to avoid severe unserved penalty
    assert t0["dieselKW"] == pytest.approx(30.0, 0.1)
    assert t0["loadServedKW"] == pytest.approx(30.0, 0.1)
    assert result["metrics"]["unservedEnergyKWh"] == pytest.approx(20.0, 0.1)
    assert "UNSERVED_ENERGY_SHORTFALL" in result["reasonCodes"]


def test_canonical_response_format():
    """Verify that the optimization result matches the exact canonical schema from the architecture."""
    payload = {
        "communityId": "community-001",
        "battery": {
            "socPercent": 50.0,
            "capacityKWh": 100.0,
            "maxChargeKW": 30.0,
            "maxDischargeKW": 30.0,
            "minSocPercent": 20.0,
        },
        "diesel": {
            "available": True,
            "maxOutputKW": 50.0,
            "fuelRemainingLiters": 100.0,
            "fuelPricePerLiter": 1.50,
        },
        "forecast": {
            "points": [
                {
                    "timestamp": "2026-09-17T13:00:00Z",
                    "solarAvailableKW": 40.0,
                    "windAvailableKW": 10.0,
                    "demandKW": 45.0,
                }
            ]
        },
    }

    result = run_optimization(payload)

    # 1. Top-level keys
    expected_top_keys = {"runId", "communityId", "createdAt", "status", "schedule", "metrics", "reasonCodes"}
    assert set(result.keys()) == expected_top_keys
    assert result["communityId"] == "community-001"
    assert result["status"] == "OPTIMAL"
    assert isinstance(result["reasonCodes"], list)

    # 2. Schedule keys
    expected_schedule_keys = {
        "timestamp",
        "solarKW",
        "windKW",
        "batteryChargeKW",
        "batteryDischargeKW",
        "dieselKW",
        "loadServedKW",
        "curtailedRenewableKW",
    }
    assert len(result["schedule"]) == 1
    assert set(result["schedule"][0].keys()) == expected_schedule_keys

    # 3. Metrics keys
    expected_metric_keys = {
        "totalCost",
        "totalCO2Kg",
        "dieselLiters",
        "renewableUtilizationPercent",
        "unservedEnergyKWh",
    }
    assert set(result["metrics"].keys()) == expected_metric_keys


def test_multi_period_behavior():
    """Verify multi-period continuous horizon optimization across 4 intervals."""
    payload = {
        "communityId": "community-001",
        "battery": {
            "socPercent": 50.0,
            "capacityKWh": 100.0,
            "maxChargeKW": 20.0,
            "maxDischargeKW": 20.0,
            "minSocPercent": 20.0,
        },
        "diesel": {
            "available": True,
            "maxOutputKW": 50.0,
            "fuelRemainingLiters": 200.0,
            "fuelPricePerLiter": 1.50,
        },
        "forecast": {
            "points": [
                {"timestamp": f"2026-09-17T{h:02d}:00:00Z", "solarAvailableKW": 30.0, "windAvailableKW": 10.0, "demandKW": 35.0}
                for h in range(12, 16)
            ]
        },
    }

    result = run_optimization(payload)
    assert result["status"] == "OPTIMAL"
    assert len(result["schedule"]) == 4
    for item in result["schedule"]:
        assert item["loadServedKW"] == pytest.approx(35.0, 0.1)
    assert result["metrics"]["unservedEnergyKWh"] == 0.0
