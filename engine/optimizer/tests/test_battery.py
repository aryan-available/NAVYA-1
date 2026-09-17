"""Pytest Suite for Battery Logic (engine/optimizer/tests/test_battery.py).

Tests:
1. Battery SOC propagation (charging, discharging, efficiencies, clipping)
2. Available discharge energy (min SOC, capacity, discharge efficiency, edge cases)
3. Maximum charge power (inverter limits, headroom limits, 100% ceiling)
4. Maximum discharge power (inverter limits, usable energy limits, min SOC floor)
5. Battery throughput and degradation accounting

Single source of truth: NAVYA-1 Architecture Specification.
"""

import pytest
from engine.optimizer.core.battery import (
    BatteryState,
    calculate_next_soc,
    calculate_available_discharge_energy,
    calculate_max_charge_power,
    calculate_max_discharge_power,
    calculate_battery_throughput,
)


def test_battery_soc_propagation():
    """Verify SOC update math for charging, discharging, and efficiency losses."""
    capacity = 100.0  # kWh
    initial_soc = 50.0  # 50 kWh

    # 1. Charging with 95% efficiency: + (20 kW * 0.95) = + 19 kWh -> 69 kWh (69%)
    soc_ch = calculate_next_soc(
        current_soc_percent=initial_soc,
        capacity_kwh=capacity,
        charge_kw=20.0,
        discharge_kw=0.0,
        dt_hours=1.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    assert soc_ch == pytest.approx(69.0, 0.01)

    # 2. Discharging with 95% efficiency: - (19 kW / 0.95) = - 20 kWh -> 30 kWh (30%)
    soc_dis = calculate_next_soc(
        current_soc_percent=initial_soc,
        capacity_kwh=capacity,
        charge_kw=0.0,
        discharge_kw=19.0,
        dt_hours=1.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    assert soc_dis == pytest.approx(30.0, 0.01)

    # 3. Overcharge clipping: cannot exceed 100%
    soc_over = calculate_next_soc(
        current_soc_percent=95.0,
        capacity_kwh=capacity,
        charge_kw=50.0,
        discharge_kw=0.0,
        dt_hours=1.0,
    )
    assert soc_over == 100.0

    # 4. Overdischarge clipping: cannot fall below 0%
    soc_under = calculate_next_soc(
        current_soc_percent=5.0,
        capacity_kwh=capacity,
        charge_kw=0.0,
        discharge_kw=50.0,
        dt_hours=1.0,
    )
    assert soc_under == 0.0


def test_available_discharge_energy():
    """Verify deliverable energy calculations respect minSocPercent, capacity, and efficiency."""
    # 1. Normal state: 60% SOC, 20% min SOC -> 40 internal kWh * 0.95 = 38.0 kWh
    avail = calculate_available_discharge_energy(
        current_soc_percent=60.0,
        min_soc_percent=20.0,
        capacity_kwh=100.0,
        discharge_efficiency=0.95,
    )
    assert avail == pytest.approx(38.0, 0.01)

    # 2. At min SOC: usable energy must be exactly 0.0
    at_min = calculate_available_discharge_energy(
        current_soc_percent=20.0,
        min_soc_percent=20.0,
        capacity_kwh=100.0,
        discharge_efficiency=0.95,
    )
    assert at_min == 0.0

    # 3. Below min SOC: usable energy must be 0.0
    below_min = calculate_available_discharge_energy(
        current_soc_percent=15.0,
        min_soc_percent=20.0,
        capacity_kwh=100.0,
        discharge_efficiency=0.95,
    )
    assert below_min == 0.0

    # 4. Zero capacity
    zero_cap = calculate_available_discharge_energy(
        current_soc_percent=80.0,
        min_soc_percent=20.0,
        capacity_kwh=0.0,
    )
    assert zero_cap == 0.0


def test_max_charge_power():
    """Verify maximum charge rate respects inverter capacity and remaining headroom."""
    # 1. Headroom is large (50 kWh): constrained by maxChargeKW (25 kW)
    p_ch1 = calculate_max_charge_power(
        current_soc_percent=50.0,
        capacity_kwh=100.0,
        max_charge_kw=25.0,
        dt_hours=1.0,
        charge_efficiency=0.95,
    )
    assert p_ch1 == 25.0

    # 2. Headroom is small (5 kWh): constrained by headroom (5 / 0.95 = 5.263 kW < 25 kW)
    p_ch2 = calculate_max_charge_power(
        current_soc_percent=95.0,
        capacity_kwh=100.0,
        max_charge_kw=25.0,
        dt_hours=1.0,
        charge_efficiency=0.95,
    )
    assert p_ch2 == pytest.approx(5.0 / 0.95, 0.01)

    # 3. Battery fully charged (100%): charge power must be 0.0
    p_ch3 = calculate_max_charge_power(
        current_soc_percent=100.0,
        capacity_kwh=100.0,
        max_charge_kw=25.0,
        dt_hours=1.0,
    )
    assert p_ch3 == 0.0


def test_max_discharge_power():
    """Verify maximum discharge rate respects inverter capacity and energy above minSocPercent."""
    # 1. Usable energy is large (38 kWh): constrained by maxDischargeKW (25 kW)
    p_dis1 = calculate_max_discharge_power(
        current_soc_percent=60.0,
        min_soc_percent=20.0,
        capacity_kwh=100.0,
        max_discharge_kw=25.0,
        dt_hours=1.0,
        discharge_efficiency=0.95,
    )
    assert p_dis1 == 25.0

    # 2. Usable energy is small (2 internal kWh * 0.95 = 1.9 kWh): constrained by energy (1.9 kW < 25 kW)
    p_dis2 = calculate_max_discharge_power(
        current_soc_percent=22.0,
        min_soc_percent=20.0,
        capacity_kwh=100.0,
        max_discharge_kw=25.0,
        dt_hours=1.0,
        discharge_efficiency=0.95,
    )
    assert p_dis2 == pytest.approx(1.9, 0.01)

    # 3. At min SOC: discharge power must be 0.0
    p_dis3 = calculate_max_discharge_power(
        current_soc_percent=20.0,
        min_soc_percent=20.0,
        capacity_kwh=100.0,
        max_discharge_kw=25.0,
        dt_hours=1.0,
    )
    assert p_dis3 == 0.0


def test_battery_throughput_and_degradation():
    """Verify battery throughput calculation sums charge and discharge energy."""
    tp = calculate_battery_throughput(charge_kwh=12.5, discharge_kwh=17.5)
    assert tp == 30.0

    # Negative inputs should be safely clamped to zero
    tp_neg = calculate_battery_throughput(charge_kwh=-5.0, discharge_kwh=10.0)
    assert tp_neg == 10.0
