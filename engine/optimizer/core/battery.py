"""Battery Modeling and Mathematical Logic for NAVYA-1.

This module provides physical and mathematical calculations for battery energy storage:
- BatteryState canonical representation
- State-of-charge (SOC) propagation
- Usable discharge energy computation respecting minimum SOC
- Headroom-constrained charge power calculation
- Energy-constrained discharge power calculation
- Cycle and throughput wear estimation

Single source of truth: NAVYA-1 Architecture Specification.
"""

from dataclasses import dataclass


@dataclass
class BatteryState:
    """Canonical representation of battery energy storage parameters."""
    socPercent: float
    capacityKWh: float
    maxChargeKW: float
    maxDischargeKW: float
    minSocPercent: float = 20.0
    chargeEfficiency: float = 0.95
    dischargeEfficiency: float = 0.95


def calculate_next_soc(
    current_soc_percent: float,
    capacity_kwh: float,
    charge_kw: float,
    discharge_kw: float,
    dt_hours: float = 1.0,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
) -> float:
    """Calculate the next battery SOC percentage after a time step dt.

    Formula:
      Delta_E = (charge_kw * eta_ch - discharge_kw / eta_dis) * dt
      Next_E = current_E + Delta_E
      Next_SOC = (Next_E / capacity_kwh) * 100.0
    """
    if capacity_kwh <= 0.0:
        return 0.0

    current_energy_kwh = (current_soc_percent / 100.0) * capacity_kwh
    net_energy_delta = (
        charge_kw * charge_efficiency - (discharge_kw / max(1e-6, discharge_efficiency))
    ) * dt_hours

    next_energy_kwh = current_energy_kwh + net_energy_delta
    next_soc = (next_energy_kwh / capacity_kwh) * 100.0
    return max(0.0, min(100.0, next_soc))


def calculate_available_discharge_energy(
    current_soc_percent: float,
    min_soc_percent: float,
    capacity_kwh: float,
    discharge_efficiency: float = 0.95,
) -> float:
    """Calculate delivered electrical energy (kWh) available above minimum SOC.

    Formula:
      Usable_Internal_kWh = max(0, (current_soc - min_soc) / 100.0 * capacity_kwh)
      Delivered_kWh = Usable_Internal_kWh * discharge_efficiency
    """
    if current_soc_percent <= min_soc_percent or capacity_kwh <= 0.0:
        return 0.0

    usable_internal_kwh = ((current_soc_percent - min_soc_percent) / 100.0) * capacity_kwh
    return max(0.0, usable_internal_kwh * discharge_efficiency)


def calculate_max_charge_power(
    current_soc_percent: float,
    capacity_kwh: float,
    max_charge_kw: float,
    dt_hours: float = 1.0,
    charge_efficiency: float = 0.95,
) -> float:
    """Calculate the maximum power (kW) the battery can accept during interval dt.

    Constrained by both converter maximum power rating and remaining capacity headroom:
      Headroom_kWh = max(0, (100.0 - current_soc) / 100.0 * capacity_kwh)
      Power_Headroom = Headroom_kWh / (charge_efficiency * dt)
      Max_P_ch = min(max_charge_kw, Power_Headroom)
    """
    if capacity_kwh <= 0.0 or current_soc_percent >= 100.0 or dt_hours <= 0.0:
        return 0.0

    headroom_kwh = ((100.0 - current_soc_percent) / 100.0) * capacity_kwh
    power_headroom = headroom_kwh / (max(1e-6, charge_efficiency) * dt_hours)
    return max(0.0, min(max_charge_kw, power_headroom))


def calculate_max_discharge_power(
    current_soc_percent: float,
    min_soc_percent: float,
    capacity_kwh: float,
    max_discharge_kw: float,
    dt_hours: float = 1.0,
    discharge_efficiency: float = 0.95,
) -> float:
    """Calculate the maximum power (kW) the battery can deliver during interval dt.

    Constrained by converter maximum power rating and available energy above min SOC:
      Deliverable_kWh = Available_Discharge_Energy(current_soc, min_soc, capacity, eta_dis)
      Power_Limit = Deliverable_kWh / dt
      Max_P_dis = min(max_discharge_kw, Power_Limit)
    """
    if current_soc_percent <= min_soc_percent or capacity_kwh <= 0.0 or dt_hours <= 0.0:
        return 0.0

    available_kwh = calculate_available_discharge_energy(
        current_soc_percent, min_soc_percent, capacity_kwh, discharge_efficiency
    )
    power_limit = available_kwh / dt_hours
    return max(0.0, min(max_discharge_kw, power_limit))


def calculate_battery_throughput(
    charge_kwh: float,
    discharge_kwh: float,
) -> float:
    """Calculate total energy throughput (kWh) contributing to degradation."""
    return max(0.0, charge_kwh) + max(0.0, discharge_kwh)
