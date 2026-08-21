"""
AstraGuard 2.4 — ASQD Multi-Device Physics Models
===================================================
Provides device-specific parametric drift trajectory generators with normalized thermal scaling:
  - DIGITAL_IC: Arrhenius IDDQ drift relative to T_ref = 25°C (298.15 K)
  - MEMS_GYROSCOPE: Viscoelastic zero-rate-offset drift + stiction risk
  - IMAGE_SENSOR: Shockley-Read-Hall dark current trap generation relative to T_ref = 25°C

Evidence:
  DIGITAL_IC model: [PE] Arrhenius I ∝ exp(-Ea/k * (1/T - 1/T_ref)); Ea=0.68 eV.
  MEMS model: [PE] Viscoelastic creep/stress relaxation; ZRO ∝ log(t) + spring constant drift.
  Image Sensor model: [PE] SRH generation: dark current thermal scaling relative to 25°C.
"""

import numpy as np


# ===========================================================================
# DIGITAL IC — Arrhenius IDDQ Drift (Normalized to T_ref = 25°C)
# ===========================================================================
class ArrheniusIDDQModel:
    """[PE] Arrhenius CMOS IDDQ model with controlled process variation."""
    def __init__(self, base_iddq_uA: float = 1.2, ea_eV: float = 0.68, aging_rate: float = 0.001):
        self.base_iddq = base_iddq_uA
        self.ea = ea_eV
        self.k_boltzmann = 8.617333262145e-5  # eV/K [PE]
        self.aging_rate = aging_rate
        self.t_ref_k = 298.15  # 25°C reference temperature

    def generate_trajectory(
        self,
        num_hours: int = 168,
        profile: str = "NOMINAL",
        comp_base_scale: float = 1.0,
        lot_base_scale: float = 1.0
    ):
        trajectory = []
        effective_base = self.base_iddq * comp_base_scale * lot_base_scale

        # Component-specific aging kinetic rate variation [SA]
        rate_var = np.random.normal(1.0, 0.08)

        for hour in range(num_hours + 1):
            temp_c = 25.0 + 100.0 * (1.0 - np.exp(-hour / 2.0))
            temp_k = temp_c + 273.15
            # Arrhenius scaling relative to 25°C [PE]
            arrhenius_factor = np.exp(-(self.ea / self.k_boltzmann) * ((1.0 / temp_k) - (1.0 / self.t_ref_k)))
            aging_component = self.aging_rate * hour * rate_var

            if profile == "NOMINAL":
                iddq = effective_base * (arrhenius_factor + aging_component * 0.5)
            elif profile == "THERMAL_RUNAWAY":
                if hour >= 24:
                    runaway = 0.5 * np.exp(0.04 * (hour - 24))
                    iddq = effective_base * (arrhenius_factor + aging_component) + runaway
                else:
                    iddq = effective_base * (arrhenius_factor + aging_component * 0.5)
            elif profile == "ELECTROMIGRATION":
                accel = 1.0 + (hour / 168.0) ** 2 * 35.0
                iddq = effective_base * (arrhenius_factor + aging_component * accel)
            elif profile == "SPATIAL_OUTLIER":
                iddq = effective_base * 4.5 * (arrhenius_factor + aging_component * 0.3)
            elif profile == "DIELECTRIC_OSCILLATION":
                osc = 0.25 * np.sin(2 * np.pi * hour / 12.0)
                iddq = effective_base * (arrhenius_factor + aging_component * 0.5) + osc
            else:
                iddq = effective_base * (arrhenius_factor + aging_component * 0.5)

            noise = np.random.normal(0, 0.015 * (1.0 + hour / 168.0))
            iddq = max(0.01, iddq + noise)
            trajectory.append(round(float(iddq), 4))
        return trajectory


# ===========================================================================
# MEMS GYROSCOPE — Viscoelastic ZRO Drift Model
# ===========================================================================
class ViscoelasticMEMSModel:
    """
    [PE] MEMS zero-rate-offset (ZRO) drift model based on viscoelastic
    stress relaxation in die attach and packaging materials.
    """
    def __init__(
        self,
        base_zro_dps: float = 0.05,
        relaxation_amplitude_dps: float = 0.03,
        relaxation_tau_hours: float = 20.0,
        creep_rate_dps: float = 0.008,
        creep_t0_hours: float = 5.0,
    ):
        self.base_zro = base_zro_dps
        self.A = relaxation_amplitude_dps
        self.tau = relaxation_tau_hours
        self.B = creep_rate_dps
        self.t0 = creep_t0_hours

    def generate_trajectory(
        self,
        num_hours: int = 168,
        profile: str = "NOMINAL",
        comp_base_scale: float = 1.0,
        lot_base_scale: float = 1.0
    ) -> list:
        trajectory = []
        # [SA] MEMS baseline is precision-stabilised: tight process variation (±3%).
        # Large comp_base_scale from lot_generator is damped here to prevent
        # baseline scatter from exceeding the physical drift signal.
        damped_base_scale = 1.0 + (comp_base_scale - 1.0) * 0.25  # compress to ±3%
        effective_base = self.base_zro * damped_base_scale * lot_base_scale
        # [SA] Kinetic rate variation drives 168h signal discriminability
        rate_var = np.random.normal(1.0, 0.15)  # ±15% rate dispersion

        for hour in range(num_hours + 1):
            relaxation = self.A * rate_var * (1.0 - np.exp(-hour / self.tau))
            creep = self.B * rate_var * np.log(1.0 + hour / self.t0)

            zro = effective_base + relaxation + creep

            if profile == "NOMINAL":
                pass
            elif profile == "MEMS_STICTION_ONSET":
                if hour >= 48:
                    zro += 0.12 * (1.0 + (hour - 48) / 120.0)
            elif profile == "PACKAGING_STRESS_RELAXATION":
                zro = effective_base + 2.5 * self.A * rate_var * (1.0 - np.exp(-hour / (self.tau * 0.4)))
            elif profile == "SPATIAL_OUTLIER":
                zro = effective_base * 4.5 + relaxation * 0.5 + creep * 0.5
            elif profile == "THERMAL_RUNAWAY":
                if hour >= 24:
                    zro += 0.002 * rate_var * (hour - 24)

            noise = np.random.normal(0, 0.0008)  # tighter measurement noise [SA]
            zro = max(0.0, zro + noise)
            trajectory.append(round(float(zro), 5))
        return trajectory


# ===========================================================================
# IMAGE SENSOR — Shockley-Read-Hall Dark Current Model
# ===========================================================================
class SRHDarkCurrentModel:
    """
    [PE] CMOS Image Sensor dark current model based on Shockley-Read-Hall (SRH)
    thermal generation in depletion region.
    """
    def __init__(
        self,
        base_dark_current_nA_cm2: float = 1.5,
        stress_temp_c: float = 60.0,
        activation_energy_eV: float = 0.55,
    ):
        self.base_dc = base_dark_current_nA_cm2
        self.T_stress_K = stress_temp_c + 273.15
        self.Ea = activation_energy_eV
        self.k_b = 8.617333262145e-5  # eV/K [PE]

    def _thermal_factor(self, temp_k: float) -> float:
        T_ref_K = 298.15
        return np.exp(-(self.Ea / self.k_b) * ((1.0 / temp_k) - (1.0 / T_ref_K)))

    def generate_trajectory(
        self,
        num_hours: int = 168,
        profile: str = "NOMINAL",
        comp_base_scale: float = 1.0,
        lot_base_scale: float = 1.0
    ) -> list:
        trajectory = []
        # [SA] Component-level process variation damped for baseline stability
        damped_base_scale = 1.0 + (comp_base_scale - 1.0) * 0.30  # compress to ~±4%
        effective_base = self.base_dc * damped_base_scale * lot_base_scale
        thermal_factor = self._thermal_factor(self.T_stress_K)
        # [PE] SRH trap generation follows exponential kinetics: I_dark(t) = I_0 * exp(gamma * t)
        # Calibrated so 168h value is 5-15x the 0h baseline for nominal devices.
        gamma = np.random.uniform(0.009, 0.018)  # trap accumulation rate [PE]

        for hour in range(num_hours + 1):
            # Exponential SRH trap growth: physically motivated defect accumulation [PE]
            trap_growth = np.exp(gamma * hour)
            dc = effective_base * thermal_factor * trap_growth

            if profile == "NOMINAL":
                pass
            elif profile == "DARK_CURRENT_SPIKE_GROWTH":
                # Accelerated trap generation cluster above 48h [PE]
                if hour >= 48:
                    dc *= np.exp(0.008 * (hour - 48))
            elif profile == "THERMAL_RUNAWAY":
                if hour >= 24:
                    dc *= np.exp(0.016 * (hour - 24))
            elif profile == "SPATIAL_OUTLIER":
                dc = effective_base * thermal_factor * trap_growth * 3.5
            elif profile == "DIELECTRIC_OSCILLATION":
                osc = 0.6 * np.sin(2 * np.pi * hour / 24.0)
                dc = max(0.05, dc + osc)

            # [SA] Measurement noise proportional to signal level (Poisson-like)
            noise = np.random.normal(0, 0.02 * dc)
            dc = max(0.01, dc + noise)
            trajectory.append(round(float(dc), 4))
        return trajectory
