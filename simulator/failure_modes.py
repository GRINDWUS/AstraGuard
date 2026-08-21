"""
AstraGuard 2.1 — Six Failure Mode Injection Engine
===================================================
Scientific failure mode archetypes:

  Class 0 — NOMINAL:         Stable Arrhenius drift, stationary noise.
  Class 1 — THERMAL_RUNAWAY: Electromigration-driven exponential I_DDQ surge after onset.
  Class 2 — MEMS_MICROCRACK: Crystal lattice fatigue → random-walk V_offset + transient spikes.
  Class 3 — FREAK_OUTLIER:   Hour-0 baseline anomaly (Modified Z > 3.5σ from lot population).
  Class 4 — INTERMITTENT:    Transient current spikes at irregular intervals (noise-like pattern).
  Class 5 — SENSOR_FAULT:    Measurement instrument anomaly — channel freezes at a constant value.

Classes 4 and 5 added per architectural review:
  - Class 4 tests whether AstraGuard distinguishes physical intermittent degradation from noise.
  - Class 5 tests whether AstraGuard identifies a broken measurement channel vs. a good component.
"""

import numpy as np
from typing import Optional


def inject_nominal(
    base_iddq: float,
    base_voffset: float,
    base_snr: float,
    t_ambient: np.ndarray,
    arrhenius_mult: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Class 0: Nominal Survivor — stable linear drift + Arrhenius thermal coupling."""
    timesteps = len(t_ambient)
    t = np.arange(timesteps)
    iddq    = base_iddq * arrhenius_mult + 0.001 * t + rng.normal(0, 0.02, timesteps)
    voffset = base_voffset + 0.002 * (t_ambient - 25.0) + rng.normal(0, 0.05, timesteps)
    snr     = base_snr    - 0.02  * (t_ambient - 25.0) + rng.normal(0, 0.10, timesteps)
    return {"iddq": iddq, "voffset": voffset, "snr": snr}


def inject_thermal_runaway(
    base_iddq: float,
    base_voffset: float,
    base_snr: float,
    t_ambient: np.ndarray,
    arrhenius_mult: np.ndarray,
    onset_hour: int,
    rng: np.random.Generator,
    lambda_k: float = 0.08
) -> dict:
    """
    Class 1: Thermal Runaway / Electromigration.
    I_DDQ(t) = I_base * Arrhenius + beta * exp(lambda * (t - t_onset)) * I(t >= t_onset)
    """
    timesteps = len(t_ambient)
    t = np.arange(timesteps)
    iddq = np.where(
        t >= onset_hour,
        base_iddq * arrhenius_mult + 0.5 * np.exp(lambda_k * (t - onset_hour)) + rng.normal(0, 0.02, timesteps),
        base_iddq * arrhenius_mult + 0.001 * t + rng.normal(0, 0.02, timesteps)
    )
    voffset = base_voffset + 0.005 * (t_ambient - 25.0) + rng.normal(0, 0.05, timesteps)
    snr_deg = np.where(t >= onset_hour, 0.05 * (t - onset_hour), 0.0)
    snr     = base_snr - snr_deg + rng.normal(0, 0.1, timesteps)
    return {"iddq": iddq, "voffset": voffset, "snr": snr}


def inject_mems_microcrack(
    base_iddq: float,
    base_voffset: float,
    base_snr: float,
    t_ambient: np.ndarray,
    arrhenius_mult: np.ndarray,
    onset_hour: int,
    rng: np.random.Generator,
) -> dict:
    """
    Class 2: Structural MEMS Micro-Cracking.
    V_offset(t >= onset) accumulates random walk + periodic transient spikes.
    """
    timesteps = len(t_ambient)
    t = np.arange(timesteps)
    iddq = base_iddq * arrhenius_mult + rng.normal(0, 0.02, timesteps)
    
    voffset = np.zeros(timesteps)
    for i in range(timesteps):
        if i < onset_hour:
            voffset[i] = base_voffset + rng.normal(0, 0.05)
        else:
            rw = np.sum(rng.normal(0, 0.15, size=i - onset_hour + 1))
            spike = 1.5 if (i % 7 == 0) else 0.0
            voffset[i] = base_voffset + rw + spike + rng.normal(0, 0.05)
    
    snr_deg = np.where(t >= onset_hour, 0.1 * (t - onset_hour), 0.0)
    snr     = base_snr - snr_deg + rng.normal(0, 0.1, timesteps)
    return {"iddq": iddq, "voffset": voffset, "snr": snr}


def inject_freak_outlier(
    base_iddq: float,   # Pre-shifted to κ * σ_lot above median
    base_voffset: float,
    base_snr: float,
    t_ambient: np.ndarray,
    arrhenius_mult: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Class 3: Spatial 'Freak' Part.
    Baseline violates lot uniformity at t=0 (Modified Z > 3.5σ) but may not catastrophically fail.
    """
    timesteps = len(t_ambient)
    t = np.arange(timesteps)
    iddq    = base_iddq * arrhenius_mult + 0.002 * t + rng.normal(0, 0.02, timesteps)
    voffset = base_voffset + rng.normal(0, 0.05, timesteps)
    snr     = base_snr - 1.5 + rng.normal(0, 0.1, timesteps)
    return {"iddq": iddq, "voffset": voffset, "snr": snr}


def inject_intermittent_fault(
    base_iddq: float,
    base_voffset: float,
    base_snr: float,
    t_ambient: np.ndarray,
    arrhenius_mult: np.ndarray,
    rng: np.random.Generator,
    spike_probability: float = 0.06,   # ~6% of hours exhibit transient spikes
    spike_magnitude_ua: float = 3.5,
) -> dict:
    """
    Class 4: Intermittent Fault.
    Otherwise normal component; random current spikes of significant magnitude at irregular intervals.
    Tests AstraGuard's ability to distinguish intermittent physical degradation from random noise.
    """
    timesteps = len(t_ambient)
    t = np.arange(timesteps)
    iddq    = base_iddq * arrhenius_mult + 0.001 * t + rng.normal(0, 0.02, timesteps)
    spike_mask = rng.random(timesteps) < spike_probability
    iddq[spike_mask] += spike_magnitude_ua * rng.uniform(0.8, 1.2, size=spike_mask.sum())
    voffset = base_voffset + rng.normal(0, 0.05, timesteps)
    snr     = base_snr + rng.normal(0, 0.1, timesteps)
    return {"iddq": iddq, "voffset": voffset, "snr": snr}


def inject_sensor_fault(
    base_iddq: float,
    base_voffset: float,
    base_snr: float,
    t_ambient: np.ndarray,
    arrhenius_mult: np.ndarray,
    rng: np.random.Generator,
    freeze_hour: Optional[int] = None,
) -> dict:
    """
    Class 5: Measurement Instrument / Sensor Channel Anomaly.
    The I_DDQ measurement channel freezes at the value recorded at freeze_hour.
    The component itself may be healthy — the ATE instrument has malfunctioned.
    Tests AstraGuard's Experiment 4: Can we distinguish component fault from measurement fault?
    """
    timesteps = len(t_ambient)
    t = np.arange(timesteps)
    true_iddq = base_iddq * arrhenius_mult + 0.001 * t + rng.normal(0, 0.02, timesteps)
    
    if freeze_hour is None:
        freeze_hour = int(rng.integers(12, 48))
    
    frozen_value = true_iddq[freeze_hour]
    observed_iddq = true_iddq.copy()
    observed_iddq[freeze_hour:] = frozen_value  # Exact freeze without noise un-freezing
    
    voffset = base_voffset + rng.normal(0, 0.05, timesteps)
    snr     = base_snr + rng.normal(0, 0.1, timesteps)
    return {
        "iddq": observed_iddq,
        "voffset": voffset,
        "snr": snr,
        "freeze_hour": freeze_hour,
        "is_measurement_fault": True
    }


def inject_dielectric_oscillation(
    base_iddq: float,
    base_voffset: float,
    base_snr: float,
    t_ambient: np.ndarray,
    arrhenius_mult: np.ndarray,
    rng: np.random.Generator,
    osc_period_h: float = 8.0,     # Oscillation period (hours) — solder resonance
    growth_rate: float = 0.04,     # Amplitude grows linearly with time
) -> dict:
    """
    Class 6: Dielectric Oscillation / Solder-Fatigue Resonance (UNKNOWN MODE).

    Physics:
      Gate-oxide trap accumulation under sustained voltage + temperature
      causes quasi-periodic leakage oscillations. The amplitude grows as
      more traps fill in: A(t) = growth_rate * t.

      Unlike THERMAL_RUNAWAY (monotonic exponential increase), this failure
      mode shows sinusoidal oscillation with growing envelope — a temporal
      pattern NEVER seen in Classes 1–5.

    CRITICAL: This class is INTENTIONALLY NOT in the standard training
    distribution. It is used EXCLUSIVELY for Experiment 8 (Known vs Unknown)
    to demonstrate that:
      - XGBoost (trained on Classes 1–5) cannot classify this correctly
      - LSTM AE (trained on NOMINAL only) flags it via high reconstruction error
      - Isolation Forest (fitted on NOMINAL) flags it via outlier score

    Scientific reference:
      Power-law degradation: ΔV_th = C * t^n (n ≈ 0.25–0.5) from IEEE Trans.
      Device and Materials Reliability. The oscillation emerges from
      resonant trap filling/emptying cycles under AC functional stress.
    """
    timesteps = len(t_ambient)
    t = np.arange(timesteps, dtype=float)

    # Arrhenius baseline (same as nominal — passes absolute limits easily)
    iddq_base = base_iddq * arrhenius_mult + 0.001 * t

    # Growing sinusoidal oscillation on top of baseline
    amplitude  = growth_rate * t                        # A(t) increases linearly
    oscillation = amplitude * np.sin(2 * np.pi * t / osc_period_h)

    iddq    = iddq_base + oscillation + rng.normal(0, 0.02, timesteps)
    iddq    = np.clip(iddq, 0.1, None)

    # V_offset shows correlated oscillation (trap coupling to bias voltage)
    voffset = base_voffset + 0.5 * amplitude * np.cos(2 * np.pi * t / osc_period_h) \
              + rng.normal(0, 0.05, timesteps)

    # SNR degrades slowly (oxide quality loss)
    power_law_deg = 0.3 * (t ** 0.3)
    snr = base_snr - power_law_deg + rng.normal(0, 0.1, timesteps)

    return {"iddq": iddq, "voffset": voffset, "snr": snr}


# Optional import guard for freeze_hour usage above
from typing import Optional

