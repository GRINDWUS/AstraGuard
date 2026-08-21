"""
AstraGuard 2.1 — Thermal & Arrhenius Degradation Model
=======================================================
Models chamber temperature profile and Arrhenius-coupled leakage dynamics.

Scientific Grounding:
  - MIL-STD-883 Method 1015: Defines condition-dependent burn-in temperature/duration.
  - Arrhenius Equation: I_leakage ∝ exp(-E_a / k*T) — well-established in semiconductor physics.
  - NASA EEE-INST-002 Class S: Defines environmental stress conditions for space-grade EEE parts.

IMPORTANT NOTE: The 168-hour observation horizon used here is the AstraGuard prototype
observation window, NOT a claimed universal qualification duration. MIL-STD-883 Method 1015
contains condition-dependent tables (e.g., 240 h at 125°C for Class S under some conditions).
Actual screening duration is governed by the applicable device qualification specification.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThermalProfile:
    """Configurable thermal chamber profile metadata."""
    t_ambient_start_c: float = 25.0
    t_ambient_peak_c: float = 125.0          # Configurable: 125°C or 150°C typical stress
    ramp_tau_hours: float = 0.8              # Exponential ramp time constant
    profile_type: str = "ramp_hold"
    profile_source: str = "prototype_policy"
    observation_horizon_hours: int = 168     # Prototype observation horizon (not a MIL-STD mandate)
    qualification_standard: str = "configured_by_user"


@dataclass
class ArrheniusParams:
    """
    Arrhenius leakage model parameters.
    I_leakage(T) = I_0 * exp(q*V_T / (n*k*T))
    Simplified thermal coupling: I(T) = I_0 * exp(alpha * (T - T_ref))
    """
    alpha: float = 0.012       # Thermal sensitivity coefficient (empirical fit)
    t_ref_c: float = 25.0      # Reference temperature
    noise_sigma_ua: float = 0.02  # ATE instrument noise, ±0.02 µA


def chamber_temperature_profile(
    time_hours: np.ndarray,
    profile: Optional[ThermalProfile] = None
) -> np.ndarray:
    """
    Generates the chamber T_ambient curve.
    Default: exponential ramp from 25°C to 125°C (tau ≈ 0.8h ramp).
    Peak sustained from t≈2h onward.
    """
    if profile is None:
        profile = ThermalProfile()
    delta = profile.t_ambient_peak_c - profile.t_ambient_start_c
    return profile.t_ambient_start_c + delta * (1.0 - np.exp(-time_hours / profile.ramp_tau_hours))


def arrhenius_coupling(
    t_ambient: np.ndarray,
    params: Optional[ArrheniusParams] = None
) -> np.ndarray:
    """
    Arrhenius thermal leakage multiplier.
    Returns a dimensionless multiplier applied to baseline I_DDQ.
    """
    if params is None:
        params = ArrheniusParams()
    return np.exp(params.alpha * (t_ambient - params.t_ref_c))


def add_measurement_uncertainty(
    true_values: np.ndarray,
    instrument_noise_sigma: float,
    quantization_lsb: float = 0.001,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Models realistic ATE measurement uncertainty:
      - Gaussian instrument noise (noise floor)
      - Quantization rounding (ADC resolution)
    
    Real measurement: observed = true + noise, then quantized to LSB.
    """
    if rng is None:
        rng = np.random.default_rng()
    noisy = true_values + rng.normal(0, instrument_noise_sigma, size=true_values.shape)
    return np.round(noisy / quantization_lsb) * quantization_lsb
