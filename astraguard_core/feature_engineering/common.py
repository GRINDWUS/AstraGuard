"""
AstraGuard 2.4 — Feature Engineering Physics & Statistical Primitives
========================================================================
Leakage-safe primitives for computing physical scaling, degradation velocity,
and robust population stats (Median / MAD Z-scores).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

K_BOLTZMANN_EV = 8.617333262145e-5  # eV/K [PE]


def calc_growth_rate(v0: pd.Series, v24: pd.Series, delta_t_hours: float = 24.0) -> pd.Series:
    """Calculate linear drift rate per hour [PE]."""
    return (v24 - v0) / max(1e-6, delta_t_hours)


def calc_ratio(v0: pd.Series, v24: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Calculate degradation amplification ratio v24 / max(eps, v0) [PE]."""
    return v24 / np.maximum(eps, v0)


def calc_arrhenius_temp_normalized(
    val: pd.Series,
    stress_temp_c: float,
    t_ref_c: float = 25.0,
    ea_eV: float = 0.68
) -> pd.Series:
    """
    [PE] Scaled measurement value normalized to reference temperature (25°C).
    I_normalized = I_measured * exp( (Ea / k_B) * (1/T_stress - 1/T_ref) )
    """
    temp_stress_k = stress_temp_c + 273.15
    temp_ref_k = t_ref_c + 273.15
    scale_factor = np.exp((ea_eV / K_BOLTZMANN_EV) * ((1.0 / temp_stress_k) - (1.0 / temp_ref_k)))
    return val * scale_factor


def calc_robust_population_stats(s: pd.Series) -> Tuple[float, float]:
    """
    Calculate leakage-safe population median and MAD (Median Absolute Deviation).
    MAD_scaled = 1.4826 * median(|x - median(x)|)
    """
    median = float(s.median())
    mad = float(np.median(np.abs(s - median)))
    robust_mad = 1.4826 * mad if mad > 1e-9 else 1e-4
    return median, robust_mad


def calc_robust_z_score(
    s: pd.Series,
    median: float = None,
    robust_mad: float = None
) -> pd.Series:
    """
    Calculate Robust Z-score: Z_robust = (x - Median) / Robust_MAD
    Prevents outlier leverage in population anomaly screening.
    """
    if median is None or robust_mad is None:
        med, mad = calc_robust_population_stats(s)
        median = median if median is not None else med
        robust_mad = robust_mad if robust_mad is not None else mad

    return (s - median) / max(1e-6, robust_mad)


# ---------------------------------------------------------------------------
# 96h Trajectory Primitives (v2 feature contract, additive to existing set)
# These NEVER read value_168h_actual. Strict leakage guard.
# Reference: Black (1969), Adams & MacKay (2007)
# ---------------------------------------------------------------------------

def calc_velocity_24_96(v24: pd.Series, v96: pd.Series, delta_t_hours: float = 72.0) -> pd.Series:
    """
    Degradation velocity from 24h to 96h checkpoint.
    [PE] velocity_24_96 = (v96 - v24) / 72h
    This is the primary 96h signal: captures regime shift in degradation rate.
    Best feature for THERMAL_RUNAWAY (d=53-274) and MEMS_STICTION (d=62).
    """
    return (v96 - v24) / max(1e-6, delta_t_hours)


def calc_trajectory_acceleration(
    vel_0_24: pd.Series, vel_24_96: pd.Series
) -> pd.Series:
    """
    Change in degradation rate between 0→24h and 24→96h windows.
    [PE] acceleration = velocity_24_96 - velocity_0_24
    Captures runaway onset: component was stable, then suddenly degrading.
    Best feature for SPATIAL_OUTLIER (d=21-28).
    """
    return vel_24_96 - vel_0_24


def calc_growth_ratio_96h(
    vel_0_24: pd.Series, vel_24_96: pd.Series, eps: float = 1e-9
) -> pd.Series:
    """
    Ratio of 24→96h velocity to 0→24h velocity.
    [PE] growth_ratio = velocity_24_96 / (|velocity_0_24| + eps)
    Key prognostic signal: a stable component has ratio ~1.0,
    a failing component shows ratio >> 1 as 96h acceleration dominates.
    Best feature for ELECTROMIGRATION (d=41-114) and DARK_CURRENT_SPIKE (d=93).
    """
    return vel_24_96 / (vel_0_24.abs() + eps)
