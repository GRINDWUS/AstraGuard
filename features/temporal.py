"""
AstraGuard 2.1 — Temporal & Population Feature Engineering
===========================================================
Implements Methods 1 and 2 from the pipeline specification:

  Method 1: Spatial Outlier Screening (Modified Z-Score via MAD at t=24h)
  Method 2: Temporal Feature Engineering (ΔX derivatives, lot-residual extraction)
  New:      Sensor-Fault Channel Detection (frozen-value variance test)

Research Questions addressed:
  Q1: Can 24h measurements predict later degradation?
  Q2: Does robust population analysis improve defect detection?
  Q3: Can we distinguish component degradation from instrument/sensor fault?
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─── Method 1: Spatial / Population Outlier Screening ────────────────────────

def compute_modified_z_scores(values: pd.Series) -> pd.Series:
    """
    Modified Z-Score per Iglewicz & Hoaglin (1993):
      M_i = 0.6745 * (x_i - median(X)) / MAD
    Any component with |M_i| > 3.5 is flagged as a Spatial Outlier (Class 3 / Freak Part).
    The 0.6745 constant makes MAD consistent with σ for Normal distributions.
    Default threshold 3.5σ per methodology; AstraGuard uses it as a flag, not a reject criterion alone.
    """
    med = values.median()
    mad = (values - med).abs().median()
    if mad < 1e-8:
        return pd.Series(0.0, index=values.index)
    return (0.6745 * (values - med)) / mad


def robust_z_score(values: pd.Series) -> pd.Series:
    """
    Robust Z-Score (1.4826 * MAD scaling — consistent with σ for Normal distributions):
      Z_robust = (x - median) / (1.4826 * MAD)
    Used as primary population anomaly score in AstraGuard's Module A.
    """
    med = values.median()
    mad = (values - med).abs().median()
    if mad < 1e-8:
        return pd.Series(0.0, index=values.index)
    return (values - med) / (1.4826 * mad)


def flag_spatial_outliers(df: pd.DataFrame, channel: str = "iddq_ua",
                           checkpoint_hour: int = 24,
                           threshold: float = 3.5) -> pd.DataFrame:
    """
    Applies Modified Z-Score spatial outlier flagging at the given checkpoint hour.
    Returns DataFrame with added columns:
      - modified_z_<channel>: Modified Z-Score at checkpoint
      - robust_z_<channel>:   Robust Z-Score at checkpoint
      - is_spatial_outlier:   Bool flag (|M_i| > threshold)
    """
    chk = df[df["hour"] == checkpoint_hour].copy()
    chk[f"modified_z_{channel}"] = compute_modified_z_scores(chk[channel])
    chk[f"robust_z_{channel}"]   = robust_z_score(chk[channel])
    chk["is_spatial_outlier"]    = chk[f"modified_z_{channel}"].abs() > threshold
    return chk


# ─── Method 2: Temporal Feature Engineering ──────────────────────────────────

def extract_checkpoint_features(df_telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts 24h checkpoint feature vectors from full time-series telemetry.
    Returns one row per component with temporal kinetic features:
      
      Δ Features (deviation from hour 0):
        - delta_iddq_24h    = I_DDQ(24) - I_DDQ(0)
        - delta_voffset_24h = V_offset(24) - V_offset(0)
        - delta_snr_24h     = SNR(24) - SNR(0)
      
      Drift Velocity:
        - drift_velocity_iddq = delta_iddq_24h / 24
      
      Relative Acceleration:
        - drift_rel_iddq = delta_iddq_24h / (I_DDQ(0) + 1e-5)
      
      Lot-Residual Extraction (deviation from lot population median at checkpoint):
        - iddq_lot_residual = I_DDQ(24) - median(I_DDQ(24) of lot)
    """
    t0  = df_telemetry[df_telemetry["hour"] == 0].set_index("component_id")
    t24 = df_telemetry[df_telemetry["hour"] == 24].set_index("component_id")
    
    feats = pd.DataFrame(index=t0.index)
    feats["lot_id"]        = t0["lot_id"]
    feats["iddq_0h"]       = t0["iddq_ua"]
    feats["iddq_24h"]      = t24["iddq_ua"]
    feats["v_offset_0h"]   = t0["v_offset_mv"]
    feats["v_offset_24h"]  = t24["v_offset_mv"]
    feats["snr_24h"]       = t24["snr_db"]
    feats["t_die_24h"]     = t24["t_die_c"]
    
    # Δ temporal derivatives
    feats["delta_iddq_24h"]    = feats["iddq_24h"] - feats["iddq_0h"]
    feats["delta_voffset_24h"] = feats["v_offset_24h"] - feats["v_offset_0h"]
    feats["delta_snr_24h"]     = feats["snr_24h"] - t0["snr_db"]
    
    # Kinetic velocities & relative acceleration
    feats["drift_velocity_iddq"]  = feats["delta_iddq_24h"] / 24.0
    feats["drift_rel_iddq"]       = feats["delta_iddq_24h"] / (feats["iddq_0h"] + 1e-5)
    
    # Population-relative scores at t=24h
    feats["robust_z_iddq_0h"]    = robust_z_score(feats["iddq_0h"])
    feats["robust_z_delta_iddq"] = robust_z_score(feats["delta_iddq_24h"])
    feats["modified_z_iddq_24h"] = compute_modified_z_scores(feats["iddq_24h"])
    
    # Lot-median residual extraction
    lot_median_24h = feats["iddq_24h"].median()
    feats["iddq_lot_residual"] = feats["iddq_24h"] - lot_median_24h
    
    return feats.reset_index()


# ─── Sensor Fault Channel Detection ──────────────────────────────────────────

def detect_frozen_channels(df_telemetry: pd.DataFrame,
                            channel: str = "iddq_ua",
                            window_hours: int = 24,
                            variance_threshold: float = 1e-4) -> pd.DataFrame:
    """
    Detects frozen/constant measurement channels (potential sensor/instrument fault).
    A channel with variance < variance_threshold over a sustained window is flagged.
    
    Experiment 4: Can AstraGuard distinguish component degradation from instrument fault?
    """
    results = []
    for comp_id, grp in df_telemetry.groupby("component_id"):
        vals = grp.sort_values("hour")[channel].values
        # Check variance in rolling windows
        frozen = False
        for start in range(0, len(vals) - window_hours, window_hours // 2):
            window_var = np.var(vals[start:start + window_hours])
            if window_var < variance_threshold:
                frozen = True
                break
        results.append({"component_id": comp_id, "channel_frozen": frozen})
    return pd.DataFrame(results)
