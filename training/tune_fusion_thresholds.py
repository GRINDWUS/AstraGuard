"""
AstraGuard 2.4 — Phase 3: Pareto-Balanced Fusion Threshold Optimizer
======================================================================
Finds the optimal family-specific decision thresholds on the VALIDATION set
by optimizing a balanced Pareto objective:
  Loss = 20 * Escape_Rate_Pct + 1 * False_Rejection_Rate_Pct

Prevents catastrophic false-rejection explosion (over-rejection) while maintaining
high defect recall and low escape rates.

BLIND TEST DATASET IS NOT TOUCHED.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from astraguard_core.feature_engineering.v2 import get_v2_engineer
from astraguard_core.preprocessing import LeakageSafePreprocessor
from astraguard_core.module_a import ModuleAScreener

VAL_CSV        = "ASQD_2.4/asqd_24_validation.csv"
V2_PREPROC_DIR = "models/v2/preprocessors"
V2_MODEL_DIR   = "models/v2/module_b"

DEVICE_FAMILIES = [
    "DIGITAL_IC", "MIXED_SIGNAL_IC", "MEMS_GYROSCOPE",
    "IMAGE_SENSOR", "PRECISION_VOLTAGE_REF",
]

SPEC_LIMITS = {
    "DIGITAL_IC":            1150.0,
    "MIXED_SIGNAL_IC":       1150.0,
    "MEMS_GYROSCOPE":          25.0,
    "IMAGE_SENSOR":            25.0,
    "PRECISION_VOLTAGE_REF": 6800.0,
}

df_val = pd.read_csv(VAL_CSV)
screener = ModuleAScreener(z_threshold=3.5)

val_predictions = {}

for fam in DEVICE_FAMILIES:
    fam_df = df_val[df_val["device_family"] == fam].copy()
    
    eng_v2  = get_v2_engineer(fam)
    prep_v2 = LeakageSafePreprocessor.load(os.path.join(V2_PREPROC_DIR, f"{fam.lower()}_preprocessor_v2.pkl"))
    m2      = pickle.load(open(os.path.join(V2_MODEL_DIR, f"{fam.lower()}_module_b_v2.pkl"), "rb"))

    X_v2 = prep_v2.transform(eng_v2.extract_features(fam_df))
    p_v2 = m2.predict(X_v2)

    mod_a = screener.screen_population(fam_df, value_col="value_24h")
    z_scores = mod_a["robust_z_scores"].values

    fam_df["_pred_v2"] = p_v2
    fam_df["_z"]       = z_scores
    fam_df["_is_defective"] = (fam_df["failure_mode_gt"] != "NOMINAL")

    val_predictions[fam] = fam_df

optimal_thresholds = {}

# Grid Search per family — fine grid
red_multipliers    = np.linspace(0.85, 1.15, 31)
yellow_multipliers = np.linspace(0.70, 1.00, 31)

for fam in DEVICE_FAMILIES:
    fdf  = val_predictions[fam]
    spec = SPEC_LIMITS[fam]
    
    best_loss = float("inf")
    best_config = (1.0, 0.9)
    best_metrics = {}

    z = fdf["_z"].values
    p = fdf["_pred_v2"].values
    is_def = fdf["_is_defective"].values
    is_nom = ~is_def

    total_def = int(is_def.sum())
    total_nom = int(is_nom.sum())

    for r_m in red_multipliers:
        for y_m in yellow_multipliers:
            if y_m >= r_m:
                continue

            red_thresh    = r_m * spec
            yellow_thresh = y_m * spec

            is_red    = (z >= 3.5) | (p >= red_thresh)
            is_yellow = (~is_red) & (p >= yellow_thresh)
            is_green  = (~is_red) & (~is_yellow)

            escapes       = int((is_green & is_def).sum())
            false_rejects = int(((is_red | is_yellow) & is_nom).sum())

            esc_rate_pct = 100.0 * escapes / max(1, total_def)
            frr_pct      = 100.0 * false_rejects / max(1, total_nom)

            # Balanced loss: 20 * Escape_Rate_Pct + 1 * FRR_Pct
            loss = (20.0 * esc_rate_pct) + (1.0 * frr_pct)

            if loss < best_loss:
                best_loss = loss
                best_metrics = {
                    "red_multiplier": float(r_m),
                    "yellow_multiplier": float(y_m),
                    "red_threshold": float(red_thresh),
                    "yellow_threshold": float(yellow_thresh),
                    "val_escapes": escapes,
                    "val_escape_rate_pct": esc_rate_pct,
                    "val_false_rejects": false_rejects,
                    "val_frr_pct": frr_pct,
                    "val_loss": loss,
                }

    optimal_thresholds[fam] = best_metrics
    print(f"[{fam}] RED: {best_metrics['red_multiplier']:.2f}x USL ({best_metrics['red_threshold']:.1f}) | YELLOW: {best_metrics['yellow_multiplier']:.2f}x USL ({best_metrics['yellow_threshold']:.1f})")
    print(f"       Val Escapes: {best_metrics['val_escapes']} ({best_metrics['val_escape_rate_pct']:.1f}%) | Val FRR: {best_metrics['val_false_rejects']} ({best_metrics['val_frr_pct']:.1f}%)")

os.makedirs("models/v2", exist_ok=True)
out_path = "models/v2/optimal_fusion_thresholds.json"
with open(out_path, "w") as f:
    json.dump(optimal_thresholds, f, indent=2)

print(f"\nPareto optimization complete. Saved to {out_path}")
