#!/usr/bin/env python3
"""
AstraGuard 2.4 — Current Model Analysis & Failure Mode Audit
============================================================
Analyzes frozen v2 XGBoost models across all device families and failure modes
on the frozen blind test dataset (ASQD_2.4/asqd_24_blind_test.csv).

Produces: reports/model_analysis_report.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, recall_score, precision_score, f1_score

from astraguard_core.feature_engineering.v2 import get_v2_engineer
from astraguard_core.preprocessing import LeakageSafePreprocessor

BLIND_CSV = "ASQD_2.4/asqd_24_blind_test.csv"
V2_PREPROC_DIR = "models/v2/preprocessors"
V2_MODEL_DIR   = "models/v2/module_b"
REPORTS_DIR    = "reports"

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

os.makedirs(REPORTS_DIR, exist_ok=True)

df_blind = pd.read_csv(BLIND_CSV)
results = {}

print("=" * 80)
print("ASTRA GUARD 2.4 — FROZEN MODEL PERFORMANCE AUDIT")
print(f"Dataset: {BLIND_CSV}")
print("=" * 80)

overall_y_true = []
overall_y_pred = []

for fam in DEVICE_FAMILIES:
    print(f"\n=== FAMILY: {fam} ===")
    fam_df = df_blind[df_blind["device_family"] == fam].copy()
    
    if len(fam_df) == 0:
        print(f"Warning: No data found for {fam}")
        continue

    # Load preprocessor & model
    prep_path = os.path.join(V2_PREPROC_DIR, f"{fam.lower()}_preprocessor_v2.pkl")
    model_path = os.path.join(V2_MODEL_DIR, f"{fam.lower()}_module_b_v2.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found")
        continue

    engineer = get_v2_engineer(fam)
    preprocessor = LeakageSafePreprocessor.load(prep_path)
    model = pickle.load(open(model_path, "rb"))

    # Extract features & predict
    feats = engineer.extract_features(fam_df)
    X = preprocessor.transform(feats)
    y_true = fam_df["value_168h_actual"].values
    y_pred = model.predict(X)

    overall_y_true.extend(y_true)
    overall_y_pred.extend(y_pred)

    # Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))

    # Binary defect metrics based on USL
    usl = SPEC_LIMITS[fam]
    y_true_bin = (y_true > usl).astype(int)
    y_pred_bin = (y_pred > usl).astype(int)

    recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    precision = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    print(f"  Samples: {len(fam_df)}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Defect Recall (USL > {usl}): {recall*100:.1f}%")
    print(f"  Defect Precision: {precision*100:.1f}%")
    print(f"  F1 Score: {f1:.4f}")

    # Per failure mode breakdown
    fm_metrics = {}
    print(f"  Per-Failure-Mode Breakdown:")
    for fm, grp in fam_df.groupby("failure_mode_gt"):
        fm_indices = grp.index - fam_df.index[0]
        y_t_fm = y_true[fm_indices]
        y_p_fm = y_pred[fm_indices]

        fm_mae = float(mean_absolute_error(y_t_fm, y_p_fm))
        fm_rmse = float(np.sqrt(np.mean((y_t_fm - y_p_fm) ** 2)))

        y_t_bin_fm = (y_t_fm > usl).astype(int)
        y_p_bin_fm = (y_p_fm > usl).astype(int)

        fm_rec = float(recall_score(y_t_bin_fm, y_p_bin_fm, zero_division=0)) if y_t_bin_fm.sum() > 0 else 1.0

        print(f"    - {fm:<32} (N={len(grp)}): MAE={fm_mae:.4f}, Recall={fm_rec*100:.1f}%")
        fm_metrics[fm] = {
            "sample_count": len(grp),
            "mae": fm_mae,
            "rmse": fm_rmse,
            "recall": fm_rec,
            "defective_count": int(y_t_bin_fm.sum())
        }

    results[fam] = {
        "sample_count": len(fam_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "usl_limit": usl,
        "per_failure_mode": fm_metrics
    }

# Overall metrics
overall_y_true = np.array(overall_y_true)
overall_y_pred = np.array(overall_y_pred)
overall_mae = float(mean_absolute_error(overall_y_true, overall_y_pred))
overall_rmse = float(np.sqrt(np.mean((overall_y_true - overall_y_pred) ** 2)))
overall_r2 = float(r2_score(overall_y_true, overall_y_pred))

print("\n" + "=" * 80)
print("OVERALL SYSTEM METRICS")
print(f"Total Blind Samples: {len(overall_y_true)}")
print(f"Overall MAE:  {overall_mae:.4f}")
print(f"Overall RMSE: {overall_rmse:.4f}")
print(f"Overall R²:   {overall_r2:.4f}")
print("=" * 80)

results["OVERALL"] = {
    "sample_count": len(overall_y_true),
    "mae": overall_mae,
    "rmse": overall_rmse,
    "r2": overall_r2
}

report_file = os.path.join(REPORTS_DIR, "model_analysis_report.json")
with open(report_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Detailed analysis saved to {report_file}")
