#!/usr/bin/env python3
"""
AstraGuard 2.4 — Telemetry Horizon Comparison Analysis
======================================================
Quantifies the exact performance delta between 24h prediction (v1)
and 96h prognostic forecasting (v2) across all device families.

Produces: reports/telemetry_horizon_analysis.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from astraguard_core.feature_engineering import feature_registry          # v1
from astraguard_core.feature_engineering.v2 import get_v2_engineer        # v2
from astraguard_core.preprocessing import LeakageSafePreprocessor

BLIND_CSV = "ASQD_2.4/asqd_24_blind_test.csv"
V1_PREPROC_DIR = "models/preprocessors"
V1_MODEL_DIR   = "models/module_b"
V2_PREPROC_DIR = "models/v2/preprocessors"
V2_MODEL_DIR   = "models/v2/module_b"
REPORTS_DIR    = "reports"

DEVICE_FAMILIES = [
    "DIGITAL_IC", "MIXED_SIGNAL_IC", "MEMS_GYROSCOPE",
    "IMAGE_SENSOR", "PRECISION_VOLTAGE_REF",
]

os.makedirs(REPORTS_DIR, exist_ok=True)
df_blind = pd.read_csv(BLIND_CSV)

print("=" * 80)
print("TELEMETRY HORIZON COMPARISON — 24H (v1) VS 96H (v2)")
print(f"Blind Test Dataset: {BLIND_CSV}")
print("=" * 80)

results = {}

for fam in DEVICE_FAMILIES:
    fam_df = df_blind[df_blind["device_family"] == fam].copy()
    y_true = fam_df["value_168h_actual"].values

    # --- v1 (0h + 24h) ---
    eng_v1 = feature_registry.get_engineer(fam)
    prep_v1 = LeakageSafePreprocessor.load(os.path.join(V1_PREPROC_DIR, f"{fam.lower()}_preprocessor.pkl"))
    m1_art = pickle.load(open(os.path.join(V1_MODEL_DIR, f"{fam.lower()}_module_b.pkl"), "rb"))
    m1 = m1_art["model"] if isinstance(m1_art, dict) else m1_art

    X_v1 = prep_v1.transform(eng_v1.extract_features(fam_df))
    pred_v1 = m1.predict(X_v1)
    mae_v1 = float(mean_absolute_error(y_true, pred_v1))
    r2_v1 = float(r2_score(y_true, pred_v1))

    # --- v2 (0h + 24h + 96h) ---
    eng_v2 = get_v2_engineer(fam)
    prep_v2 = LeakageSafePreprocessor.load(os.path.join(V2_PREPROC_DIR, f"{fam.lower()}_preprocessor_v2.pkl"))
    m2 = pickle.load(open(os.path.join(V2_MODEL_DIR, f"{fam.lower()}_module_b_v2.pkl"), "rb"))

    X_v2 = prep_v2.transform(eng_v2.extract_features(fam_df))
    pred_v2 = m2.predict(X_v2)
    mae_v2 = float(mean_absolute_error(y_true, pred_v2))
    r2_v2 = float(r2_score(y_true, pred_v2))

    mae_improvement_pct = float((mae_v1 - mae_v2) / max(1e-6, mae_v1) * 100.0)

    print(f"\n[{fam}]")
    print(f"  v1 (24h Horizon) -> MAE: {mae_v1:>9.4f} | R²: {r2_v1:>7.4f}")
    print(f"  v2 (96h Horizon) -> MAE: {mae_v2:>9.4f} | R²: {r2_v2:>7.4f}")
    print(f"  MAE Improvement  -> {mae_improvement_pct:+.2f}%")

    results[fam] = {
        "v1_24h": {"mae": mae_v1, "r2": r2_v1},
        "v2_96h": {"mae": mae_v2, "r2": r2_v2},
        "mae_improvement_pct": mae_improvement_pct
    }

# Overall comparison
all_y_true = df_blind["value_168h_actual"].values
all_pred_v1 = []
all_pred_v2 = []

for fam in DEVICE_FAMILIES:
    fam_df = df_blind[df_blind["device_family"] == fam]
    eng_v1 = feature_registry.get_engineer(fam)
    prep_v1 = LeakageSafePreprocessor.load(os.path.join(V1_PREPROC_DIR, f"{fam.lower()}_preprocessor.pkl"))
    m1_art = pickle.load(open(os.path.join(V1_MODEL_DIR, f"{fam.lower()}_module_b.pkl"), "rb"))
    m1 = m1_art["model"] if isinstance(m1_art, dict) else m1_art
    all_pred_v1.extend(m1.predict(prep_v1.transform(eng_v1.extract_features(fam_df))))

    eng_v2 = get_v2_engineer(fam)
    prep_v2 = LeakageSafePreprocessor.load(os.path.join(V2_PREPROC_DIR, f"{fam.lower()}_preprocessor_v2.pkl"))
    m2 = pickle.load(open(os.path.join(V2_MODEL_DIR, f"{fam.lower()}_module_b_v2.pkl"), "rb"))
    all_pred_v2.extend(m2.predict(prep_v2.transform(eng_v2.extract_features(fam_df))))

overall_mae_v1 = float(mean_absolute_error(all_y_true, all_pred_v1))
overall_mae_v2 = float(mean_absolute_error(all_y_true, all_pred_v2))
overall_r2_v1  = float(r2_score(all_y_true, all_pred_v1))
overall_r2_v2  = float(r2_score(all_y_true, all_pred_v2))
overall_imp    = float((overall_mae_v1 - overall_mae_v2) / overall_mae_v1 * 100.0)

print("\n" + "=" * 80)
print("OVERALL HORIZON GAIN")
print(f"  v1 Overall MAE: {overall_mae_v1:.4f} | R²: {overall_r2_v1:.4f}")
print(f"  v2 Overall MAE: {overall_mae_v2:.4f} | R²: {overall_r2_v2:.4f}")
print(f"  Overall Error Reduction: {overall_imp:.2f}%")
print("=" * 80)

results["OVERALL"] = {
    "v1_24h": {"mae": overall_mae_v1, "r2": overall_r2_v1},
    "v2_96h": {"mae": overall_mae_v2, "r2": overall_r2_v2},
    "mae_improvement_pct": overall_imp
}

report_file = os.path.join(REPORTS_DIR, "telemetry_horizon_analysis.json")
with open(report_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Telemetry horizon analysis saved to {report_file}")
