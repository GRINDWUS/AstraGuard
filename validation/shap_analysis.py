#!/usr/bin/env python3
"""
AstraGuard 2.4 — SHAP Feature Attribution Analysis
==================================================
Runs game-theoretic SHAP feature attribution across all device families
using frozen v2 models and preprocessors.

Produces: reports/shap_analysis_report.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from astraguard_core.feature_engineering.v2 import get_v2_engineer
from astraguard_core.preprocessing import LeakageSafePreprocessor
from astraguard_core.explainability.shap_engine import SHAPExplainabilityEngine

BLIND_CSV = "ASQD_2.4/asqd_24_blind_test.csv"
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
print("SHAP FEATURE ATTRIBUTION ANALYSIS")
print(f"Blind Test Dataset: {BLIND_CSV}")
print("=" * 80)

results = {}

for fam in DEVICE_FAMILIES:
    fam_df = df_blind[df_blind["device_family"] == fam].copy()

    prep_path = os.path.join(V2_PREPROC_DIR, f"{fam.lower()}_preprocessor_v2.pkl")
    model_path = os.path.join(V2_MODEL_DIR, f"{fam.lower()}_module_b_v2.pkl")

    engineer = get_v2_engineer(fam)
    preprocessor = LeakageSafePreprocessor.load(prep_path)
    model = pickle.load(open(model_path, "rb"))

    feats = engineer.extract_features(fam_df)
    X = preprocessor.transform(feats)

    # Initialize SHAP engine
    shap_engine = SHAPExplainabilityEngine(model, feature_names=list(X.columns))
    
    # Calculate average absolute SHAP values across 100 sample rows
    X_sample = X.iloc[:min(100, len(X))]
    sample_attributions = []

    for i in range(len(X_sample)):
        single_row = X_sample.iloc[[i]]
        exp = shap_engine.explain_component(single_row)
        sample_attributions.append(exp["feature_attributions"])

    # Compute mean absolute SHAP per feature
    feature_keys = list(X.columns)
    mean_abs_shap = {}

    for k in feature_keys:
        vals = [abs(attr.get(k, 0.0)) for attr in sample_attributions]
        mean_abs_shap[k] = float(np.mean(vals))

    sorted_shap = dict(sorted(mean_abs_shap.items(), key=lambda x: x[1], reverse=True))
    top_5 = dict(list(sorted_shap.items())[:5])

    print(f"\n[{fam}] Top 5 Predictive Features (Mean |SHAP|):")
    for feat, shap_val in top_5.items():
        print(f"  - {feat:<35}: {shap_val:.6f}")

    results[fam] = {
        "top_5_features": top_5,
        "all_feature_importance": sorted_shap
    }

report_file = os.path.join(REPORTS_DIR, "shap_analysis_report.json")
with open(report_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ SHAP feature attribution saved to {report_file}")
