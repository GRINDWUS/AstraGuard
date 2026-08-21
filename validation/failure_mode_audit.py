#!/usr/bin/env python3
"""
AstraGuard 2.4 — Failure Mode Observability & Audit Engine
==========================================================
Audits the physical observability and decision fusion metrics for every individual
failure mode present in the frozen blind test dataset.

Produces: reports/failure_mode_audit.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from astraguard_core.feature_engineering.v2 import get_v2_engineer
from astraguard_core.preprocessing import LeakageSafePreprocessor
from astraguard_core.module_a import ModuleAScreener
from astraguard_core.decision import DecisionFusionEngine

BLIND_CSV = "ASQD_2.4/asqd_24_blind_test.csv"
V2_PREPROC_DIR = "models/v2/preprocessors"
V2_MODEL_DIR   = "models/v2/module_b"
THRESHOLDS_PATH = "models/v2/optimal_fusion_thresholds.json"
REPORTS_DIR    = "reports"

DEVICE_FAMILIES = [
    "DIGITAL_IC", "MIXED_SIGNAL_IC", "MEMS_GYROSCOPE",
    "IMAGE_SENSOR", "PRECISION_VOLTAGE_REF",
]

os.makedirs(REPORTS_DIR, exist_ok=True)

df_blind = pd.read_csv(BLIND_CSV)
screener = ModuleAScreener(z_threshold=3.5)
fusion = DecisionFusionEngine()

with open(THRESHOLDS_PATH) as f:
    threshold_config = json.load(f)

print("=" * 80)
print("FAILURE MODE OBSERVABILITY AUDIT")
print(f"Blind Test Dataset: {BLIND_CSV}")
print("=" * 80)

audit_by_mode = []

for fam in DEVICE_FAMILIES:
    fam_df = df_blind[df_blind["device_family"] == fam].copy()

    prep_path = os.path.join(V2_PREPROC_DIR, f"{fam.lower()}_preprocessor_v2.pkl")
    model_path = os.path.join(V2_MODEL_DIR, f"{fam.lower()}_module_b_v2.pkl")

    engineer = get_v2_engineer(fam)
    preprocessor = LeakageSafePreprocessor.load(prep_path)
    model = pickle.load(open(model_path, "rb"))

    feats = engineer.extract_features(fam_df)
    X = preprocessor.transform(feats)
    pred_168h = model.predict(X)

    mod_a = screener.screen_population(fam_df, value_col="value_24h")
    z_scores = mod_a["robust_z_scores"].values

    fam_df["_pred_168h"] = pred_168h
    fam_df["_robust_z"] = z_scores

    # Evaluate decision rule
    r_thresh = threshold_config[fam]["red_threshold"]
    y_thresh = threshold_config[fam]["yellow_threshold"]

    decisions = []
    for z, p in zip(z_scores, pred_168h):
        if z >= 3.5 or p >= r_thresh:
            decisions.append("RED")
        elif p >= y_thresh:
            decisions.append("YELLOW")
        else:
            decisions.append("GREEN")

    fam_df["_decision"] = decisions

    # Breakdown per failure mode
    for fm, grp in fam_df.groupby("failure_mode_gt"):
        n_samples = len(grp)
        y_true = grp["value_168h_actual"].values
        y_pred = grp["_pred_168h"].values
        decs = grp["_decision"].values

        mae = float(mean_absolute_error(y_true, y_pred))

        n_red = int((decs == "RED").sum())
        n_yellow = int((decs == "YELLOW").sum())
        n_green = int((decs == "GREEN").sum())

        if fm == "NOMINAL":
            false_rejection_rate = float((n_red + n_yellow) / max(1, n_samples) * 100.0)
            recall = 100.0 - false_rejection_rate
            escape_rate = 0.0
            observability = "NOMINAL_BASELINE"
        else:
            recall = float((n_red + n_yellow) / max(1, n_samples) * 100.0)
            escape_rate = float(n_green / max(1, n_samples) * 100.0)
            false_rejection_rate = 0.0

            if recall >= 90.0:
                observability = "HIGHLY_OBSERVABLE"
            elif recall >= 50.0:
                observability = "PARTIALLY_OBSERVABLE"
            else:
                observability = "PHYSICALLY_UNOBSERVABLE_AT_96H"

        print(f"\n[{fam}] Mode: {fm:<32} (N={n_samples})")
        print(f"  MAE: {mae:.4f}")
        print(f"  Tier Decisions -> RED: {n_red:<4} | YELLOW: {n_yellow:<4} | GREEN: {n_green:<4}")
        print(f"  Recall: {recall:.1f}% | Escape Rate: {escape_rate:.1f}% | Classification: {observability}")

        audit_by_mode.append({
            "device_family": fam,
            "failure_mode": fm,
            "sample_count": n_samples,
            "mae": mae,
            "decisions": {"RED": n_red, "YELLOW": n_yellow, "GREEN": n_green},
            "recall_pct": recall,
            "escape_rate_pct": escape_rate,
            "observability_classification": observability
        })

report_file = os.path.join(REPORTS_DIR, "failure_mode_audit.json")
with open(report_file, "w") as f:
    json.dump(audit_by_mode, f, indent=2)

print(f"\n✅ Failure mode observability audit saved to {report_file}")
