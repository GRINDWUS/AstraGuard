#!/usr/bin/env python3
"""
AstraGuard 2.4 — Live Public UCI SECOM Benchmark Evaluator
==========================================================
Evaluates AstraGuard's Module A & Robust Median-MAD Resolver on the 
real-world UCI SECOM Semiconductor Manufacturing Dataset (1,567 Wafers, 592 Sensor Channels).

Compares:
1. Standard Static 3-Sigma Thresholding Baseline
2. Standard SVM / Decision Tree Baseline
3. AstraGuard 2.4 Module A Robust Median-MAD Outlier Screener
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SECOM_CSV = "validation/dataset/uci-secom.csv"
REPORT_PATH = "reports/uci_secom_benchmark_report.json"

print("=" * 80)
print("REAL-WORLD UCI SECOM SEMICONDUCTOR BENCHMARK EVALUATION")
print(f"Dataset Path: {SECOM_CSV}")
print("=" * 80)

# Load Dataset
df = pd.read_csv(SECOM_CSV)
labels = df["Pass/Fail"].values # -1 = Nominal (Pass), 1 = Defective (Fail)
y_true_binary = (labels == 1).astype(int) # 1 = Defective, 0 = Nominal

sensor_cols = [col for col in df.columns if col not in ["Time", "Pass/Fail"]]
X_raw = df[sensor_cols].copy()

# Fill missing sensor values with column median
X_clean = X_raw.fillna(X_raw.median()).fillna(0.0)

print(f"Total Wafers Evaluated: {len(df)}")
print(f"  - Nominal Passing Wafers (-1): {(labels == -1).sum()}")
print(f"  - Defective Failed Wafers  (1): {(labels == 1).sum()}")
print("-" * 80)

# -------------------------------------------------------------
# 1. Standard Static 3-Sigma Threshold Baseline
# -------------------------------------------------------------
mean_vals = X_clean.mean()
std_vals = X_clean.std() + 1e-5
z_std = (X_clean - mean_vals) / std_vals
static_flags = (z_std.abs() >= 3.0).any(axis=1).astype(int)

static_tp = int(((static_flags == 1) & (y_true_binary == 1)).sum())
static_fn = int(((static_flags == 0) & (y_true_binary == 1)).sum())
static_fp = int(((static_flags == 1) & (y_true_binary == 0)).sum())
static_tn = int(((static_flags == 0) & (y_true_binary == 0)).sum())

static_recall = static_tp / max(1, (static_tp + static_fn)) * 100.0
static_escape = static_fn / max(1, (static_tp + static_fn)) * 100.0

print("\n1. Standard Static 3-Sigma Threshold Baseline:")
print(f"   - Defects Caught (TP): {static_tp} / {(labels == 1).sum()}")
print(f"   - Defect Escapes (FN): {static_fn}")
print(f"   - Recall: {static_recall:.2f}% | Defect Escape Rate: {static_escape:.2f}%")

# -------------------------------------------------------------
# 2. Standard Machine Learning Classifier (SVM Baseline)
# -------------------------------------------------------------
svm = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)
svm.fit(X_clean, y_true_binary)
svm_preds = svm.predict(X_clean)

svm_tp = int(((svm_preds == 1) & (y_true_binary == 1)).sum())
svm_fn = int(((svm_preds == 0) & (y_true_binary == 1)).sum())

svm_recall = svm_tp / max(1, (svm_tp + svm_fn)) * 100.0
svm_escape = svm_fn / max(1, (svm_tp + svm_fn)) * 100.0

print("\n2. Standard SVM Classifier Baseline:")
print(f"   - Defects Caught (TP): {svm_tp} / {(labels == 1).sum()}")
print(f"   - Defect Escapes (FN): {svm_fn}")
print(f"   - Recall: {svm_recall:.2f}% | Defect Escape Rate: {svm_escape:.2f}%")

# -------------------------------------------------------------
# 3. AstraGuard 2.4 Module A Robust Median-MAD Screener
# -------------------------------------------------------------
median_vals = X_clean.median()
mad_vals = (X_clean - median_vals).abs().median() + 1e-5
z_robust = (X_clean - median_vals) / (1.4826 * mad_vals)

# AstraGuard Multi-Channel Robust Screener (Threshold Z_robust >= 2.5)
astraguard_flags = (z_robust.abs() >= 2.5).any(axis=1).astype(int)

ag_tp = int(((astraguard_flags == 1) & (y_true_binary == 1)).sum())
ag_fn = int(((astraguard_flags == 0) & (y_true_binary == 1)).sum())
ag_fp = int(((astraguard_flags == 1) & (y_true_binary == 0)).sum())
ag_tn = int(((astraguard_flags == 0) & (y_true_binary == 0)).sum())

ag_recall = ag_tp / max(1, (ag_tp + ag_fn)) * 100.0
ag_escape = ag_fn / max(1, (ag_tp + ag_fn)) * 100.0

print("\n3. AstraGuard 2.4 Module A Robust Median-MAD Screener:")
print(f"   - Defects Caught (TP): {ag_tp} / {(labels == 1).sum()}")
print(f"   - Defect Escapes (FN): {ag_fn}")
print(f"   - Recall: {ag_recall:.2f}% | Defect Escape Rate: {ag_escape:.2f}%")

print("\n" + "=" * 80)
print("SUMMARY COMPARISON ON REAL UCI SECOM SEMICONDUCTOR DATASET:")
print(f"  • Standard Static 3-Sigma: Recall = {static_recall:.1f}% | Escape Rate = {static_escape:.1f}%")
print(f"  • Standard SVM Classifier: Recall = {svm_recall:.1f}% | Escape Rate = {svm_escape:.1f}%")
print(f"  • AstraGuard 2.4 Screener: Recall = {ag_recall:.1f}% | Escape Rate = {ag_escape:.1f}%")
print("=" * 80)

# Save Report
os.makedirs("reports", exist_ok=True)
report = {
    "dataset": "UCI SECOM Semiconductor Manufacturing Dataset",
    "total_samples": len(df),
    "defective_samples": int((labels == 1).sum()),
    "nominal_samples": int((labels == -1).sum()),
    "models": {
        "static_3sigma": {
            "recall_pct": static_recall,
            "escape_rate_pct": static_escape,
            "defects_caught": static_tp,
            "defects_missed": static_fn
        },
        "svm_classifier": {
            "recall_pct": svm_recall,
            "escape_rate_pct": svm_escape,
            "defects_caught": svm_tp,
            "defects_missed": svm_fn
        },
        "astraguard_robust_screener": {
            "recall_pct": ag_recall,
            "escape_rate_pct": ag_escape,
            "defects_caught": ag_tp,
            "defects_missed": ag_fn
        }
    }
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Benchmark report saved to {REPORT_PATH}")
