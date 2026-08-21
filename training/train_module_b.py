"""
AstraGuard 2.4 — Module B Degradation Model Training & Blind Evaluation
=========================================================================
Executes multi-candidate model tournament (Ridge, RandomForest, XGBoost, LightGBM)
across all 5 space device families.

Enforces zero-lookahead blind evaluation protocol:
  1. Train candidates on TRAIN split (24 lots).
  2. Select winner based STRICTLY on Validation split (6 lots).
  3. LOCK winner model.
  4. Perform Blind Test evaluation ONCE (6 lots).
"""

import os
import json
import pandas as pd
import numpy as np

from training.matrix_builder import TrainingMatrixBuilder
from astraguard_core.module_b import ModuleBTrainer, ModuleBEvaluator

OUTPUT_DIR = "models/module_b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE_FAMILIES = [
    "DIGITAL_IC",
    "MIXED_SIGNAL_IC",
    "MEMS_GYROSCOPE",
    "IMAGE_SENSOR",
    "PRECISION_VOLTAGE_REF",
]

# Upper Spec Limits (USL) for defect escape evaluation [PE]
SPEC_LIMITS = {
    "DIGITAL_IC": 1150.0,            # uA (IDDQ limit)
    "MIXED_SIGNAL_IC": 1150.0,       # uA (IDDQ limit)
    "MEMS_GYROSCOPE": 25.0,          # dps (ZRO limit)
    "IMAGE_SENSOR": 25.0,            # nA/cm2 (Dark current limit)
    "PRECISION_VOLTAGE_REF": 6800.0, # uV (Drift limit)
}

builder = TrainingMatrixBuilder()

print("=" * 80)
print("ASTRAGUARD 2.4 — MODULE B MODEL TOURNAMENT & BLIND EVALUATION")
print("=" * 80)

overall_summary = {}

for dev_family in DEVICE_FAMILIES:
    print(f"\n" + "-" * 80)
    print(f"DEVICE FAMILY: {dev_family}")
    print("-" * 80)

    # 1. Load training matrix
    matrix = builder.load_matrix(dev_family)
    spec_lim = SPEC_LIMITS.get(dev_family)

    print(f"Matrix Shapes -> Train: {matrix.X_train.shape}, Val: {matrix.X_val.shape}, Blind Test: {matrix.X_test.shape}")
    print(f"Primary Target: {matrix.target_name} (Upper Spec Limit: {spec_lim})")

    # 2. Run Tournament & Select Winner on Validation Set
    trainer = ModuleBTrainer(device_family=dev_family)
    winning_model, val_info = trainer.run_tournament(matrix, spec_limit=spec_lim)

    print(f"\n🏆 Tournament Winner (Selected on Val MAE): {trainer.winning_model_name_}")
    print("  Leaderboard:")
    for row in trainer.leaderboard_:
        tag = " 🌟 [WINNER]" if row["model_name"] == trainer.winning_model_name_ else ""
        print(f"    - {row['model_name']:15s} | Val MAE: {row['val_mae']:8.4f} | Val R2: {row['val_r2']:7.4f} | OOD Gap: {row['ood_generalization_gap']:7.4f}{tag}")

    # 3. Lock Model Artifact
    model_path = os.path.join(OUTPUT_DIR, f"{dev_family.lower()}_module_b.pkl")
    sha_hash = trainer.save_model(model_path)
    print(f"  🔒 Model parameters locked & saved to {model_path} (SHA-256: {sha_hash[:12]}...)")

    # 4. Perform Blind Test Evaluation ONCE
    y_test_pred = winning_model.predict(matrix.X_test)
    test_metrics = ModuleBEvaluator.evaluate(matrix.y_test, y_test_pred, spec_limit=spec_lim)

    print(f"\n🎯 BLIND TEST EVALUATION RESULTS (Unseen Lot Generalization):")
    print(f"    - Blind Test MAE         : {test_metrics['mae']:.4f}")
    print(f"    - Blind Test RMSE        : {test_metrics['rmse']:.4f}")
    print(f"    - Blind Test Relative Err: {test_metrics['mape_pct']:.2f}%")
    print(f"    - Blind Test R² Score    : {test_metrics['r2_score']:.4f}")
    if "escaped_defect_rate_pct" in test_metrics:
        print(f"    - Escaped Defect Rate    : {test_metrics['escaped_defect_rate_pct']:.2f}% ({test_metrics['escaped_defects']}/{test_metrics['total_defects']} escaped)")

    overall_summary[dev_family] = {
        "winning_model": trainer.winning_model_name_,
        "val_mae": val_info["val_mae"],
        "val_r2": val_info["val_r2"],
        "ood_generalization_gap": val_info["ood_generalization_gap"],
        "blind_test_mae": test_metrics["mae"],
        "blind_test_rmse": test_metrics["rmse"],
        "blind_test_mape_pct": test_metrics["mape_pct"],
        "blind_test_r2": test_metrics["r2_score"],
        "blind_test_escaped_defect_rate_pct": test_metrics.get("escaped_defect_rate_pct", 0.0),
        "model_sha256": sha_hash
    }

print("\n" + "=" * 80)
print("MODULE B TOURNAMENT & BLIND EVALUATION COMPLETE")
print("=" * 80)

summary_path = os.path.join(OUTPUT_DIR, "tournament_summary.json")
with open(summary_path, "w") as f:
    json.dump(overall_summary, f, indent=2)

print(f"Overall summary saved to {summary_path}")
