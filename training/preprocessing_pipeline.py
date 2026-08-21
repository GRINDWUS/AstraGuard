"""
AstraGuard 2.4 — Training Preprocessing Pipeline
=================================================
Constructs leakage-safe scaled feature sets for all 5 space device families
from ASQD 2.4 TRAIN, VALIDATION, and BLIND_TEST split files.

Saves preprocessor artifacts to models/preprocessors/ for training and inference.
Strictly isolates validation and blind test splits from scaler fitting.
"""

import os
import json
import pandas as pd
import numpy as np

from astraguard_core.feature_engineering import feature_registry
from astraguard_core.preprocessing import LeakageSafePreprocessor

ASQD_DIR = "ASQD_2.4"
OUTPUT_DIR = "models/preprocessors"
PROCESSED_DATA_DIR = "ASQD_2.4/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

DEVICE_FAMILIES = [
    "DIGITAL_IC",
    "MIXED_SIGNAL_IC",
    "MEMS_GYROSCOPE",
    "IMAGE_SENSOR",
    "PRECISION_VOLTAGE_REF",
]

# Read split datasets
df_train_raw = pd.read_csv(os.path.join(ASQD_DIR, "asqd_24_train.csv"))
df_val_raw   = pd.read_csv(os.path.join(ASQD_DIR, "asqd_24_validation.csv"))
df_test_raw  = pd.read_csv(os.path.join(ASQD_DIR, "asqd_24_blind_test.csv"))

print("=" * 75)
print("ASTRAGUARD 2.4 — LEAKAGE-SAFE PREPROCESSING PIPELINE BUILD")
print("=" * 75)

pipeline_summary = {}

for dev_family in DEVICE_FAMILIES:
    print(f"\n[DEVICE FAMILY: {dev_family}]")

    # Filter split DataFrames by device family
    df_tr_fam = df_train_raw[df_train_raw["device_family"] == dev_family].copy()
    df_va_fam = df_val_raw[df_val_raw["device_family"] == dev_family].copy()
    df_te_fam = df_test_raw[df_test_raw["device_family"] == dev_family].copy()

    print(f"  Rows -> Train: {len(df_tr_fam)}, Val: {len(df_va_fam)}, Blind Test: {len(df_te_fam)}")

    # 1. Feature Engineering (Raw measurement -> domain physical features)
    X_train_raw, feat_names, target_col = feature_registry.extract_features(df_tr_fam, device_family=dev_family)
    
    # Calculate population stats ONLY on training set for leakage-safe z-scores
    med0, mad0 = X_train_raw["value_0h"].median(), 1.4826 * np.median(np.abs(X_train_raw["value_0h"] - X_train_raw["value_0h"].median()))
    med24, mad24 = X_train_raw["value_24h"].median(), 1.4826 * np.median(np.abs(X_train_raw["value_24h"] - X_train_raw["value_24h"].median()))
    train_pop_stats = {
        "value_0h": {"median": float(med0), "mad": float(mad0)},
        "value_24h": {"median": float(med24), "mad": float(mad24)},
    }

    # Extract features for Val & Test using TRAIN population stats (Zero lookahead leakage!)
    X_val_raw, _, _ = feature_registry.extract_features(df_va_fam, device_family=dev_family, population_stats=train_pop_stats)
    X_test_raw, _, _ = feature_registry.extract_features(df_te_fam, device_family=dev_family, population_stats=train_pop_stats)

    y_train = df_tr_fam[target_col].values
    y_val   = df_va_fam[target_col].values
    y_test  = df_te_fam[target_col].values

    # 2. Fit Preprocessor STRICTLY on Training Set
    preprocessor = LeakageSafePreprocessor(device_family=dev_family, use_log1p=True)
    preprocessor.fit(X_train_raw)

    # 3. Transform Train, Val, and Blind Test
    X_train_scaled = preprocessor.transform(X_train_raw)
    X_val_scaled   = preprocessor.transform(X_val_raw)
    X_test_scaled  = preprocessor.transform(X_test_raw)

    # Sanity checks
    assert not X_train_scaled.isna().any().any(), "Train scaled contains NaNs"
    assert not X_val_scaled.isna().any().any(), "Val scaled contains NaNs"
    assert not X_test_scaled.isna().any().any(), "Test scaled contains NaNs"

    # 4. Save Preprocessor State & Artifacts
    json_path = os.path.join(OUTPUT_DIR, f"{dev_family.lower()}_preprocessor.json")
    pkl_path  = os.path.join(OUTPUT_DIR, f"{dev_family.lower()}_preprocessor.pkl")
    
    sha_json = preprocessor.save(json_path)
    sha_pkl  = preprocessor.save(pkl_path)

    # Save processed matrices for training
    prefix = dev_family.lower()
    X_train_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_train.csv"), index=False)
    X_val_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_val.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_X_test.csv"), index=False)
    
    pd.DataFrame({"target": y_train}).to_csv(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_train.csv"), index=False)
    pd.DataFrame({"target": y_val}).to_csv(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_val.csv"), index=False)
    pd.DataFrame({"target": y_test}).to_csv(os.path.join(PROCESSED_DATA_DIR, f"{prefix}_y_test.csv"), index=False)

    print(f"  ✅ Fitted Preprocessor & Saved Artifacts:")
    print(f"     JSON: {json_path} (SHA-256: {sha_json[:12]}...)")
    print(f"     PKL:  {pkl_path} (SHA-256: {sha_pkl[:12]}...)")
    print(f"     Processed Matrices saved to {PROCESSED_DATA_DIR}/{prefix}_X_*.csv")

    pipeline_summary[dev_family] = {
        "features": feat_names,
        "n_features": len(feat_names),
        "target": target_col,
        "train_rows": len(df_tr_fam),
        "val_rows": len(df_va_fam),
        "test_rows": len(df_te_fam),
        "preprocessor_sha256": sha_json
    }

print("\n" + "=" * 75)
print("PREPROCESSING PIPELINE BUILD COMPLETE")
print("=" * 75)
with open(os.path.join(OUTPUT_DIR, "pipeline_summary.json"), "w") as f:
    json.dump(pipeline_summary, f, indent=2)
