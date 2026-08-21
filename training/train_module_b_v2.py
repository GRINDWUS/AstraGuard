"""
AstraGuard 2.4 — Module B v2 Training Pipeline
================================================
Trains v2 models using the same ASQD 2.4 train/val splits as v1.

KEY GUARANTEES:
  1. v1 artifacts in models/preprocessors/ and models/module_b/ are NEVER touched.
  2. v2 preprocessors saved to models/v2/preprocessors/
  3. v2 model artifacts saved to models/v2/module_b/
  4. Scaler fitted ONLY on train split. Val and test use train stats (zero leakage).
  5. value_168h_actual is ONLY read for y_train / y_val as the regression target.
  6. Blind test (asqd_24_blind_test.csv) is NOT used in training or val winner selection.

Same tournament logic as v1: Ridge, RandomForest, XGBoost, LightGBM.
Winner selected on val MAE only.
"""

import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from astraguard_core.feature_engineering.v2 import get_v2_engineer
from astraguard_core.preprocessing import LeakageSafePreprocessor

ASQD_DIR          = "ASQD_2.4"
OUT_PREPROC       = "models/v2/preprocessors"
OUT_MODEL         = "models/v2/module_b"
PROCESSED_V2_DIR  = "ASQD_2.4/processed_v2"

for d in [OUT_PREPROC, OUT_MODEL, PROCESSED_V2_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE_FAMILIES = [
    "DIGITAL_IC",
    "MIXED_SIGNAL_IC",
    "MEMS_GYROSCOPE",
    "IMAGE_SENSOR",
    "PRECISION_VOLTAGE_REF",
]

SPEC_LIMITS = {
    "DIGITAL_IC":            1150.0,
    "MIXED_SIGNAL_IC":       1150.0,
    "MEMS_GYROSCOPE":          25.0,
    "IMAGE_SENSOR":            25.0,
    "PRECISION_VOLTAGE_REF": 6800.0,
}

print("=" * 80)
print("ASTRAGUARD 2.4 — MODULE B v2 TRAINING (0h + 24h + 96h features)")
print("V1 ARTIFACTS NOT TOUCHED | Output: models/v2/")
print("=" * 80)

df_train = pd.read_csv(os.path.join(ASQD_DIR, "asqd_24_train.csv"))
df_val   = pd.read_csv(os.path.join(ASQD_DIR, "asqd_24_validation.csv"))
df_test  = pd.read_csv(os.path.join(ASQD_DIR, "asqd_24_blind_test.csv"))

print(f"  Train: {len(df_train)} rows | Val: {len(df_val)} rows | Blind: {len(df_test)} rows")
print(f"  Leakage guard: fitting scaler on TRAIN only. Test set is read-once at compare time.")

tournament_summary = {}

for fam in DEVICE_FAMILIES:
    print(f"\n{'=' * 80}")
    print(f"DEVICE FAMILY: {fam}")

    engineer = get_v2_engineer(fam)
    print(f"  Feature contract v2: {len(engineer.feature_names)} features")
    new_feats = [f for f in engineer.feature_names if "96h" in f or "velocity_24" in f
                 or "acceleration" in f or "growth_ratio" in f]
    print(f"  New 96h features: {new_feats}")

    df_tr = df_train[df_train["device_family"] == fam].copy()
    df_va = df_val[df_val["device_family"] == fam].copy()
    df_te = df_test[df_test["device_family"] == fam].copy()

    print(f"  Rows -> Train: {len(df_tr)}, Val: {len(df_va)}, Blind: {len(df_te)}")

    # -----------------------------------------------------------------
    # 1. Feature extraction — leakage-safe population stats from TRAIN
    # -----------------------------------------------------------------
    X_tr_raw = engineer.extract_features(df_tr)

    # Compute population stats on train for z-score leakage guard
    pstats = {}
    for col in ["value_0h", "value_24h", "value_96h"]:
        med = float(X_tr_raw[col].median())
        mad = 1.4826 * float(np.median(np.abs(X_tr_raw[col] - med)))
        pstats[col] = {"median": med, "mad": max(mad, 1e-6)}

    X_va_raw = engineer.extract_features(df_va, population_stats=pstats)
    X_te_raw = engineer.extract_features(df_te, population_stats=pstats)

    y_tr = df_tr["value_168h_actual"].values
    y_va = df_va["value_168h_actual"].values
    y_te = df_te["value_168h_actual"].values

    # -----------------------------------------------------------------
    # 2. Fit LeakageSafePreprocessor on TRAIN only
    # -----------------------------------------------------------------
    preprocessor = LeakageSafePreprocessor(device_family=fam, use_log1p=True)
    preprocessor.fit(X_tr_raw)

    X_tr = preprocessor.transform(X_tr_raw)
    X_va = preprocessor.transform(X_va_raw)
    X_te = preprocessor.transform(X_te_raw)

    assert not X_tr.isna().any().any(), "Train NaN after scaling"
    assert not X_va.isna().any().any(), "Val NaN after scaling"
    assert not X_te.isna().any().any(), "Test NaN after scaling"

    # Save preprocessor
    pkl_path = os.path.join(OUT_PREPROC, f"{fam.lower()}_preprocessor_v2.pkl")
    preprocessor.save(pkl_path)

    # Save processed matrices
    pref = fam.lower()
    X_tr.to_csv(os.path.join(PROCESSED_V2_DIR, f"{pref}_X_train_v2.csv"), index=False)
    X_va.to_csv(os.path.join(PROCESSED_V2_DIR, f"{pref}_X_val_v2.csv"),   index=False)
    X_te.to_csv(os.path.join(PROCESSED_V2_DIR, f"{pref}_X_test_v2.csv"),  index=False)

    # -----------------------------------------------------------------
    # 3. Model tournament — same candidates as v1 training
    # -----------------------------------------------------------------
    candidates = {
        "Ridge":        Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              random_state=42, verbosity=0, n_jobs=-1)
    if HAS_LGB:
        candidates["LightGBM"] = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8,
                                               random_state=42, verbose=-1, n_jobs=-1)

    leaderboard = []
    for model_name, model in candidates.items():
        model.fit(X_tr, y_tr)
        y_va_pred = model.predict(X_va)
        val_mae = mean_absolute_error(y_va, y_va_pred)
        val_r2  = r2_score(y_va, y_va_pred)
        leaderboard.append({"name": model_name, "model": model,
                             "val_mae": val_mae, "val_r2": val_r2})
        print(f"    {model_name:15s} | Val MAE: {val_mae:9.4f} | Val R2: {val_r2:.4f}")

    leaderboard.sort(key=lambda x: x["val_mae"])
    winner = leaderboard[0]
    print(f"  WINNER: {winner['name']} | Val MAE: {winner['val_mae']:.4f}")

    # -----------------------------------------------------------------
    # 4. Lock winner — save model artifact with SHA-256
    # -----------------------------------------------------------------
    model_path = os.path.join(OUT_MODEL, f"{fam.lower()}_module_b_v2.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(winner["model"], f)
    sha256 = hashlib.sha256(open(model_path, "rb").read()).hexdigest()

    # -----------------------------------------------------------------
    # 5. Evaluate on blind test — ONE SHOT, results used only in compare script
    # -----------------------------------------------------------------
    y_te_pred = winner["model"].predict(X_te)
    blind_mae  = mean_absolute_error(y_te, y_te_pred)
    blind_rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
    blind_r2   = r2_score(y_te, y_te_pred)

    spec = SPEC_LIMITS[fam]
    escaped    = int(np.sum((y_te > spec) & (y_te_pred <= spec)))
    total_def  = int(np.sum(y_te > spec))
    esc_rate   = 100.0 * escaped / max(1, total_def)

    print(f"  Blind MAE: {blind_mae:.4f} | R2: {blind_r2:.4f} | Escape rate: {esc_rate:.2f}%")

    tournament_summary[fam] = {
        "version": "v2",
        "n_features": len(engineer.feature_names),
        "new_96h_features": new_feats,
        "winning_model": winner["name"],
        "val_mae": winner["val_mae"],
        "val_r2": winner["val_r2"],
        "blind_test_mae": blind_mae,
        "blind_test_rmse": blind_rmse,
        "blind_test_r2": blind_r2,
        "blind_escape_rate_pct": esc_rate,
        "blind_escaped_defects": escaped,
        "blind_total_defects": total_def,
        "model_sha256": sha256,
        "population_stats": pstats,
    }

    # Save metadata
    meta_path = os.path.join(OUT_MODEL, f"{fam.lower()}_metadata_v2.json")
    with open(meta_path, "w") as f:
        json.dump(tournament_summary[fam], f, indent=2)

print("\n" + "=" * 80)
print("V2 TRAINING COMPLETE")
summary_path = os.path.join(OUT_MODEL, "v2_tournament_summary.json")
with open(summary_path, "w") as f:
    json.dump(tournament_summary, f, indent=2)
print(f"Summary saved to {summary_path}")
print("=" * 80)
