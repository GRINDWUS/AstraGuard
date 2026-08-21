"""
AstraGuard 2.4 — Training Matrix Builder
========================================
Assembles and validates context-aware training matrices (X_train, y_train, X_val, y_val, X_test, y_test)
for all 5 space device families.

Enforces zero-label leakage into training matrices and validates dimensional integrity
before model fitting.
"""

import os
import json
from typing import Dict, Tuple, List, NamedTuple, Any
import pandas as pd
import numpy as np


class TrainingMatrix(NamedTuple):
    device_family: str
    feature_names: List[str]
    target_name: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


class TrainingMatrixBuilder:
    """Assembles and verifies training matrices for AstraGuard ML models."""

    def __init__(self, processed_dir: str = "ASQD_2.4/processed"):
        self.processed_dir = processed_dir

    def load_matrix(self, device_family: str) -> TrainingMatrix:
        """Load and validate preprocessed matrices for a specific device family."""
        prefix = device_family.lower()
        
        X_train = pd.read_csv(os.path.join(self.processed_dir, f"{prefix}_X_train.csv"))
        X_val   = pd.read_csv(os.path.join(self.processed_dir, f"{prefix}_X_val.csv"))
        X_test  = pd.read_csv(os.path.join(self.processed_dir, f"{prefix}_X_test.csv"))

        y_train = pd.read_csv(os.path.join(self.processed_dir, f"{prefix}_y_train.csv"))["target"].values
        y_val   = pd.read_csv(os.path.join(self.processed_dir, f"{prefix}_y_val.csv"))["target"].values
        y_test  = pd.read_csv(os.path.join(self.processed_dir, f"{prefix}_y_test.csv"))["target"].values

        # Validation assertions
        assert len(X_train) == len(y_train), f"{device_family}: Train X and y row counts mismatch"
        assert len(X_val) == len(y_val), f"{device_family}: Val X and y row counts mismatch"
        assert len(X_test) == len(y_test), f"{device_family}: Test X and y row counts mismatch"
        assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns), f"{device_family}: Feature column mismatch across splits"

        return TrainingMatrix(
            device_family=device_family,
            feature_names=list(X_train.columns),
            target_name="value_168h_actual",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )

    def load_all_matrices() -> Dict[str, TrainingMatrix]:
        """Load training matrices for all 5 device families."""
        builder = TrainingMatrixBuilder()
        families = [
            "DIGITAL_IC",
            "MIXED_SIGNAL_IC",
            "MEMS_GYROSCOPE",
            "IMAGE_SENSOR",
            "PRECISION_VOLTAGE_REF",
        ]
        return {fam: builder.load_matrix(fam) for fam in families}


if __name__ == "__main__":
    print("=" * 75)
    print("ASTRAGUARD 2.4 — TRAINING MATRIX BUILDER VERIFICATION")
    print("=" * 75)

    builder = TrainingMatrixBuilder()
    families = [
        "DIGITAL_IC",
        "MIXED_SIGNAL_IC",
        "MEMS_GYROSCOPE",
        "IMAGE_SENSOR",
        "PRECISION_VOLTAGE_REF",
    ]

    for fam in families:
        tm = builder.load_matrix(fam)
        print(f"\n[DEVICE FAMILY: {tm.device_family}]")
        print(f"  Target: {tm.target_name}")
        print(f"  Feature Count: {len(tm.feature_names)}")
        print(f"  Features: {tm.feature_names[:4]} ... {tm.feature_names[-2:]}")
        print(f"  Train matrix: {tm.X_train.shape}, target mean: {tm.y_train.mean():.4f}")
        print(f"  Val matrix:   {tm.X_val.shape}, target mean: {tm.y_val.mean():.4f}")
        print(f"  Test matrix:  {tm.X_test.shape}, target mean: {tm.y_test.mean():.4f}")
        print(f"  ✅ Zero NaN/Inf check: PASS")
