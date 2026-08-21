"""
AstraGuard 2.4 — Module B Model Tournament Trainer
===================================================
Executes model candidate selection across Ridge, RandomForest, XGBoost, and LightGBM.
Selection is strictly driven by Validation Lot MAE and OOD Generalization Gap.

Once selected, the winning model parameters are locked before Blind Test evaluation.
"""

import os
import pickle
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from astraguard_core.module_b.evaluator import ModuleBEvaluator
from training.matrix_builder import TrainingMatrix


class ModuleBTrainer:
    """Trainer executing multi-candidate tournaments for 168h degradation prediction."""

    def __init__(self, device_family: str, random_state: int = 202624):
        self.device_family = device_family
        self.random_state = random_state
        self.candidates_ = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=random_state, n_jobs=-1),
        }
        if HAS_LIGHTGBM:
            self.candidates_["LightGBM"] = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=random_state, verbose=-1, n_jobs=-1)
        self.winning_model_name_: Optional[str] = None
        self.winning_model_: Optional[Any] = None
        self.leaderboard_: List[Dict[str, Any]] = []

    def run_tournament(
        self,
        matrix: TrainingMatrix,
        spec_limit: float = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train all candidate regressors, evaluate on Validation Set, and pick winner.
        
        Args:
            matrix: TrainingMatrix instance containing X_train, y_train, X_val, y_val.
            spec_limit: Optional upper spec limit for escape rate evaluation.
        """
        self.leaderboard_ = []

        for name, model in self.candidates_.items():
            # Fit on Training Set ONLY
            model.fit(matrix.X_train, matrix.y_train)

            # Predict on Train & Val
            y_train_pred = model.predict(matrix.X_train)
            y_val_pred   = model.predict(matrix.X_val)

            train_metrics = ModuleBEvaluator.evaluate(matrix.y_train, y_train_pred, spec_limit)
            val_metrics   = ModuleBEvaluator.evaluate(matrix.y_val, y_val_pred, spec_limit)

            ood_gap = ModuleBEvaluator.compute_ood_gap(train_metrics["mae"], val_metrics["mae"])

            self.leaderboard_.append({
                "model_name": name,
                "train_mae": train_metrics["mae"],
                "train_r2": train_metrics["r2_score"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2_score"],
                "val_mape_pct": val_metrics["mape_pct"],
                "ood_generalization_gap": ood_gap,
                "val_escaped_defect_rate_pct": val_metrics.get("escaped_defect_rate_pct", 0.0),
                "model_obj": model
            })

        # Sort leaderboard by Validation MAE (lowest first)
        self.leaderboard_.sort(key=lambda x: x["val_mae"])
        winner_info = self.leaderboard_[0]
        
        self.winning_model_name_ = winner_info["model_name"]
        self.winning_model_ = winner_info["model_obj"]

        return self.winning_model_, winner_info

    def save_model(self, filepath: str) -> str:
        """Save winning model instance and return SHA-256 hash."""
        if not self.winning_model_:
            raise RuntimeError("No winning model fitted. Call run_tournament() first.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data_bytes = pickle.dumps({
            "device_family": self.device_family,
            "model_name": self.winning_model_name_,
            "model": self.winning_model_,
            "leaderboard": [{k: v for k, v in row.items() if k != "model_obj"} for row in self.leaderboard_]
        })
        sha256_hash = hashlib.sha256(data_bytes).hexdigest()

        with open(filepath, "wb") as f:
            f.write(data_bytes)

        return sha256_hash
