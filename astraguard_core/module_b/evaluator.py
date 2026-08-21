"""
AstraGuard 2.4 — Module B Evaluation Protocol
==============================================
Defines aerospace reliability metrics for degradation forecasting:
  - MAE (Physical units)
  - MAPE (%)
  - R2 Score
  - OOD Lot Generalization Gap (|MAE_val - MAE_train|)
  - Escaped Defect Rate (0% target for space flight qualification)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class ModuleBEvaluator:
    """Evaluates degradation forecasting models against physical aerospace standards."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        spec_limit: float = None
    ) -> Dict[str, float]:
        """
        Compute standard regression and aerospace safety metrics.
        
        Args:
            y_true: Ground truth 168h values.
            y_pred: Model predicted 168h values.
            spec_limit: Optional upper specification limit for defect escape rate calculation.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # MAPE with zero protection
        denom = np.maximum(1e-6, np.abs(y_true))
        mape_pct = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

        # R2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / max(1e-9, ss_tot)))

        res = {
            "mae": mae,
            "rmse": rmse,
            "mape_pct": mape_pct,
            "r2_score": r2
        }

        # Defect Escape Rate calculation if upper spec limit is provided
        if spec_limit is not None:
            actual_defects = y_true > spec_limit
            predicted_passes = y_pred <= spec_limit
            false_negatives = np.sum(actual_defects & predicted_passes)
            total_defects = np.sum(actual_defects)
            escape_rate_pct = float((false_negatives / total_defects * 100.0) if total_defects > 0 else 0.0)
            res["total_defects"] = int(total_defects)
            res["escaped_defects"] = int(false_negatives)
            res["escaped_defect_rate_pct"] = escape_rate_pct

        return res

    @staticmethod
    def compute_ood_gap(train_mae: float, val_mae: float) -> float:
        """Compute Out-of-Distribution Lot Generalization Gap |MAE_val - MAE_train|."""
        return float(abs(val_mae - train_mae))
