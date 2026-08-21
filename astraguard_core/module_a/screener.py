"""
AstraGuard 2.4 — Module A Statistical Anomaly Screener
======================================================
Implements population-level Robust Z-Score anomaly screening using Median Absolute
Deviation (MAD). Operates at 0h and 24h checkpoints prior to ML forecasting.

Equations:
  MAD = median(|X - median(X)|)
  Robust Z = 0.6745 * (X - median(X)) / MAD
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class ModuleAScreener:
    """Statistical anomaly screener using Robust Z / MAD population statistics."""

    def __init__(self, z_threshold: float = 3.5):
        """
        Args:
            z_threshold: Threshold above which a component is flagged as an anomaly.
                        Standard aerospace screening uses |Z_robust| > 3.5 [PE].
        """
        self.z_threshold = z_threshold

    @staticmethod
    def compute_robust_z(series: pd.Series) -> pd.Series:
        """Compute Robust Z-score for a pandas Series using MAD."""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0 or np.isnan(mad):
            mad = 1e-9
        return 0.6745 * (series - median) / mad

    def screen_population(
        self,
        df: pd.DataFrame,
        value_col: str = "value_24h",
        profile_col: str = "failure_mode_gt"
    ) -> Dict[str, Any]:
        """
        Screen a population (lot) of components at a specific test checkpoint.
        
        Returns:
          Dictionary containing anomaly counts, Z-scores, and performance metrics.
        """
        values = df[value_col]
        robust_z = self.compute_robust_z(values)

        is_flagged = robust_z.abs() > self.z_threshold

        # If ground-truth failure profile is available, compute DDR and FAR
        metrics = {
            "total_components": len(df),
            "flagged_anomalies": int(is_flagged.sum()),
            "flagged_rate_pct": float(100.0 * is_flagged.sum() / len(df)),
            "robust_z_scores": robust_z,
            "is_flagged": is_flagged
        }

        if profile_col in df.columns:
            is_anomaly_gt = df[profile_col] != "NOMINAL"
            is_nominal_gt = df[profile_col] == "NOMINAL"

            tp = (is_flagged & is_anomaly_gt).sum()
            fp = (is_flagged & is_nominal_gt).sum()
            tn = (~is_flagged & is_nominal_gt).sum()
            fn = (~is_flagged & is_anomaly_gt).sum()

            ddr = float(100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 100.0
            far = float(100.0 * fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            metrics.update({
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "defect_detection_rate_pct": ddr,
                "false_alarm_rate_pct": far,
            })

        return metrics
