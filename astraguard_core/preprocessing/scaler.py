"""
AstraGuard 2.4 — Robust Leakage-Safe Scaler
============================================
Provides outlier-robust scaling using Median and Median Absolute Deviation (MAD):
  x_scaled = (x - median_train) / (1.4826 * MAD_train)

Prevents extreme semiconductor degradation outliers (e.g., thermal runaway IDDQ spikes)
from distorting scale parameters. Strictly enforced training set parameter isolation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class RobustMADScaler:
    """
    Robust Scaler using Training Set Median and MAD.
    
    Formula:
      x_scaled = (x - median_train) / (1.4826 * MAD_train)
    """

    def __init__(self, feature_names: Optional[List[str]] = None, eps: float = 1e-6):
        self.feature_names = feature_names
        self.eps = eps
        self.medians_: Dict[str, float] = {}
        self.mads_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "RobustMADScaler":
        """Fit scaler parameters ONLY on training set X."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.feature_names is None:
            self.feature_names = list(X.columns)

        self.medians_ = {}
        self.mads_ = {}

        for col in self.feature_names:
            s = X[col].astype(float)
            med = float(s.median())
            mad_raw = float(np.median(np.abs(s - med)))
            mad_scaled = 1.4826 * mad_raw if mad_raw > self.eps else 1.0

            self.medians_[col] = med
            self.mads_[col] = mad_scaled

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned training set median/MAD parameters to X."""
        if not self.is_fitted_:
            raise RuntimeError("RobustMADScaler is not fitted yet. Call fit() first on training set.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        X_scaled = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            med = self.medians_[col]
            mad = self.mads_[col]
            X_scaled[col] = (X[col].astype(float) - med) / mad

        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on training X and return transformed training X."""
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Export parameters as dictionary for inspection and JSON serialization."""
        return {
            "scaler_type": "RobustMADScaler",
            "feature_names": self.feature_names,
            "medians": self.medians_,
            "mads": self.mads_,
            "is_fitted": self.is_fitted_
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustMADScaler":
        """Reconstruct scaler from parameter dictionary."""
        scaler = cls(feature_names=data.get("feature_names"))
        scaler.medians_ = data.get("medians", {})
        scaler.mads_ = data.get("mads", {})
        scaler.is_fitted_ = data.get("is_fitted", False)
        return scaler
