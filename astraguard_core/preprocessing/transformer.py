"""
AstraGuard 2.4 — Non-Linear Feature Transformers
=================================================
Provides skewness reduction and power transformations (Log1p, Soft-clipping)
for exponential parametric degradation distributions (IDDQ runaway, dark current spikes).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class Log1pTransformer:
    """
    Applies log(1 + x) transformation to non-negative skewed telemetry features.
    Saves and restores transformation state.
    """

    def __init__(self, log_features: Optional[List[str]] = None):
        self.log_features = log_features or []
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame) -> "Log1pTransformer":
        """Determine which features are non-negative and highly right-skewed (> 1.5)."""
        if not self.log_features:
            skewed = []
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    if (X[col] >= 0).all():
                        skew = float(X[col].skew())
                        if skew > 1.5:
                            skewed.append(col)
            self.log_features = skewed
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to fitted skewed feature columns."""
        X_trans = X.copy()
        for col in self.log_features:
            if col in X_trans.columns:
                X_trans[col] = np.log1p(np.maximum(0.0, X_trans[col].astype(float)))
        return X_trans

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transformer_type": "Log1pTransformer",
            "log_features": self.log_features,
            "is_fitted": self.is_fitted_
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Log1pTransformer":
        t = cls(log_features=data.get("log_features"))
        t.is_fitted_ = data.get("is_fitted", False)
        return t
