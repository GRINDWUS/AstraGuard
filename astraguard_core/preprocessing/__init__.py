"""
AstraGuard 2.4 — Leakage-Safe Preprocessing Pipeline
=====================================================
Orchestrates context-aware feature scaling and non-linear transformations
with strict training set boundary isolation.

Guarantees zero data leakage from Validation or Blind Test sets into scaling statistics.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from astraguard_core.preprocessing.scaler import RobustMADScaler
from astraguard_core.preprocessing.transformer import Log1pTransformer
from astraguard_core.preprocessing.artifacts import ArtifactManager


class LeakageSafePreprocessor:
    """
    Leakage-Safe Preprocessing Pipeline for AstraGuard 2.4.
    
    Fits transformations strictly on training data and applies pre-learned parameters
    to validation and blind test sets without lookahead or split leakage.
    """

    def __init__(
        self,
        device_family: str = "DIGITAL_IC",
        use_log1p: bool = True,
        feature_names: Optional[List[str]] = None
    ):
        self.device_family = device_family
        self.use_log1p = use_log1p
        self.feature_names = feature_names

        self.transformer = Log1pTransformer() if use_log1p else None
        self.scaler = RobustMADScaler(feature_names=feature_names)
        self.is_fitted_ = False
        self.fit_metadata_: Dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame) -> "LeakageSafePreprocessor":
        """Fit preprocessing transformations ONLY on training set X_train."""
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        self.feature_names = list(X_train.columns)
        self.scaler.feature_names = self.feature_names

        X_curr = X_train.copy()
        if self.use_log1p and self.transformer:
            X_curr = self.transformer.fit_transform(X_curr)

        self.scaler.fit(X_curr)

        self.is_fitted_ = True
        self.fit_metadata_ = {
            "device_family": self.device_family,
            "use_log1p": self.use_log1p,
            "n_training_samples": len(X_train),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X using learned training parameters. Zero lookahead leakage."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor is not fitted yet. Call fit() on training set first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        X_curr = X.copy()
        if self.use_log1p and self.transformer:
            X_curr = self.transformer.transform(X_curr)

        X_scaled = self.scaler.transform(X_curr)
        return X_scaled

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training X_train and return transformed training set."""
        return self.fit(X_train).transform(X_train)

    def to_dict(self) -> Dict[str, Any]:
        """Export state dictionary for inspection and serialization."""
        return {
            "pipeline_type": "LeakageSafePreprocessor",
            "device_family": self.device_family,
            "use_log1p": self.use_log1p,
            "is_fitted": self.is_fitted_,
            "fit_metadata": self.fit_metadata_,
            "transformer": self.transformer.to_dict() if self.transformer else None,
            "scaler": self.scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeakageSafePreprocessor":
        """Reconstruct preprocessor instance from state dictionary."""
        prep = cls(
            device_family=data.get("device_family", "DIGITAL_IC"),
            use_log1p=data.get("use_log1p", True),
            feature_names=data.get("fit_metadata", {}).get("feature_names")
        )
        prep.is_fitted_ = data.get("is_fitted", False)
        prep.fit_metadata_ = data.get("fit_metadata", {})

        if data.get("transformer"):
            prep.transformer = Log1pTransformer.from_dict(data["transformer"])
        if data.get("scaler"):
            prep.scaler = RobustMADScaler.from_dict(data["scaler"])

        return prep

    def save(self, filepath: str) -> str:
        """Save preprocessor state to JSON/PKL artifact and return SHA-256 hash."""
        if filepath.endswith(".json"):
            return ArtifactManager.save_json(self.to_dict(), filepath)
        else:
            return ArtifactManager.save_pickle(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LeakageSafePreprocessor":
        """Load preprocessor instance from JSON or PKL artifact."""
        if filepath.endswith(".json"):
            data, _ = ArtifactManager.load_json(filepath)
            return cls.from_dict(data)
        else:
            return ArtifactManager.load_pickle(filepath)


__all__ = [
    "LeakageSafePreprocessor",
    "RobustMADScaler",
    "Log1pTransformer",
    "ArtifactManager",
]
