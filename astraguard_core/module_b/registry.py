"""
AstraGuard 2.4 — Module B Model Registry
=========================================
Registry managing locked trained Module B regressor models across all 5 device families.
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class ModuleBRegistry:
    """Registry providing locked trained Module B models for 168h degradation inference."""

    def __init__(self, models_dir: str = "models/module_b"):
        self.models_dir = models_dir
        self._loaded_models: Dict[str, Any] = {}

    def load_model(self, device_family: str) -> Any:
        """Load trained Module B model for a specific device family."""
        family_clean = str(device_family).upper().strip()
        if family_clean in self._loaded_models:
            return self._loaded_models[family_clean]

        filepath = os.path.join(self.models_dir, f"{family_clean.lower()}_module_b.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Module B model artifact not found for {family_clean}: {filepath}")

        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        model = payload["model"]
        self._loaded_models[family_clean] = model
        return model

    def predict_168h(self, device_family: str, X_features: pd.DataFrame) -> np.ndarray:
        """Execute 168h degradation prediction using locked device-specific model."""
        model = self.load_model(device_family)
        return model.predict(X_features)


# Global singleton instance
module_b_registry = ModuleBRegistry()
