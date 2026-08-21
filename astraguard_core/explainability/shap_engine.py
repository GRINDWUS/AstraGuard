"""
AstraGuard 2.4 — SHAP Explainability Engine
============================================
Calculates SHAP feature contribution values for Module B degradation forecasts.
Uses LinearExplainer for Ridge models and TreeExplainer for Tree models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class SHAPExplainabilityEngine:
    """Computes feature attributions for Module B predictions."""

    def __init__(self, model: Any, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names

        import shap  # Lazy import to avoid LLVM/numba C-extension conflicts during unpickling on Windows

        # Initialize appropriate SHAP explainer
        if hasattr(model, "coef_"):
            # Linear / Ridge model
            self.explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(data=np.zeros((1, len(model.coef_)))))
            self.explainer_type = "LinearExplainer"
        else:
            # Tree model (RandomForest / XGBoost)
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = "TreeExplainer"

    def explain_component(self, X_single: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute SHAP values for a single component feature vector.
        
        Args:
            X_single: Single row DataFrame of engineered/scaled features.
            
        Returns:
            Dictionary of feature contributions and base value.
        """
        shap_values = self.explainer(X_single)
        
        values = shap_values.values[0]
        base_val = shap_values.base_values[0] if hasattr(shap_values, "base_values") else 0.0

        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        feature_cols = list(X_single.columns)
        attributions = {}
        for col, val in zip(feature_cols, values):
            attributions[col] = float(val)

        # Sort attributions by absolute magnitude
        sorted_attributions = dict(sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "base_value": base_val,
            "predicted_value": float(base_val + sum(values)),
            "feature_attributions": sorted_attributions
        }
