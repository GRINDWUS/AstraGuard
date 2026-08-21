"""
AstraGuard 2.4 — Base Feature Engineer Interface
=================================================
Abstract base class defining the contract for all device-specific feature engineers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class BaseFeatureEngineer(ABC):
    """Abstract Base Class for device-specific feature engineering in AstraGuard 2.4."""

    @property
    @abstractmethod
    def device_family(self) -> str:
        """Return the device family string (e.g., 'DIGITAL_IC')."""
        pass

    @property
    @abstractmethod
    def primary_parameter(self) -> str:
        """Return the primary measurement parameter name (e.g., 'IDDQ')."""
        pass

    @property
    @abstractmethod
    def target_name(self) -> str:
        """Return the target prediction column name (e.g., 'value_168h_actual')."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return the list of engineered feature column names."""
        pass

    @abstractmethod
    def extract_features(
        self,
        df: pd.DataFrame,
        context_resolution: Optional[Dict[str, Any]] = None,
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Extract domain-specific physical and statistical features from raw ATE telemetry.
        
        Args:
            df: Input DataFrame containing raw ASQD measurement columns.
            context_resolution: Optional context resolver dictionary metadata.
            population_stats: Optional dict of precomputed lot population medians & MADs
                              to prevent data leakage during train/val/test splits.
                              Format: {'value_0h': {'median': float, 'mad': float}, ...}
        
        Returns:
            pd.DataFrame containing engineered feature columns ready for model input.
        """
        pass
