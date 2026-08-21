"""
AstraGuard 2.4 — Feature Engineer Registry
============================================
Central registry routing resolved device contexts to their respective feature engineering contracts.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

from astraguard_core.feature_engineering.base import BaseFeatureEngineer
from astraguard_core.feature_engineering.digital_ic import DigitalICFeatureEngineer
from astraguard_core.feature_engineering.mixed_signal import MixedSignalFeatureEngineer
from astraguard_core.feature_engineering.mems_gyro import MEMSGyroFeatureEngineer
from astraguard_core.feature_engineering.image_sensor import ImageSensorFeatureEngineer
from astraguard_core.feature_engineering.voltage_reference import VoltageReferenceFeatureEngineer


class FeatureEngineerRegistry:
    """Registry routing device families to specialized feature engineering implementations."""

    def __init__(self):
        self._engineers: Dict[str, BaseFeatureEngineer] = {
            "DIGITAL_IC": DigitalICFeatureEngineer(),
            "MIXED_SIGNAL_IC": MixedSignalFeatureEngineer(),
            "MEMS_GYROSCOPE": MEMSGyroFeatureEngineer(),
            "IMAGE_SENSOR": ImageSensorFeatureEngineer(),
            "PRECISION_VOLTAGE_REF": VoltageReferenceFeatureEngineer(),
        }
        # Default fallback
        self._default_engineer = DigitalICFeatureEngineer()

    def get_engineer(self, device_family: Optional[str]) -> BaseFeatureEngineer:
        """
        Retrieve feature engineer for a specific device family.
        Falls back to DigitalICFeatureEngineer if context is unspecified.
        """
        if not device_family:
            return self._default_engineer
        
        family_clean = str(device_family).upper().strip()
        return self._engineers.get(family_clean, self._default_engineer)

    def extract_features(
        self,
        df: pd.DataFrame,
        device_family: Optional[str] = None,
        context_resolution: Optional[Dict[str, Any]] = None,
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Route DataFrame to appropriate feature engineer and extract features.
        
        Returns:
            Tuple of (features_df, feature_names_list, target_name_str)
        """
        resolved_family = device_family
        if not resolved_family and context_resolution:
            resolved_family = context_resolution.get("resolved_device_family")
        if not resolved_family and "device_family" in df.columns and not df["device_family"].isna().all():
            resolved_family = df["device_family"].iloc[0]

        engineer = self.get_engineer(resolved_family)
        features_df = engineer.extract_features(
            df=df,
            context_resolution=context_resolution,
            population_stats=population_stats
        )
        
        return features_df, engineer.feature_names, engineer.target_name


# Global singleton instance
feature_registry = FeatureEngineerRegistry()
