"""
AstraGuard 2.4 — Feature Engineering Package
=============================================
Provides context-aware feature engineering for 5 supported space device families:
  - DIGITAL_IC
  - MIXED_SIGNAL_IC
  - MEMS_GYROSCOPE
  - IMAGE_SENSOR
  - PRECISION_VOLTAGE_REF
"""

from astraguard_core.feature_engineering.base import BaseFeatureEngineer
from astraguard_core.feature_engineering.registry import FeatureEngineerRegistry, feature_registry
from astraguard_core.feature_engineering.digital_ic import DigitalICFeatureEngineer
from astraguard_core.feature_engineering.mixed_signal import MixedSignalFeatureEngineer
from astraguard_core.feature_engineering.mems_gyro import MEMSGyroFeatureEngineer
from astraguard_core.feature_engineering.image_sensor import ImageSensorFeatureEngineer
from astraguard_core.feature_engineering.voltage_reference import VoltageReferenceFeatureEngineer

__all__ = [
    "BaseFeatureEngineer",
    "FeatureEngineerRegistry",
    "feature_registry",
    "DigitalICFeatureEngineer",
    "MixedSignalFeatureEngineer",
    "MEMSGyroFeatureEngineer",
    "ImageSensorFeatureEngineer",
    "VoltageReferenceFeatureEngineer",
]
