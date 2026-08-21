"""
AstraGuard 2.4 — Feature Engineering v2 Package
================================================
Registry of v2 feature engineers (0h + 24h + 96h feature set).
V1 engineers in the parent package are NOT modified.
"""

from astraguard_core.feature_engineering.v2.digital_ic_v2 import DigitalICFeatureEngineerV2
from astraguard_core.feature_engineering.v2.mems_gyro_v2 import MEMSGyroFeatureEngineerV2
from astraguard_core.feature_engineering.v2.image_sensor_v2 import ImageSensorFeatureEngineerV2
from astraguard_core.feature_engineering.v2.mixed_signal_v2 import MixedSignalFeatureEngineerV2
from astraguard_core.feature_engineering.v2.voltage_reference_v2 import VoltageReferenceFeatureEngineerV2

V2_ENGINEERS = {
    "DIGITAL_IC":           DigitalICFeatureEngineerV2(),
    "MEMS_GYROSCOPE":       MEMSGyroFeatureEngineerV2(),
    "IMAGE_SENSOR":         ImageSensorFeatureEngineerV2(),
    "MIXED_SIGNAL_IC":      MixedSignalFeatureEngineerV2(),
    "PRECISION_VOLTAGE_REF": VoltageReferenceFeatureEngineerV2(),
}


def get_v2_engineer(device_family: str):
    eng = V2_ENGINEERS.get(device_family)
    if eng is None:
        raise ValueError(f"No v2 engineer registered for device family: {device_family}")
    return eng
