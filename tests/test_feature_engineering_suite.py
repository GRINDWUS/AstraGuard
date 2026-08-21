"""
AstraGuard 2.4 — Feature Engineering Suite Test
=================================================
Tests all 5 device-specific feature contracts against regenerated ASQD 2.4 datasets.
"""

import os
import unittest
import pandas as pd
import numpy as np
from astraguard_core.feature_engineering import feature_registry


class TestFeatureEngineeringSuite(unittest.TestCase):

    def setUp(self):
        self.asqd_dir = "ASQD_2.4"
        self.device_families = [
            "DIGITAL_IC",
            "MIXED_SIGNAL_IC",
            "MEMS_GYROSCOPE",
            "IMAGE_SENSOR",
            "PRECISION_VOLTAGE_REF",
        ]

    def test_all_feature_engineers(self):
        """Verify that all 5 device families extract features cleanly without NaN/Nulls."""
        for dev_family in self.device_families:
            filename = f"{dev_family.lower()}_lot_00.csv"
            filepath = os.path.join(self.asqd_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"File {filepath} must exist")

            df = pd.read_csv(filepath)
            self.assertGreater(len(df), 0, "DataFrame should not be empty")

            features_df, feature_names, target_name = feature_registry.extract_features(
                df=df,
                device_family=dev_family
            )

            # Assertions
            self.assertEqual(len(features_df), len(df), f"{dev_family}: Row count must match")
            self.assertEqual(list(features_df.columns), feature_names, f"{dev_family}: Column schema must match feature_names")
            self.assertFalse(features_df.isna().any().any(), f"{dev_family}: Features must not contain NaN")
            self.assertFalse(np.isinf(features_df).any().any(), f"{dev_family}: Features must not contain Inf")
            self.assertTrue(target_name in df.columns, f"{dev_family}: Target column '{target_name}' must exist in raw dataset")

            print(f"✅ {dev_family:25s}: {len(feature_names)} features extracted cleanly across {len(features_df)} rows")


if __name__ == "__main__":
    unittest.main()
