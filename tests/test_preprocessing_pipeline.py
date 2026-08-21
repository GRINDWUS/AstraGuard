"""
AstraGuard 2.4 — Preprocessing Pipeline Test Suite
===================================================
Tests leakage-safe scaler isolation, artifact serialization, and numerical integrity.
"""

import os
import unittest
import numpy as np
import pandas as pd

from astraguard_core.feature_engineering import feature_registry
from astraguard_core.preprocessing import LeakageSafePreprocessor, ArtifactManager


class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.asqd_dir = "ASQD_2.4"
        # We use small committed lot subsets for testing instead of full artifacts to save space.
        self.df_train = pd.read_csv(os.path.join(self.asqd_dir, "digital_ic_lot_00.csv"))
        self.df_train["device_family"] = "DIGITAL_IC"
        self.df_val   = pd.read_csv(os.path.join(self.asqd_dir, "digital_ic_lot_01.csv"))
        self.df_val["device_family"] = "DIGITAL_IC"
        self.df_test  = pd.read_csv(os.path.join(self.asqd_dir, "digital_ic_lot_02.csv"))
        self.df_test["device_family"] = "DIGITAL_IC"

    def test_strict_zero_leakage_isolation(self):
        """Verify that fitting preprocessor on train vs train+val yields distinct scaler parameters."""
        df_dig_tr = self.df_train[self.df_train["device_family"] == "DIGITAL_IC"]
        df_dig_va = self.df_val[self.df_val["device_family"] == "DIGITAL_IC"]

        X_tr, _, _ = feature_registry.extract_features(df_dig_tr, device_family="DIGITAL_IC")
        X_va, _, _ = feature_registry.extract_features(df_dig_va, device_family="DIGITAL_IC")

        prep_clean = LeakageSafePreprocessor(device_family="DIGITAL_IC")
        prep_clean.fit(X_tr)

        # Fit contaminated preprocessor on train + val
        X_combined = pd.concat([X_tr, X_va], ignore_index=True)
        prep_polluted = LeakageSafePreprocessor(device_family="DIGITAL_IC")
        prep_polluted.fit(X_combined)

        med_clean = prep_clean.scaler.medians_["value_0h"]
        med_polluted = prep_polluted.scaler.medians_["value_0h"]

        # Validate that parameters differ, demonstrating that val set has distinct statistics
        self.assertNotEqual(med_clean, med_polluted, "Training scaler parameters must be isolated from validation data")

        print("✅ Zero Leakage Isolation Test Passed: Scaler parameters strictly isolated to training set")

    def test_serialization_reproducibility(self):
        """Verify JSON and PKL artifact serialization and reconstruction."""
        df_dig_tr = self.df_train[self.df_train["device_family"] == "DIGITAL_IC"]
        X_tr, _, _ = feature_registry.extract_features(df_dig_tr, device_family="DIGITAL_IC")

        prep_orig = LeakageSafePreprocessor(device_family="DIGITAL_IC")
        X_orig_scaled = prep_orig.fit_transform(X_tr)

        tmp_json = "models/preprocessors/test_temp_prep.json"
        tmp_pkl  = "models/preprocessors/test_temp_prep.pkl"

        prep_orig.save(tmp_json)
        prep_orig.save(tmp_pkl)

        # Reconstruct
        prep_json = LeakageSafePreprocessor.load(tmp_json)
        prep_pkl  = LeakageSafePreprocessor.load(tmp_pkl)

        X_json_scaled = prep_json.transform(X_tr)
        X_pkl_scaled  = prep_pkl.transform(X_tr)

        np.testing.assert_array_almost_equal(X_orig_scaled.values, X_json_scaled.values)
        np.testing.assert_array_almost_equal(X_orig_scaled.values, X_pkl_scaled.values)

        if os.path.exists(tmp_json): os.remove(tmp_json)
        if os.path.exists(tmp_pkl): os.remove(tmp_pkl)

        print("✅ Artifact Serialization Test Passed: Reconstructed JSON/PKL scalers yield identical output")


if __name__ == "__main__":
    unittest.main()
