"""
AstraGuard 2.4 — Module A Statistical Screening Test Suite
============================================================
Tests Robust Z/MAD anomaly screener against known population outliers.
"""

import unittest
import numpy as np
import pandas as pd
from astraguard_core.module_a import ModuleAScreener


class TestModuleAScreener(unittest.TestCase):

    def test_robust_z_computation(self):
        # 100 nominal components around 10.0, 1 outlier at 100.0
        data = np.random.normal(10.0, 0.1, 100)
        data[50] = 100.0
        df = pd.DataFrame({"value_24h": data, "failure_mode_gt": ["NOMINAL"] * 100})
        df.loc[50, "failure_mode_gt"] = "SPATIAL_OUTLIER"

        screener = ModuleAScreener(z_threshold=3.5)
        metrics = screener.screen_population(df, value_col="value_24h")

        self.assertEqual(metrics["flagged_anomalies"], 1)
        self.assertEqual(metrics["false_alarm_rate_pct"], 0.0)
        self.assertEqual(metrics["defect_detection_rate_pct"], 100.0)
        print("✅ Module A screener test passed")


if __name__ == "__main__":
    unittest.main()
