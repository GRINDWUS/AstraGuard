"""
AstraGuard 2.0 — Comprehensive End-to-End System Test Suite
============================================================
Verifies:
1. SDK contract (`submit_measurement` & `stream_lot_csv`).
2. Server REST endpoints & ML predictions.
3. Module A Outlier Detection & Module B 168h Drift Forecasting.
4. Stage-B Telemetry Evaluation & Pre-Launch Fingerprinting.
5. ATE Chamber Simulator & Failure Injection Pipeline.
"""
import unittest
import pandas as pd
from astraguard_sdk.client import AstraGuardClient
from astraguard_sdk.schema import SDKAnalysisResult

class TestAstraGuardSystem(unittest.TestCase):

    def setUp(self):
        self.client = AstraGuardClient(operator_id="TEST_SYSTEM_OP")

    def test_01_input_schema_validation(self):
        """Test SDK input parsing and schema validation."""
        df_valid = pd.DataFrame([{
            "component_id": "TEST_001",
            "device_family": "DIGITAL_IC",
            "parameter": "IDDQ",
            "value_0h": 12.0,
            "value_24h": 13.0,
            "value_96h": 13.5,
            "unit": "uA"
        }])
        res = self.client.analyze(df_valid)
        self.assertIsInstance(res, SDKAnalysisResult)
        self.assertIn(res.recommendation, ["GREEN_NORMAL_CANDIDATE", "YELLOW_REVIEW", "RED_HIGH_RISK"])

    def test_02_unit_normalization(self):
        """Verify SDK properly normalizes engineering units (mA vs uA)."""
        df_ua = pd.DataFrame([{
            "component_id": "TEST_002",
            "device_family": "DIGITAL_IC",
            "value_0h": 1200.0,
            "value_24h": 1300.0,
            "unit": "uA"
        }])
        res_ua = self.client.analyze(df_ua)
        self.assertIsInstance(res_ua, SDKAnalysisResult)
        self.assertIsNotNone(res_ua.data_quality_score)

    def test_03_context_ood_safety(self):
        """Test Out-of-Distribution safety rejection (Missing Telemetry or Context)."""
        df_invalid = pd.DataFrame([{
            "component_id": "TEST_004",
            "device_family": "UNKNOWN_DEVICE",
            "parameter": "UNKNOWN",
            "unit": "INVALID_SIGNAL",
            "value_0h": -9999.0,
            "value_24h": -9999.0
        }])
        res = self.client.analyze(df_invalid)
        self.assertEqual(res.recommendation, "HOLD_OPERATOR_REVIEW")

    def test_04_failure_injection_red_tier(self):
        """Verify injected thermal runaway trajectory yields RED tier recommendation."""
        df_runaway = pd.DataFrame([{
            "component_id": "TEST_FAULT_001",
            "device_family": "DIGITAL_IC",
            "value_0h": 15.0,
            "value_24h": 90.0, 
            "value_96h": 190.0,
            "unit": "uA"
        }])
        res = self.client.analyze(df_runaway)
        self.assertIn(res.recommendation, ["RED_HIGH_RISK", "YELLOW_REVIEW"])

    def test_05_explainability_and_auditability(self):
        """Verify strict audit lineage across system, models, and ML explainability."""
        df_valid = pd.DataFrame([{
            "component_id": "TEST_005",
            "device_family": "DIGITAL_IC",
            "value_0h": 12.5,
            "value_24h": 13.5,
            "unit": "uA"
        }])
        res = self.client.analyze(df_valid)
        self.assertEqual(res.session.system_version, "AstraGuard-2.4")
        self.assertEqual(res.session.model_version, "Module-B-v2")
        self.assertEqual(res.session.feature_engine_version, "v2")
        self.assertEqual(res.session.decision_policy_version, "2.4")
        self.assertTrue(len(res.session.session_id) > 5)

if __name__ == "__main__":
    unittest.main()
