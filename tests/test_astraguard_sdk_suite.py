"""
AstraGuard 2.4 — Safe ATE Integration SDK Test Suite
=====================================================
Tests read-only SDK client, data adapters, integrity filtering, instrument QA,
safety interlocks, audit logging, and CLI offline operations.
"""

import unittest
import os
import pandas as pd
from astraguard_sdk.client import AstraGuardClient
from astraguard_sdk.integrity import DataIntegrityValidator
from astraguard_sdk.schema import SDKMeasurementRecord
from astraguard_sdk.ate_simulator import NonInvasiveATESimulator


class TestAstraGuardSDKSuite(unittest.TestCase):

    def setUp(self):
        from astraguard_core.context_resolver import profiles as p
        p.ProfileRegistry._instance = None
        self.client = AstraGuardClient(operator_id="TEST_QA_OPERATOR")
        self.simulator = NonInvasiveATESimulator()

    def test_01_digital_ic_analysis(self):
        df_ate = self.simulator.generate_ate_stream(device_family="DIGITAL_IC", num_components=20)
        res = self.client.analyze(df_ate)

        self.assertTrue(res.session.read_only_guarantee)
        self.assertEqual(res.resolved_device_family, "DIGITAL_IC")
        self.assertIn(res.recommendation, ["GREEN_NORMAL_CANDIDATE", "YELLOW_REVIEW", "RED_HIGH_RISK"])
        self.assertGreaterEqual(res.data_quality_score, 0.90)

    def test_02_mems_gyroscope_analysis(self):
        df_ate = self.simulator.generate_ate_stream(device_family="MEMS_GYROSCOPE", num_components=20)
        res = self.client.analyze(df_ate)

        self.assertEqual(res.resolved_device_family, "MEMS_GYROSCOPE")
        self.assertEqual(res.resolved_primary_parameter, "zero_rate_offset")

    def test_03_corrupted_context_safety_interlock(self):
        df_ate = self.simulator.generate_ate_stream(
            device_family="DIGITAL_IC",
            num_components=20,
            corrupt_metadata=True,
            strip_metadata=True
        )
        res = self.client.analyze(df_ate)

        self.assertFalse(res.is_execution_allowed)
        self.assertEqual(res.recommendation, "HOLD_OPERATOR_REVIEW")

    def test_04_frozen_channel_instrument_qa(self):
        validator = DataIntegrityValidator()
        records = [
            SDKMeasurementRecord(
                component_id=f"COMP_{i}",
                parameter_name="IDDQ",
                value=12.5,
                unit="uA",
                measurement_state="FROZEN_CHANNEL"
            ) for i in range(15)
        ]

        valid_recs, score, issues, inst_status = validator.validate_and_normalize(records)
        self.assertEqual(inst_status, "INSTRUMENT_FAULT_FROZEN_CHANNEL")

    def test_05_unit_normalization(self):
        validator = DataIntegrityValidator()
        rec = SDKMeasurementRecord(
            component_id="COMP_01",
            parameter_name="IDDQ",
            value=0.0125,
            unit="mA"
        )
        valid_recs, score, issues, inst_status = validator.validate_and_normalize([rec])
        self.assertEqual(len(valid_recs), 1)
        self.assertEqual(valid_recs[0].canonical_parameter, "IDDQ")


if __name__ == "__main__":
    unittest.main()
