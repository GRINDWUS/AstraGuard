"""
AstraGuard 2.2 — Context Discovery & Instrument QA Test Suite
Validates Level 1, 2, 3 Test Identity Resolvers, Device Profiles, and Instrument Health Models.
"""

import unittest
import pandas as pd
from astraguard_core.context_resolver.schema import (
    TestContext,
    DeviceMetadata,
    TestMetadata,
    MeasurementRecord,
    ParameterCategory,
    IdentificationSource,
    ResolutionStatus,
)
from astraguard_core.context_resolver.profiles import ProfileRegistry
from astraguard_core.context_resolver.explicit_parser import ExplicitMetadataParser
from astraguard_core.context_resolver.behavioral_infer import BehavioralInferenceEngine
from astraguard_core.instrument_qa.health_model import InstrumentHealthModel
from dataset_generator.lot_generator import LotSimulator


class TestContextDiscoverySuite(unittest.TestCase):

    def setUp(self):
        from astraguard_core.context_resolver import profiles as p
        p.ProfileRegistry._instance = None

    def test_profile_registry_loading(self):
        registry = ProfileRegistry()
        dev_families = registry.list_device_families()
        test_types = registry.list_test_types()

        self.assertIn("DIGITAL_IC", dev_families)
        self.assertIn("MEMS_GYROSCOPE", dev_families)
        self.assertIn("IMAGE_SENSOR", dev_families)
        self.assertIn("THERMAL_BURN_IN", test_types)

        mems_prof = registry.get_device_profile("MEMS_GYROSCOPE")
        self.assertEqual(mems_prof.primary_parameter, "zero_rate_offset")
        self.assertIn("MEMS_STICTION", mems_prof.physical_failure_modes)

    def test_level_1_explicit_resolution(self):
        parser = ExplicitMetadataParser()
        ctx = TestContext(
            device_metadata=DeviceMetadata(device_family="MEMS_GYROSCOPE"),
            test_metadata=TestMetadata(test_type="THERMAL_BURN_IN")
        )
        res = parser.resolve(test_context=ctx, observed_parameters=["zero_rate_offset", "supply_current", "resonant_frequency"])

        self.assertEqual(res.resolved_device_family, "MEMS_GYROSCOPE")
        self.assertEqual(res.resolved_test_type, "THERMAL_BURN_IN")
        self.assertGreaterEqual(res.confidence_score, 0.95)
        self.assertEqual(res.identification_source, IdentificationSource.EXPLICIT_METADATA)

    def test_level_2_schema_resolution(self):
        parser = ExplicitMetadataParser()
        res = parser.resolve(observed_parameters=["dark_current_density", "DSNU", "hot_pixel_count"])

        self.assertEqual(res.resolved_device_family, "IMAGE_SENSOR")
        self.assertGreaterEqual(res.confidence_score, 0.75)
        self.assertEqual(res.identification_source, IdentificationSource.METADATA_MAPPING)

    def test_level_3_behavioral_inference(self):
        infer_engine = BehavioralInferenceEngine()
        records = [
            MeasurementRecord(component_id="COMP_01", parameter_name="bias_offset", value=0.045, unit="dps"),
            MeasurementRecord(component_id="COMP_01", parameter_name="bias_offset", value=0.048, unit="dps"),
        ]
        res = infer_engine.infer(records=records)

        self.assertEqual(res.resolved_device_family, "MEMS_GYROSCOPE")
        self.assertGreaterEqual(res.confidence_score, 0.75)
        self.assertEqual(res.identification_source, IdentificationSource.BEHAVIORAL_INFERENCE)

    def test_instrument_qa_adc_stuck(self):
        health_model = InstrumentHealthModel()
        records = [
            {"component_id": "COMP_01", "checkpoint_name": "0h", "value": 10.0},
            {"component_id": "COMP_01", "checkpoint_name": "24h", "value": 10.0},
            {"component_id": "COMP_01", "checkpoint_name": "96h", "value": 10.0},
            {"component_id": "COMP_02", "checkpoint_name": "0h", "value": 10.0},
            {"component_id": "COMP_02", "checkpoint_name": "24h", "value": 10.0},
            {"component_id": "COMP_02", "checkpoint_name": "96h", "value": 10.0},
        ]
        status = health_model.evaluate(lot_measurements=records)

        self.assertFalse(status.is_instrument_healthy)
        self.assertEqual(status.fault_type, "ADC_STUCK_DATA_LOGGER_FAULT")

    def test_asqd_2_multi_device_generator_backward_compatibility(self):
        sim = LotSimulator()
        df_mems = sim.generate_lot(lot_id=1, num_components=50, device_family="MEMS_GYROSCOPE")

        self.assertIn("iddq_0h", df_mems.columns)
        self.assertIn("iddq_24h", df_mems.columns)
        self.assertIn("iddq_168h_actual", df_mems.columns)
        self.assertIn("delta_iddq", df_mems.columns)

        self.assertIn("device_family", df_mems.columns)
        self.assertEqual(df_mems["device_family"].iloc[0], "MEMS_GYROSCOPE")
        self.assertIn("test_type", df_mems.columns)
        self.assertIn("instrument_status", df_mems.columns)


if __name__ == "__main__":
    unittest.main()
