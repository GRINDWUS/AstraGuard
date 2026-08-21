"""
AstraGuard 2.4 — Explainability Test Suite
===========================================
Tests SHAP engine, physics mapper, and QA report generation.
"""

import unittest
import pandas as pd
import numpy as np

from astraguard_core.module_b import module_b_registry
from training.matrix_builder import TrainingMatrixBuilder
from astraguard_core.explainability import (
    SHAPExplainabilityEngine,
    PhysicsAttributionMapper,
    QAExplanationReportGenerator,
)


class TestExplainability(unittest.TestCase):

    def test_shap_and_physics_mapping(self):
        builder = TrainingMatrixBuilder()
        matrix = builder.load_matrix("DIGITAL_IC")
        model = module_b_registry.load_model("DIGITAL_IC")

        engine = SHAPExplainabilityEngine(model)
        X_single = matrix.X_test.iloc[[0]]

        explanation = engine.explain_component(X_single)

        self.assertIn("base_value", explanation)
        self.assertIn("feature_attributions", explanation)
        self.assertGreater(len(explanation["feature_attributions"]), 0)

        mapped = PhysicsAttributionMapper.map_attributions("DIGITAL_IC", explanation["feature_attributions"])
        self.assertEqual(len(mapped), 3)
        self.assertIn("physical_mechanism", mapped[0])

        report = QAExplanationReportGenerator.generate_report(
            component_id="COMP_TEST_001",
            device_family="DIGITAL_IC",
            predicted_168h=950.0,
            spec_limit=1150.0,
            shap_engine=engine,
            X_single=X_single
        )

        self.assertIn("qa_inspector_summary", report)
        print("✅ Explainability Test Passed")
        print("\nGenerated QA Report Preview:\n" + report["qa_inspector_summary"])


if __name__ == "__main__":
    unittest.main()
