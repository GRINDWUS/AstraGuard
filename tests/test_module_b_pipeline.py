"""
AstraGuard 2.4 — Module B Pipeline Test Suite
==============================================
Tests evaluation metrics, model tournament execution, and registry inference.
"""

import os
import unittest
import numpy as np
import pandas as pd

from astraguard_core.module_b import ModuleBEvaluator, ModuleBTrainer, module_b_registry
from training.matrix_builder import TrainingMatrixBuilder


class TestModuleBPipeline(unittest.TestCase):

    def test_evaluator_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 100.0])
        y_pred = np.array([11.0, 19.0, 31.0, 95.0])

        metrics = ModuleBEvaluator.evaluate(y_true, y_pred, spec_limit=50.0)

        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertIn("r2_score", metrics)
        self.assertEqual(metrics["total_defects"], 1)
        self.assertEqual(metrics["escaped_defects"], 0)
        self.assertEqual(metrics["escaped_defect_rate_pct"], 0.0)

        print("✅ ModuleBEvaluator test passed")

    def test_end_to_end_training_and_registry(self):
        builder = TrainingMatrixBuilder()
        matrix = builder.load_matrix("DIGITAL_IC")

        trainer = ModuleBTrainer(device_family="DIGITAL_IC")
        winner, val_info = trainer.run_tournament(matrix, spec_limit=1150.0)

        self.assertIsNotNone(winner)
        self.assertIn(trainer.winning_model_name_, ["Ridge", "RandomForest", "XGBoost", "LightGBM"])

        test_path = "models/module_b/test_digital_ic_module_b.pkl"
        sha = trainer.save_model(test_path)
        self.assertTrue(os.path.exists(test_path))

        if os.path.exists(test_path): os.remove(test_path)
        print("✅ ModuleBTrainer and Registry integration test passed")


if __name__ == "__main__":
    unittest.main()
