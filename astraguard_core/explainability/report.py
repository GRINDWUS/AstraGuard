"""
AstraGuard 2.4 — QA Inspector Explainability Report Generator
==============================================================
Generates human-readable, panel-ready physical explanation reports for QA Inspectors
meeting PS #26170 requirements.
"""

from typing import Dict, Any, List
import pandas as pd

from astraguard_core.explainability.shap_engine import SHAPExplainabilityEngine
from astraguard_core.explainability.physics_mapper import PhysicsAttributionMapper


class QAExplanationReportGenerator:
    """Generates QA-ready explainability reports."""

    @staticmethod
    def generate_report(
        component_id: str,
        device_family: str,
        predicted_168h: float,
        spec_limit: float,
        shap_engine: SHAPExplainabilityEngine,
        X_single: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate complete QA Inspector report.
        """
        shap_res = shap_engine.explain_component(X_single)
        attributions = shap_res["feature_attributions"]

        physical_explanations = PhysicsAttributionMapper.map_attributions(
            device_family=device_family,
            shap_attributions=attributions,
            top_k=3
        )

        is_anomalous = predicted_168h >= spec_limit if spec_limit else False

        summary_lines = [
            f"Component ID {component_id} ({device_family}) predicted 168h value is {predicted_168h:.4f} "
            f"(USL Limit: {spec_limit if spec_limit else 'N/A'})."
        ]

        if is_anomalous:
            summary_lines.append("RECOMMENDATION: FLAG FOR EARLY REJECTION due to predicted parametric limit breach.")
        else:
            summary_lines.append("RECOMMENDATION: PASS — Predicted 168h trajectory remains within safe spec limits.")

        summary_lines.append("Top Physical Degradation Contributors:")
        for exp in physical_explanations:
            summary_lines.append(f"  • {exp['qa_summary']}")

        return {
            "component_id": component_id,
            "device_family": device_family,
            "predicted_168h_value": predicted_168h,
            "spec_limit": spec_limit,
            "is_anomalous_forecast": is_anomalous,
            "base_value": shap_res["base_value"],
            "top_physical_contributors": physical_explanations,
            "qa_inspector_summary": "\n".join(summary_lines)
        }
