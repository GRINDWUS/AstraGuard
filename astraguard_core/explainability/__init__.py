"""
AstraGuard 2.4 — Explainability Package
========================================
Provides SHAP feature attribution and physics mapping for QA inspectors.
"""

from astraguard_core.explainability.shap_engine import SHAPExplainabilityEngine
from astraguard_core.explainability.physics_mapper import PhysicsAttributionMapper
from astraguard_core.explainability.report import QAExplanationReportGenerator

__all__ = [
    "SHAPExplainabilityEngine",
    "PhysicsAttributionMapper",
    "QAExplanationReportGenerator",
]
