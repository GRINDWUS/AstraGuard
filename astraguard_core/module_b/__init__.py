"""
AstraGuard 2.4 — Module B Degradation Forecasting Package
===========================================================
Provides 168h trajectory prediction using device-specific ML regressors.
"""

from astraguard_core.module_b.evaluator import ModuleBEvaluator
from astraguard_core.module_b.trainer import ModuleBTrainer
from astraguard_core.module_b.registry import ModuleBRegistry, module_b_registry

__all__ = [
    "ModuleBEvaluator",
    "ModuleBTrainer",
    "ModuleBRegistry",
    "module_b_registry",
]
