"""
AstraGuard 2.4 — 3-Tier Decision Fusion Engine
===============================================
Merges Module A statistical outlier scores, Module B 168h forecasts, and test context
confidence into an automated 3-tier aerospace screening decision:
  - GREEN_PASS
  - YELLOW_ACCELERATE_CHAMBER
  - RED_EARLY_REJECT
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class DecisionFusionEngine:
    """Combines Module A + Module B + Context into a 3-tier risk decision."""

    SPEC_LIMITS = {
        "DIGITAL_IC": 1150.0,            # uA
        "MIXED_SIGNAL_IC": 1150.0,       # uA
        "MEMS_GYROSCOPE": 25.0,          # dps
        "IMAGE_SENSOR": 25.0,            # nA/cm2
        "PRECISION_VOLTAGE_REF": 6800.0, # uV
    }

    def evaluate_component(
        self,
        component_id: str,
        device_family: str,
        robust_z_24h: float,
        predicted_168h: float,
        context_confidence: float = 1.0,
        custom_spec_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Synthesize multi-module evidence into a single decision.
        """
        spec_lim = custom_spec_limit or self.SPEC_LIMITS.get(device_family.upper(), 1000.0)
        safety_threshold = spec_lim * 0.90  # 10% safety margin

        risk_tier = "GREEN_PASS"
        action = "PASS — Continue qualification"
        justifications = []

        # 1. Module A Outlier Check
        abs_z = abs(robust_z_24h)
        if abs_z > 3.5:
            risk_tier = "RED_EARLY_REJECT"
            action = "EARLY REJECT — Severe 24h population anomaly detected (Module A)"
            justifications.append(f"Module A: 24h Robust Z-score ({robust_z_24h:+.2f}) exceeds threshold (|Z| > 3.5).")
        elif abs_z > 2.5:
            if risk_tier != "RED_EARLY_REJECT":
                risk_tier = "YELLOW_ACCELERATE_CHAMBER"
                action = "ACCELERATE ESS — Moderate 24h parametric shift detected"
            justifications.append(f"Module A: Moderate Z-score anomaly ({robust_z_24h:+.2f}).")

        # 2. Module B 168h Forecast Check
        if predicted_168h >= spec_lim:
            risk_tier = "RED_EARLY_REJECT"
            action = "EARLY REJECT — Predicted 168h value exceeds upper spec limit"
            justifications.append(f"Module B: Predicted 168h value ({predicted_168h:.2f}) breaches USL ({spec_lim:.2f}).")
        elif predicted_168h >= safety_threshold:
            if risk_tier != "RED_EARLY_REJECT":
                risk_tier = "YELLOW_ACCELERATE_CHAMBER"
                action = "ACCELERATE ESS — Forecasted trajectory near upper spec limit"
            justifications.append(f"Module B: Predicted 168h value ({predicted_168h:.2f}) within 10% safety margin of USL.")

        # 3. Context & Metadata Integrity Check
        if context_confidence < 0.70:
            if risk_tier != "RED_EARLY_REJECT":
                risk_tier = "YELLOW_ACCELERATE_CHAMBER"
                action = "MANUAL QA REVIEW — Low test identity confidence score"
            justifications.append(f"Context: Low metadata confidence ({context_confidence:.2f}).")

        if risk_tier == "GREEN_PASS":
            justifications.append("All parametric metrics nominal and forecasted 168h trajectory safe.")

        return {
            "component_id": component_id,
            "device_family": device_family,
            "risk_tier": risk_tier,
            "recommended_action": action,
            "predicted_168h_value": float(predicted_168h),
            "spec_limit": float(spec_lim),
            "robust_z_24h": float(robust_z_24h),
            "context_confidence": float(context_confidence),
            "decision_justifications": justifications
        }
