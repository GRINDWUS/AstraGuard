"""
AstraGuard 2.4 — Physics Attribution Mapper
=============================================
Maps mathematical SHAP feature attributions back to physical failure mechanisms
and semiconductor degradation principles.
"""

from typing import Dict, Any, List


_PHYSICS_KNOWLEDGE_BASE = {
    "DIGITAL_IC": {
        "robust_z_score_24h": "Arrhenius thermal leakage baseline drift (Ea = 0.68 eV)",
        "degradation_ratio_24h_0h": "Early-life kinetic degradation acceleration (24h/0h ratio)",
        "growth_rate_per_hour": "Continuous standby current (IDDQ) aging slope",
        "temp_normalized_24h": "Temperature-compensated leakage current at stress temperature",
        "temp_normalized_0h": "Baseline initial process corner leakage offset",
        "propagation_delay_ns": "Gate oxide degradation and carrier mobility reduction",
        "input_leakage_nA": "ESD protection diode / gate oxide leakage",
        "snr_24h": "Thermal noise to signal ratio at 24h burn-in",
    },
    "MIXED_SIGNAL_IC": {
        "robust_z_score_24h": "Arrhenius thermal leakage baseline drift (Ea = 0.68 eV)",
        "degradation_ratio_24h_0h": "Early-life kinetic degradation acceleration",
        "v_offset_24h": "Analog differential pair offset voltage thermal drift",
        "v_offset_0h": "Initial differential pair mismatch offset",
        "snr_24h": "Analog signal-to-noise ratio degradation under burn-in",
    },
    "MEMS_GYROSCOPE": {
        "robust_z_score_24h": "Viscoelastic die-attach package stress relaxation outlier",
        "viscoelastic_relaxation_index": "Exponential package stress relaxation drift [PE]",
        "logarithmic_creep_rate": "Logarithmic material creep rate in micro-machined beam",
        "delta_value_0_24h": "Early zero-rate offset (ZRO) shift during thermal stress",
        "scale_factor_error_ppm": "Drive loop mechanical resonant frequency detuning",
    },
    "IMAGE_SENSOR": {
        "robust_z_score_24h": "Shockley-Read-Hall (SRH) depletion region dark current outlier",
        "srh_temp_normalized_24h": "Normalized thermal electron-hole pair generation rate",
        "growth_rate_per_hour": "Silicon bulk trap generation rate under thermal stress",
        "hot_pixel_count": "Local micro-defect trap clustering density",
        "DSNU_DN": "Dark signal non-uniformity spatial variance",
    },
    "PRECISION_VOLTAGE_REF": {
        "robust_z_score_24h": "Bandgap reference zener/MOSFET thermal drift outlier",
        "degradation_ratio_24h_0h": "Early-life reference voltage shift ratio",
        "growth_rate_per_hour": "Continuous bandgap reference voltage drift rate",
        "temp_normalized_24h": "Temperature-compensated output voltage drift",
    }
}


class PhysicsAttributionMapper:
    """Translates mathematical SHAP values into physical QA justifications."""

    @staticmethod
    def map_attributions(
        device_family: str,
        shap_attributions: Dict[str, float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Map top K SHAP feature contributions to physical failure mechanisms.
        """
        family_clean = str(device_family).upper().strip()
        physics_map = _PHYSICS_KNOWLEDGE_BASE.get(family_clean, _PHYSICS_KNOWLEDGE_BASE["DIGITAL_IC"])

        mapped_explanations = []
        for feat_name, shap_val in list(shap_attributions.items())[:top_k]:
            physical_mechanism = physics_map.get(
                feat_name, f"Parametric variation in feature '{feat_name}'"
            )
            direction = "increased" if shap_val > 0 else "decreased"
            
            explanation_text = (
                f"Feature '{feat_name}' ({direction} predicted 168h drift by {abs(shap_val):.4f}): "
                f"{physical_mechanism}."
            )

            mapped_explanations.append({
                "feature_name": feat_name,
                "shap_value": shap_val,
                "impact_direction": direction,
                "physical_mechanism": physical_mechanism,
                "qa_summary": explanation_text
            })

        return mapped_explanations
