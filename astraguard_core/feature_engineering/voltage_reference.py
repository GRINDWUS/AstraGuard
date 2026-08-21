"""
AstraGuard 2.4 — Precision Voltage Reference Feature Engineer
==============================================================
Domain-specific feature contract for Precision Voltage Reference output drift.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from astraguard_core.feature_engineering.base import BaseFeatureEngineer
from astraguard_core.feature_engineering.common import (
    calc_growth_rate,
    calc_ratio,
    calc_robust_population_stats,
    calc_robust_z_score,
)


class VoltageReferenceFeatureEngineer(BaseFeatureEngineer):
    """Engineered feature contract for PRECISION_VOLTAGE_REF devices."""

    @property
    def device_family(self) -> str:
        return "PRECISION_VOLTAGE_REF"

    @property
    def primary_parameter(self) -> str:
        return "output_voltage_drift"

    @property
    def target_name(self) -> str:
        return "value_168h_actual"

    @property
    def feature_names(self) -> List[str]:
        return [
            "value_0h",
            "value_24h",
            "delta_value_0_24h",
            "drift_rate_per_hour",
            "ppm_drift_24h",
            "robust_z_score_0h",
            "robust_z_score_24h",
            "v_offset_24h",
            "snr_24h",
        ]

    def extract_features(
        self,
        df: pd.DataFrame,
        context_resolution: Optional[Dict[str, Any]] = None,
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        v0 = df["value_0h"]
        v24 = df["value_24h"]

        delta_v = v24 - v0
        drift_rate = calc_growth_rate(v0, v24, 24.0)

        # Parts-per-million (PPM) drift relative to 2.5V reference nominal [PE]
        v_nom_uv = 2.5e6  # 2.5V in uV
        ppm_drift = (v24 / v_nom_uv) * 1e6

        if population_stats and "value_0h" in population_stats:
            med0, mad0 = population_stats["value_0h"]["median"], population_stats["value_0h"]["mad"]
        else:
            med0, mad0 = calc_robust_population_stats(v0)

        if population_stats and "value_24h" in population_stats:
            med24, mad24 = population_stats["value_24h"]["median"], population_stats["value_24h"]["mad"]
        else:
            med24, mad24 = calc_robust_population_stats(v24)

        z0 = calc_robust_z_score(v0, med0, mad0)
        z24 = calc_robust_z_score(v24, med24, mad24)

        v_off_24 = df["v_offset_24h"] if "v_offset_24h" in df.columns else pd.Series(0.0, index=df.index)
        snr_24 = df["snr_24h"] if "snr_24h" in df.columns else pd.Series(35.0, index=df.index)

        res = pd.DataFrame({
            "value_0h": v0,
            "value_24h": v24,
            "delta_value_0_24h": delta_v,
            "drift_rate_per_hour": drift_rate,
            "ppm_drift_24h": ppm_drift,
            "robust_z_score_0h": z0,
            "robust_z_score_24h": z24,
            "v_offset_24h": v_off_24,
            "snr_24h": snr_24,
        }, index=df.index)

        return res[self.feature_names]
