"""
AstraGuard 2.4 — Digital IC Feature Engineer
=============================================
Domain-specific feature contract for CMOS Digital IC IDDQ degradation & aging.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from astraguard_core.feature_engineering.base import BaseFeatureEngineer
from astraguard_core.feature_engineering.common import (
    calc_growth_rate,
    calc_ratio,
    calc_arrhenius_temp_normalized,
    calc_robust_population_stats,
    calc_robust_z_score,
)


class DigitalICFeatureEngineer(BaseFeatureEngineer):
    """Engineered feature contract for DIGITAL_IC devices (IDDQ primary parameter)."""

    @property
    def device_family(self) -> str:
        return "DIGITAL_IC"

    @property
    def primary_parameter(self) -> str:
        return "IDDQ"

    @property
    def target_name(self) -> str:
        return "value_168h_actual"

    @property
    def feature_names(self) -> List[str]:
        return [
            "value_0h",
            "value_24h",
            "delta_value_0_24h",
            "growth_rate_per_hour",
            "degradation_ratio_24h_0h",
            "temp_normalized_0h",
            "temp_normalized_24h",
            "robust_z_score_0h",
            "robust_z_score_24h",
            "ICC_active_mA",
            "input_leakage_nA",
            "propagation_delay_ns",
            "v_offset_24h",
            "snr_24h",
        ]

    def extract_features(
        self,
        df: pd.DataFrame,
        context_resolution: Optional[Dict[str, Any]] = None,
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        v0 = df["value_0h"] if "value_0h" in df.columns else df["iddq_0h"]
        v24 = df["value_24h"] if "value_24h" in df.columns else df["iddq_24h"]

        stress_temp = df["stress_temperature_c"].iloc[0] if "stress_temperature_c" in df.columns else 125.0

        # Primary features
        delta_v = v24 - v0
        growth_rate = calc_growth_rate(v0, v24, 24.0)
        ratio = calc_ratio(v0, v24)

        # Thermal normalization (Ea = 0.68 eV [PE])
        t_norm_0 = calc_arrhenius_temp_normalized(v0, stress_temp, t_ref_c=25.0, ea_eV=0.68)
        t_norm_24 = calc_arrhenius_temp_normalized(v24, stress_temp, t_ref_c=25.0, ea_eV=0.68)

        # Leakage-safe population stats
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

        # Secondary measurement columns
        icc = df["ICC_active_mA"] if "ICC_active_mA" in df.columns else pd.Series(25.0, index=df.index)
        leakage = df["input_leakage_nA"] if "input_leakage_nA" in df.columns else pd.Series(2.0, index=df.index)
        delay = df["propagation_delay_ns"] if "propagation_delay_ns" in df.columns else pd.Series(12.5, index=df.index)
        v_off_24 = df["v_offset_24h"] if "v_offset_24h" in df.columns else pd.Series(0.0, index=df.index)
        snr_24 = df["snr_24h"] if "snr_24h" in df.columns else pd.Series(35.0, index=df.index)

        res = pd.DataFrame({
            "value_0h": v0,
            "value_24h": v24,
            "delta_value_0_24h": delta_v,
            "growth_rate_per_hour": growth_rate,
            "degradation_ratio_24h_0h": ratio,
            "temp_normalized_0h": t_norm_0,
            "temp_normalized_24h": t_norm_24,
            "robust_z_score_0h": z0,
            "robust_z_score_24h": z24,
            "ICC_active_mA": icc,
            "input_leakage_nA": leakage,
            "propagation_delay_ns": delay,
            "v_offset_24h": v_off_24,
            "snr_24h": snr_24,
        }, index=df.index)

        return res[self.feature_names]
