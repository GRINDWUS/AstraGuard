"""
AstraGuard 2.4 — Image Sensor Feature Engineer
================================================
Domain-specific feature contract for CMOS Image Sensor dark current density & hot pixel growth.
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


class ImageSensorFeatureEngineer(BaseFeatureEngineer):
    """Engineered feature contract for IMAGE_SENSOR devices."""

    @property
    def device_family(self) -> str:
        return "IMAGE_SENSOR"

    @property
    def primary_parameter(self) -> str:
        return "dark_current_density"

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
            "trap_generation_rate",
            "srh_temp_normalized_24h",
            "robust_z_score_0h",
            "robust_z_score_24h",
            "DSNU_DN",
            "hot_pixel_count",
            "supply_current_mA",
            "dark_signal_mean_DN",
        ]

    def extract_features(
        self,
        df: pd.DataFrame,
        context_resolution: Optional[Dict[str, Any]] = None,
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        v0 = df["value_0h"]
        v24 = df["value_24h"]

        stress_temp = df["stress_temperature_c"].iloc[0] if "stress_temperature_c" in df.columns else 60.0

        delta_v = v24 - v0
        growth_rate = calc_growth_rate(v0, v24, 24.0)

        # SRH trap generation rate indicator [PE]
        trap_gen_rate = (v24 / np.maximum(1e-5, v0) - 1.0) / 24.0

        # Thermal normalization relative to 25°C (Ea = 0.55 eV [PE])
        srh_norm_24 = calc_arrhenius_temp_normalized(v24, stress_temp, t_ref_c=25.0, ea_eV=0.55)

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

        dsnu = df["DSNU_DN"] if "DSNU_DN" in df.columns else pd.Series(12.0, index=df.index)
        hot_pixels = df["hot_pixel_count"] if "hot_pixel_count" in df.columns else pd.Series(5, index=df.index)
        icc = df["supply_current_mA"] if "supply_current_mA" in df.columns else pd.Series(42.0, index=df.index)
        dark_mean = df["dark_signal_mean_DN"] if "dark_signal_mean_DN" in df.columns else pd.Series(8.0, index=df.index)

        res = pd.DataFrame({
            "value_0h": v0,
            "value_24h": v24,
            "delta_value_0_24h": delta_v,
            "growth_rate_per_hour": growth_rate,
            "trap_generation_rate": trap_gen_rate,
            "srh_temp_normalized_24h": srh_norm_24,
            "robust_z_score_0h": z0,
            "robust_z_score_24h": z24,
            "DSNU_DN": dsnu,
            "hot_pixel_count": hot_pixels,
            "supply_current_mA": icc,
            "dark_signal_mean_DN": dark_mean,
        }, index=df.index)

        return res[self.feature_names]
