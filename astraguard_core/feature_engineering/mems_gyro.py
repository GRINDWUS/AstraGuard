"""
AstraGuard 2.4 — MEMS Gyroscope Feature Engineer
=================================================
Domain-specific feature contract for MEMS zero-rate-offset (ZRO) viscoelastic relaxation & creep.
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


class MEMSGyroFeatureEngineer(BaseFeatureEngineer):
    """Engineered feature contract for MEMS_GYROSCOPE devices."""

    @property
    def device_family(self) -> str:
        return "MEMS_GYROSCOPE"

    @property
    def primary_parameter(self) -> str:
        return "zero_rate_offset"

    @property
    def target_name(self) -> str:
        return "value_168h_actual"

    @property
    def feature_names(self) -> List[str]:
        return [
            "value_0h",
            "value_24h",
            "delta_value_0_24h",
            "drift_velocity_per_hour",
            "logarithmic_creep_rate",
            "viscoelastic_relaxation_index",
            "robust_z_score_0h",
            "robust_z_score_24h",
            "scale_factor_error_ppm",
            "noise_density_dps_rtHz",
            "supply_current_mA",
            "temperature_sensitivity_dps_C",
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
        drift_velocity = calc_growth_rate(v0, v24, 24.0)
        
        # Logarithmic creep aging velocity B = delta_ZRO / ln(1 + 24/5.0) [PE]
        log_creep_rate = delta_v / np.log(1.0 + 24.0 / 5.0)
        viscoelastic_index = calc_ratio(v0, v24, eps=1e-5)

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

        # Secondary parameters
        sf_err = df["scale_factor_error_ppm"] if "scale_factor_error_ppm" in df.columns else pd.Series(150.0, index=df.index)
        noise_den = df["noise_density_dps_rtHz"] if "noise_density_dps_rtHz" in df.columns else pd.Series(0.004, index=df.index)
        icc = df["supply_current_mA"] if "supply_current_mA" in df.columns else pd.Series(5.5, index=df.index)
        temp_sens = df["temperature_sensitivity_dps_C"] if "temperature_sensitivity_dps_C" in df.columns else pd.Series(0.008, index=df.index)

        res = pd.DataFrame({
            "value_0h": v0,
            "value_24h": v24,
            "delta_value_0_24h": delta_v,
            "drift_velocity_per_hour": drift_velocity,
            "logarithmic_creep_rate": log_creep_rate,
            "viscoelastic_relaxation_index": viscoelastic_index,
            "robust_z_score_0h": z0,
            "robust_z_score_24h": z24,
            "scale_factor_error_ppm": sf_err,
            "noise_density_dps_rtHz": noise_den,
            "supply_current_mA": icc,
            "temperature_sensitivity_dps_C": temp_sens,
        }, index=df.index)

        return res[self.feature_names]
