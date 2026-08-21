"""
AstraGuard 2.4 — Feature Engineering v2: MEMS Gyroscope (0h + 24h + 96h)
==========================================================================
Extends v1 contract with five 96h trajectory features.

96h features justified by separability experiment:
  - MEMS_STICTION_ONSET : vel_24_96 Cohen d = 61.8  (currently 100% escape)
  - THERMAL_RUNAWAY     : vel_24_96 Cohen d = 53.1  (currently 95% escape)
  - SPATIAL_OUTLIER     : z_96      Cohen d = 23.8
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
    calc_velocity_24_96,
    calc_trajectory_acceleration,
    calc_growth_ratio_96h,
)


class MEMSGyroFeatureEngineerV2(BaseFeatureEngineer):
    """V2 feature contract for MEMS_GYROSCOPE."""

    @property
    def version(self) -> str:
        return "v2"

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
            # --- v1 features ---
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
            # --- v2 96h trajectory features ---
            "value_96h",
            "velocity_24_96",
            "trajectory_acceleration",
            "growth_ratio_96h",
            "robust_z_score_96h",
        ]

    def extract_features(
        self,
        df: pd.DataFrame,
        context_resolution: Optional[Dict[str, Any]] = None,
        population_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        v0  = df["value_0h"]
        v24 = df["value_24h"]
        v96 = df["value_96h"]

        # --- v1 features ---
        delta_v            = v24 - v0
        drift_velocity     = calc_growth_rate(v0, v24, 24.0)
        log_creep_rate     = delta_v / np.log(1.0 + 24.0 / 5.0)
        viscoelastic_index = calc_ratio(v0, v24, eps=1e-5)

        if population_stats and "value_0h"  in population_stats:
            med0,  mad0  = population_stats["value_0h"]["median"],  population_stats["value_0h"]["mad"]
        else:
            med0, mad0 = calc_robust_population_stats(v0)

        if population_stats and "value_24h" in population_stats:
            med24, mad24 = population_stats["value_24h"]["median"], population_stats["value_24h"]["mad"]
        else:
            med24, mad24 = calc_robust_population_stats(v24)

        z0  = calc_robust_z_score(v0,  med0,  mad0)
        z24 = calc_robust_z_score(v24, med24, mad24)

        sf_err    = df["scale_factor_error_ppm"]       if "scale_factor_error_ppm"       in df.columns else pd.Series(150.0,  index=df.index)
        noise_den = df["noise_density_dps_rtHz"]       if "noise_density_dps_rtHz"       in df.columns else pd.Series(0.004,  index=df.index)
        icc       = df["supply_current_mA"]            if "supply_current_mA"            in df.columns else pd.Series(5.5,    index=df.index)
        temp_sens = df["temperature_sensitivity_dps_C"] if "temperature_sensitivity_dps_C" in df.columns else pd.Series(0.008,  index=df.index)

        # --- v2: 96h trajectory features ---
        vel_0_24  = calc_growth_rate(v0, v24, 24.0)
        vel_24_96 = calc_velocity_24_96(v24, v96, 72.0)
        accel     = calc_trajectory_acceleration(vel_0_24, vel_24_96)
        g_ratio   = calc_growth_ratio_96h(vel_0_24, vel_24_96)

        if population_stats and "value_96h" in population_stats:
            med96, mad96 = population_stats["value_96h"]["median"], population_stats["value_96h"]["mad"]
        else:
            med96, mad96 = calc_robust_population_stats(v96)
        z96 = calc_robust_z_score(v96, med96, mad96)

        res = pd.DataFrame({
            "value_0h": v0, "value_24h": v24,
            "delta_value_0_24h": delta_v, "drift_velocity_per_hour": drift_velocity,
            "logarithmic_creep_rate": log_creep_rate,
            "viscoelastic_relaxation_index": viscoelastic_index,
            "robust_z_score_0h": z0, "robust_z_score_24h": z24,
            "scale_factor_error_ppm": sf_err, "noise_density_dps_rtHz": noise_den,
            "supply_current_mA": icc, "temperature_sensitivity_dps_C": temp_sens,
            # v2 additions
            "value_96h": v96, "velocity_24_96": vel_24_96,
            "trajectory_acceleration": accel, "growth_ratio_96h": g_ratio,
            "robust_z_score_96h": z96,
        }, index=df.index)

        return res[self.feature_names]
