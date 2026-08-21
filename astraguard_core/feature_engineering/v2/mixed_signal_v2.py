"""
AstraGuard 2.4 — Feature Engineering v2: Mixed Signal IC (0h + 24h + 96h)
==========================================================================
Extends v1 contract with five 96h trajectory features.

96h features justified by separability experiment:
  - THERMAL_RUNAWAY   : vel_24_96 Cohen d = 264.3
  - ELECTROMIGRATION  : growth_ratio Cohen d = 62.6
  - SPATIAL_OUTLIER   : accel Cohen d = 23.8
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
    calc_velocity_24_96,
    calc_trajectory_acceleration,
    calc_growth_ratio_96h,
)


class MixedSignalFeatureEngineerV2(BaseFeatureEngineer):
    """V2 feature contract for MIXED_SIGNAL_IC."""

    @property
    def version(self) -> str:
        return "v2"

    @property
    def device_family(self) -> str:
        return "MIXED_SIGNAL_IC"

    @property
    def primary_parameter(self) -> str:
        return "IDDQ"

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
            "growth_rate_per_hour",
            "degradation_ratio_24h_0h",
            "temp_normalized_24h",
            "robust_z_score_0h",
            "robust_z_score_24h",
            "analog_digital_interaction_ratio",
            "v_offset_24h",
            "snr_24h",
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
        v0  = df["value_0h"]  if "value_0h"  in df.columns else df["iddq_0h"]
        v24 = df["value_24h"] if "value_24h" in df.columns else df["iddq_24h"]
        v96 = df["value_96h"]

        stress_temp = df["stress_temperature_c"].iloc[0] if "stress_temperature_c" in df.columns else 125.0

        # --- v1 features ---
        delta_v    = v24 - v0
        growth_rate = calc_growth_rate(v0, v24, 24.0)
        ratio       = calc_ratio(v0, v24)
        t_norm_24   = calc_arrhenius_temp_normalized(v24, stress_temp, t_ref_c=25.0, ea_eV=0.68)

        if population_stats and "value_0h" in population_stats:
            med0,  mad0  = population_stats["value_0h"]["median"],  population_stats["value_0h"]["mad"]
        else:
            med0, mad0 = calc_robust_population_stats(v0)

        if population_stats and "value_24h" in population_stats:
            med24, mad24 = population_stats["value_24h"]["median"], population_stats["value_24h"]["mad"]
        else:
            med24, mad24 = calc_robust_population_stats(v24)

        z0  = calc_robust_z_score(v0,  med0,  mad0)
        z24 = calc_robust_z_score(v24, med24, mad24)

        v_off = df["v_offset_24h"] if "v_offset_24h" in df.columns else pd.Series(0.0,  index=df.index)
        snr   = df["snr_24h"]      if "snr_24h"      in df.columns else pd.Series(35.0, index=df.index)
        ad_interaction = v24 / np.maximum(1e-3, np.abs(v_off) + 1.0)

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
            "delta_value_0_24h": delta_v, "growth_rate_per_hour": growth_rate,
            "degradation_ratio_24h_0h": ratio, "temp_normalized_24h": t_norm_24,
            "robust_z_score_0h": z0, "robust_z_score_24h": z24,
            "analog_digital_interaction_ratio": ad_interaction,
            "v_offset_24h": v_off, "snr_24h": snr,
            # v2 additions
            "value_96h": v96, "velocity_24_96": vel_24_96,
            "trajectory_acceleration": accel, "growth_ratio_96h": g_ratio,
            "robust_z_score_96h": z96,
        }, index=df.index)

        return res[self.feature_names]
