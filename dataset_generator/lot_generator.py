"""
AstraGuard 2.4 — ASQD Context-Aware Multi-Device Lot Simulator
================================================================
Generates qualification lots for 5 device families using device-specific
physics-based drift models:
  - DIGITAL_IC:             ArrheniusIDDQModel      (Ea=0.68 eV [PE])
  - MIXED_SIGNAL_IC:        ArrheniusIDDQModel      (Ea=0.68 eV [PE])
  - MEMS_GYROSCOPE:         ViscoelasticMEMSModel   (ZRO drift [PE])
  - IMAGE_SENSOR:           SRHDarkCurrentModel     (Dark current [PE])
  - PRECISION_VOLTAGE_REF:  ArrheniusIDDQModel      (VOS drift [PE])

Each lot carries the canonical ASQD 2.4 three-dimension schema:
  Component × Test Context × Time-Series Measurement

Backward compatibility:
  All output DataFrames include 'iddq_0h', 'iddq_24h', 'iddq_96h_actual',
  'iddq_168h_actual', 'delta_iddq' columns (aliased from primary parameter)
  so legacy PS benchmark pipeline runs without modification.

Synthetic parameters labelled [SA]; physics models labelled [PE].
"""

import numpy as np
import pandas as pd
from dataset_generator.iddq_model import (
    ArrheniusIDDQModel,
    ViscoelasticMEMSModel,
    SRHDarkCurrentModel,
)


# ---------------------------------------------------------------------------
# Device family → physics model constructor
# ---------------------------------------------------------------------------
_DEVICE_MODEL_MAP = {
    "DIGITAL_IC":          lambda: ArrheniusIDDQModel(base_iddq_uA=1.2, ea_eV=0.68),
    "MIXED_SIGNAL_IC":     lambda: ArrheniusIDDQModel(base_iddq_uA=2.0, ea_eV=0.68),
    "MEMS_GYROSCOPE":      lambda: ViscoelasticMEMSModel(base_zro_dps=0.05),
    "IMAGE_SENSOR":        lambda: SRHDarkCurrentModel(base_dark_current_nA_cm2=1.5, stress_temp_c=60.0),
    "PRECISION_VOLTAGE_REF": lambda: ArrheniusIDDQModel(base_iddq_uA=15.0, ea_eV=0.62),
}

# Device family → primary parameter config
_DEVICE_PARAM_CONFIG = {
    "DIGITAL_IC": {
        "primary_parameter": "IDDQ",
        "unit": "uA",
        "category": "ELECTRICAL",
        "spec_max": 50.0,
        "default_test_type": "THERMAL_BURN_IN",
        "stress_temperature_c": 125.0,
        "stress_voltage_v": 5.0,
    },
    "MIXED_SIGNAL_IC": {
        "primary_parameter": "IDDQ",
        "unit": "uA",
        "category": "ELECTRICAL",
        "spec_max": 75.0,
        "default_test_type": "THERMAL_BURN_IN",
        "stress_temperature_c": 125.0,
        "stress_voltage_v": 3.3,
    },
    "MEMS_GYROSCOPE": {
        "primary_parameter": "zero_rate_offset",
        "unit": "dps",
        "category": "MECHANICAL",
        "spec_max": 0.5,
        "default_test_type": "MEMS_THERMAL_STRESS",
        "stress_temperature_c": 85.0,
        "stress_voltage_v": 3.3,
    },
    "IMAGE_SENSOR": {
        "primary_parameter": "dark_current_density",
        "unit": "nA/cm2",
        "category": "OPTICAL",
        "spec_max": 10.0,
        "default_test_type": "IMAGE_SENSOR_DARK_CURRENT_TEST",
        "stress_temperature_c": 60.0,
        "stress_voltage_v": 3.3,
    },
    "PRECISION_VOLTAGE_REF": {
        "primary_parameter": "output_voltage_drift",
        "unit": "uV",
        "category": "ELECTRICAL",
        "spec_max": 100.0,
        "default_test_type": "THERMAL_BURN_IN",
        "stress_temperature_c": 125.0,
        "stress_voltage_v": 5.0,
    },
}

# Failure profile names per device family
_FAILURE_PROFILES = {
    "DIGITAL_IC": [
        ("NOMINAL", 0.965),
        ("THERMAL_RUNAWAY", 0.01),
        ("ELECTROMIGRATION", 0.01),
        ("SPATIAL_OUTLIER", 0.005),
        ("DIELECTRIC_OSCILLATION", 0.01),
    ],
    "MIXED_SIGNAL_IC": [
        ("NOMINAL", 0.965),
        ("THERMAL_RUNAWAY", 0.01),
        ("ELECTROMIGRATION", 0.01),
        ("SPATIAL_OUTLIER", 0.005),
        ("DIELECTRIC_OSCILLATION", 0.01),
    ],
    "MEMS_GYROSCOPE": [
        ("NOMINAL", 0.955),
        ("MEMS_STICTION_ONSET", 0.015),
        ("PACKAGING_STRESS_RELAXATION", 0.015),
        ("SPATIAL_OUTLIER", 0.005),
        ("THERMAL_RUNAWAY", 0.01),
    ],
    "IMAGE_SENSOR": [
        ("NOMINAL", 0.960),
        ("DARK_CURRENT_SPIKE_GROWTH", 0.015),
        ("THERMAL_RUNAWAY", 0.01),
        ("SPATIAL_OUTLIER", 0.005),
        ("DIELECTRIC_OSCILLATION", 0.01),
    ],
    "PRECISION_VOLTAGE_REF": [
        ("NOMINAL", 0.965),
        ("THERMAL_RUNAWAY", 0.01),
        ("ELECTROMIGRATION", 0.01),
        ("SPATIAL_OUTLIER", 0.005),
        ("DIELECTRIC_OSCILLATION", 0.01),
    ],
}


class LotSimulator:
    """
    ASQD 2.4 multi-device qualification lot generator.
    Uses device-specific physics models for all primary parameter trajectories.
    Backward-compat aliases ensure legacy PS benchmark pipeline runs unchanged.
    """

    def __init__(self, base_iddq: float = 1.2, ea_eV: float = 0.68):
        self._legacy_base_iddq = base_iddq
        self._legacy_ea_eV = ea_eV

    def _assign_failure_profiles(self, num_components: int, device_family: str) -> list:
        profiles_spec = _FAILURE_PROFILES.get(
            device_family, _FAILURE_PROFILES["DIGITAL_IC"]
        )
        names = [p[0] for p in profiles_spec]
        weights = [p[1] for p in profiles_spec]
        total = sum(weights)
        weights = [w / total for w in weights]
        return list(np.random.choice(names, size=num_components, p=weights))

    def _get_physics_model(self, device_family: str):
        factory = _DEVICE_MODEL_MAP.get(device_family, _DEVICE_MODEL_MAP["DIGITAL_IC"])
        model = factory()
        return model

    def generate_lot(
        self,
        lot_id: int = 0,
        num_components: int = 2000,
        num_hours: int = 168,
        device_family: str = "DIGITAL_IC",
        test_type: str = None,
        corrupt_metadata: bool = False,
        strip_metadata: bool = False,
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Generate an ASQD 2.4 lot DataFrame.
        Includes controlled process variation, lot baseline scaling, and secondary parameter noise.
        """
        if seed is not None:
            np.random.seed(seed)

        cfg = _DEVICE_PARAM_CONFIG.get(device_family, _DEVICE_PARAM_CONFIG["DIGITAL_IC"])
        effective_test_type = test_type or cfg["default_test_type"]

        profiles = self._assign_failure_profiles(num_components, device_family)
        physics_model = self._get_physics_model(device_family)

        # Lot-level baseline variation [SA]
        lot_base_scale = float(np.clip(np.random.normal(1.0, 0.08), 0.85, 1.20))
        records = []

        for comp_idx in range(num_components):
            profile = profiles[comp_idx]
            comp_id = f"{device_family[:3]}_LOT{lot_id:02d}_COMP{comp_idx:04d}"

            # Component process corner variation [SA]
            comp_base_scale = float(np.clip(np.random.normal(1.0, 0.12), 0.70, 1.35))

            # === Physics trajectory (device-specific) ===
            traj = physics_model.generate_trajectory(
                num_hours=num_hours,
                profile=profile,
                comp_base_scale=comp_base_scale,
                lot_base_scale=lot_base_scale
            )

            v0   = traj[0]
            v24  = traj[24]  if len(traj) > 24  else traj[-1]
            v96  = traj[96]  if len(traj) > 96  else traj[-1]
            v168 = traj[168] if len(traj) > 168 else traj[-1]

            # === Metadata fields for context resolution ===
            assigned_family   = None if strip_metadata else device_family
            assigned_test     = None if strip_metadata else effective_test_type
            assigned_param    = cfg["primary_parameter"]
            assigned_unit     = cfg["unit"]
            conf_score        = 1.0

            if corrupt_metadata:
                assigned_family = "CORRUPTED_FAMILY_HEADER"
                assigned_test   = "CORRUPTED_TEST_TYPE"
                assigned_param  = "CORRUPTED_PARAM_NOISE"
                assigned_unit   = "CORRUPTED_UNIT"
                conf_score      = 0.10

            # === Secondary parameters (device-specific, [SA] distributions) ===
            secondary = self._generate_secondary_params(device_family, v0, v24, profile)

            records.append({
                # --- ASQD Layer 1: Component Metadata ---
                "component_id":        comp_id,
                "lot_id":              f"LOT_{lot_id:02d}",
                "device_family":       assigned_family,
                "package_type":        "CQFP-64" if device_family in ("DIGITAL_IC", "MIXED_SIGNAL_IC") else "LCC-20",
                "part_number":         f"{device_family[:3]}-SIH-2026",

                # --- ASQD Layer 2: Test Metadata ---
                "test_type":           assigned_test,
                "test_id":             f"T{lot_id:02d}_{comp_idx:04d}",
                "procedure_id":        "PROC-ISRO-IISU-01",
                "station_id":          "ATE-STATION-01",
                "chamber_id":          "CHAMBER-01",
                "test_identity_confidence": conf_score,

                # --- ASQD Layer 3: Stress Conditions ---
                "stress_temperature_c": cfg["stress_temperature_c"],
                "stress_voltage_v":     cfg["stress_voltage_v"],
                "duration_hours":        num_hours,

                # --- ASQD Layer 4a: Primary Measurement ---
                "primary_parameter":   assigned_param,
                "parameter_category":  cfg["category"],
                "unit":                assigned_unit,
                "value_0h":            v0,
                "value_24h":           v24,
                "value_96h":           v96,
                "value_168h_actual":   v168,
                "delta_value_0_24h":   round(v24 - v0, 6),

                # --- ASQD Layer 4b: Secondary Parameters ([SA]) ---
                **secondary,

                # --- Legacy Aliases (PS Backward Compat — DO NOT REMOVE) ---
                "iddq_0h":             v0,
                "iddq_24h":            v24,
                "iddq_96h_actual":     v96,
                "iddq_168h_actual":    v168,
                "delta_iddq":          round(v24 - v0, 6),
                "spec_max_iddq":       cfg["spec_max"],

                # --- ASQD Layer 5: Ground Truth ---
                "is_defective_gt":     profile != "NOMINAL",
                "failure_mode_gt":     profile,
                "instrument_status":   "HEALTHY",

                # --- ASQD Layer 6: Spatial Coordinates ---
                "wafer_x":             round(float(np.random.uniform(-50, 50)), 2),
                "wafer_y":             round(float(np.random.uniform(-50, 50)), 2),
            })

        return pd.DataFrame(records)

    def _generate_secondary_params(
        self, device_family: str, v0: float, v24: float, profile: str
    ) -> dict:
        """Generate device-specific secondary measurement columns with synthetic variation."""
        v_offset_0 = round(float(np.random.normal(0, 0.02)), 4)
        v_offset_24 = round(float(v_offset_0 + np.random.normal(0, 0.05)), 4)
        snr_0 = round(float(np.random.normal(35.0, 1.2)), 2)
        snr_24 = round(float(snr_0 + np.random.normal(0, 0.5)), 2)

        if device_family == "DIGITAL_IC":
            return {
                "ICC_active_mA":       round(float(np.random.normal(25.0, 1.5)), 3),
                "input_leakage_nA":    round(float(np.random.normal(2.0, 0.3)), 4),
                "propagation_delay_ns": round(float(np.random.normal(12.5, 0.8)), 3),
                "v_offset_0h":         v_offset_0,
                "v_offset_24h":        v_offset_24,
                "snr_0h":              snr_0,
                "snr_24h":             snr_24,
            }
        elif device_family == "MEMS_GYROSCOPE":
            return {
                "scale_factor_error_ppm": round(float(np.random.normal(150.0, 25.0)), 2),
                "noise_density_dps_rtHz": round(float(np.random.uniform(0.003, 0.006)), 5),
                "supply_current_mA":      round(float(np.random.normal(5.5, 0.2)), 3),
                "temperature_sensitivity_dps_C": round(float(np.random.normal(0.008, 0.002)), 5),
                "v_offset_0h":            v_offset_0,
                "v_offset_24h":           v_offset_24,
                "snr_0h":                 round(float(np.random.normal(28.0, 1.0)), 2),
                "snr_24h":               round(float(28.0 + np.random.normal(0, 0.3)), 2),
            }
        elif device_family == "IMAGE_SENSOR":
            return {
                "DSNU_DN":               round(float(np.random.normal(12.0, 3.0)), 2),
                "hot_pixel_count":       int(max(0, np.random.poisson(5))),
                "supply_current_mA":     round(float(np.random.normal(42.0, 2.0)), 2),
                "dark_signal_mean_DN":   round(float(np.random.normal(8.0, 1.5)), 2),
                "v_offset_0h":           v_offset_0,
                "v_offset_24h":          v_offset_24,
                "snr_0h":                round(float(np.random.normal(62.0, 1.5)), 2),
                "snr_24h":               round(float(62.0 + np.random.normal(0, 0.5)), 2),
            }
        else:
            return {
                "v_offset_0h":  v_offset_0,
                "v_offset_24h": v_offset_24,
                "snr_0h":       snr_0,
                "snr_24h":      snr_24,
            }
