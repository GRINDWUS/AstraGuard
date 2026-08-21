"""
AstraGuard 2.1 — ASQD Generator (Phase 1 + Phase 2: Provenance)
================================================================
Implements the three locked architectural principles:
  1. ASQD is explicitly synthetic — NOT ISRO data.
  2. AstraGuard recommends; QA/qualification authority decides.
  3. Every result is reproducible from: lot_split + policy_version + model_version + test_condition.

Generates two output artefacts per lot:
  A. Raw Telemetry (1h-sampled, t=0..168):
     component_id, lot_id, wafer_id, hour, t_ambient_c, t_die_c,
     iddq_ua, v_cc_v, v_offset_mv, snr_db, failure_class, instrument_id
  
  B. Checkpoint Summary (t=0, t=24, t=168 pivoted to one row per component):
     Used as Stage A input to the ML pipeline.

  C. Provenance JSON per lot:
     dataset_id, generator_version, seed, lot metadata, failure distribution,
     temperature_profile, measurement_specs, synthetic=true
"""

import numpy as np
import pandas as pd
import os
import json
import hashlib
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from simulator.thermal_model import (
    ThermalProfile, ArrheniusParams,
    chamber_temperature_profile, arrhenius_coupling,
)
from simulator.failure_modes import (
    inject_nominal, inject_thermal_runaway, inject_mems_microcrack,
    inject_freak_outlier, inject_intermittent_fault, inject_sensor_fault,
)
from simulator.measurement_model import (
    InstrumentSpec, DEFAULT_INSTRUMENT_SPECS, measure_all_channels,
)

GENERATOR_VERSION = "2.1.0"
DATASET_NAME = "AstraGuard Synthetic Qualification Dataset (ASQD)"
DATASET_NOTE = (
    "Synthetic data generated using physically motivated degradation models "
    "and configurable test conditions. NOT ISRO proprietary data. "
    "Does not reproduce any specific ISRO test facility."
)

CLASS_NAMES = {
    0: "NOMINAL",
    1: "THERMAL_RUNAWAY",
    2: "MEMS_MICROCRACK",
    3: "FREAK_OUTLIER",
    4: "INTERMITTENT_FAULT",
    5: "SENSOR_FAULT",
}

# Training-phase class distribution (8% defect total for proper ML training)
TRAINING_CLASS_DIST = {0: 0.92, 1: 0.03, 2: 0.02, 3: 0.01, 4: 0.01, 5: 0.01}

# Stress-test distributions for Experiment 3 (rare-failure behaviour)
STRESS_DISTRIBUTIONS = {
    "D8":  {0: 0.92, 1: 0.03, 2: 0.02, 3: 0.01, 4: 0.01, 5: 0.01},
    "D5":  {0: 0.95, 1: 0.02, 2: 0.015, 3: 0.008, 4: 0.004, 5: 0.003},
    "D2":  {0: 0.98, 1: 0.009, 2: 0.005, 3: 0.003, 4: 0.002, 5: 0.001},
    "D1":  {0: 0.99, 1: 0.004, 2: 0.003, 3: 0.002, 4: 0.001, 5: 0.000},
    "D05": {0: 0.995, 1: 0.002, 2: 0.001, 3: 0.001, 4: 0.001, 5: 0.000},
}

V_CC_STRESS = 3.96       # +20% overvoltage stress (3.3V → 3.96V) for accelerated aging
R_TH = 0.05             # Thermal resistance °C·V/µA (prototype model constant)


def _assign_classes(num_components: int, dist: Dict[int, float], rng: np.random.Generator) -> np.ndarray:
    classes = list(dist.keys())
    probs = np.array(list(dist.values()), dtype=float)
    probs /= probs.sum()
    return rng.choice(classes, size=num_components, p=probs)


def generate_lot(
    lot_id: str,
    num_components: int = 1000,
    split: str = "TRAIN",
    class_distribution: Optional[Dict[int, float]] = None,
    thermal_profile: Optional[ThermalProfile] = None,
    arrhenius_params: Optional[ArrheniusParams] = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generate one full synthetic qualification lot.

    Returns:
        df_telemetry : Hourly time-series (num_components × 169 rows)
        df_checkpoint: One-row-per-component checkpoint summary
        provenance   : Reproducibility metadata dict
    """
    rng = np.random.default_rng(random_seed)

    if class_distribution is None:
        class_distribution = TRAINING_CLASS_DIST
    if thermal_profile is None:
        thermal_profile = ThermalProfile()
    if arrhenius_params is None:
        arrhenius_params = ArrheniusParams()

    # Normalise distribution
    dist = dict(class_distribution)
    total = sum(dist.values())
    dist = {k: v / total for k, v in dist.items()}

    # Time axis
    time_hours = np.arange(thermal_profile.observation_horizon_hours + 1)  # 0..168
    t_ambient  = chamber_temperature_profile(time_hours, thermal_profile)
    arrh_mult  = arrhenius_coupling(t_ambient, arrhenius_params)

    # Lot-level baseline draws (used for freak-outlier κ calculation)
    lot_base_iddq    = rng.normal(1.2, 0.08, num_components)
    lot_base_voffset = rng.normal(5.0, 0.15, num_components)
    lot_base_snr     = rng.normal(45.0, 0.5, num_components)
    kappa            = rng.uniform(3.5, 5.5, num_components)

    labels = _assign_classes(num_components, dist, rng)

    telemetry_rows  = []
    checkpoint_rows = []

    for ci in range(num_components):
        comp_id   = f"{lot_id}_COMP_{ci:04d}"
        wafer_id  = f"{lot_id}_W{ci // 100:02d}"
        wx        = round(float(rng.uniform(-1, 1)), 4)
        wy        = round(float(rng.uniform(-1, 1)), 4)
        label     = int(labels[ci])
        onset_h   = int(rng.integers(24, 49))

        base_iq  = lot_base_iddq[ci]
        base_vo  = lot_base_voffset[ci]
        base_snr = lot_base_snr[ci]

        # Freak outlier shifts baseline κ·σ above lot median
        if label == 3:
            base_iq = 1.2 + kappa[ci] * 0.08

        # Inject failure class
        if   label == 0: ch = inject_nominal(base_iq, base_vo, base_snr, t_ambient, arrh_mult, rng)
        elif label == 1: ch = inject_thermal_runaway(base_iq, base_vo, base_snr, t_ambient, arrh_mult, onset_h, rng)
        elif label == 2: ch = inject_mems_microcrack(base_iq, base_vo, base_snr, t_ambient, arrh_mult, onset_h, rng)
        elif label == 3: ch = inject_freak_outlier(base_iq, base_vo, base_snr, t_ambient, arrh_mult, rng)
        elif label == 4: ch = inject_intermittent_fault(base_iq, base_vo, base_snr, t_ambient, arrh_mult, rng)
        else:            ch = inject_sensor_fault(base_iq, base_vo, base_snr, t_ambient, arrh_mult, rng)

        true_iddq    = ch["iddq"]
        true_voffset = ch["voffset"]
        true_snr     = ch["snr"]
        true_tdie    = t_ambient + ((true_iddq * V_CC_STRESS) / 1000.0) * R_TH

        # Apply ATE measurement model (noise + quantization)
        obs = measure_all_channels(true_iddq, true_voffset, true_snr, true_tdie, rng)
        if label == 5:
            # Preserve exact channel freeze for Class 5 (SENSOR_FAULT)
            obs["iddq_ua"] = true_iddq

        # Raw hourly telemetry
        for t in range(len(time_hours)):
            telemetry_rows.append({
                "component_id":  comp_id,
                "lot_id":        lot_id,
                "wafer_id":      wafer_id,
                "wafer_x":       wx,
                "wafer_y":       wy,
                "hour":          int(time_hours[t]),
                "t_ambient_c":   round(float(t_ambient[t]), 2),
                "t_die_c":       round(float(obs["t_die_c"][t]), 2),
                "iddq_ua":       round(float(obs["iddq_ua"][t]), 3),
                "v_cc_v":        V_CC_STRESS,
                "v_offset_mv":   round(float(obs["v_offset_mv"][t]), 3),
                "snr_db":        round(float(obs["snr_db"][t]), 2),
                "instrument_id": "ASQD-ATE-SIM-01",
                "failure_class": label,
                "is_sensor_fault": int(label == 5),
            })

        # Checkpoint summary
        def at(ch_arr, t): return round(float(ch_arr[t]), 3)
        checkpoint_rows.append({
            "component_id":      comp_id,
            "lot_id":            lot_id,
            "wafer_id":          wafer_id,
            "wafer_x":           wx,
            "wafer_y":           wy,
            "device_family":     "SPACE_MEMS_SENSOR",
            "device_spec_id":    "ASQD-MEMS-SENS-2026",
            "operating_voltage_v": V_CC_STRESS,
            "test_temperature_c": thermal_profile.t_ambient_peak_c,
            "observation_horizon_h": thermal_profile.observation_horizon_hours,
            # Channel readings at checkpoints
            "iddq_0h":          at(obs["iddq_ua"], 0),
            "iddq_24h":         at(obs["iddq_ua"], 24),
            "iddq_168h_actual": at(obs["iddq_ua"], 168),
            "v_offset_0h":      at(obs["v_offset_mv"], 0),
            "v_offset_24h":     at(obs["v_offset_mv"], 24),
            "v_offset_168h":    at(obs["v_offset_mv"], 168),
            "snr_0h":           at(obs["snr_db"], 0),
            "snr_24h":          at(obs["snr_db"], 24),
            "snr_168h":         at(obs["snr_db"], 168),
            "t_die_0h":         at(obs["t_die_c"], 0),
            "t_die_24h":        at(obs["t_die_c"], 24),
            # Deltas (temporal derivatives)
            "delta_iddq":       round(at(obs["iddq_ua"], 24) - at(obs["iddq_ua"], 0), 3),
            "delta_v_offset":   round(at(obs["v_offset_mv"], 24) - at(obs["v_offset_mv"], 0), 3),
            "delta_snr":        round(at(obs["snr_db"], 24) - at(obs["snr_db"], 0), 3),
            # Specification limits (configurable)
            "spec_max_iddq":    15.0,
            # Ground truth labels
            "is_defective_gt":  0 if label == 0 else 1,
            "failure_class":    label,
            "failure_mode_gt":  CLASS_NAMES[label],
            "is_sensor_fault":  int(label == 5),
        })

    df_tel = pd.DataFrame(telemetry_rows)
    df_chk = pd.DataFrame(checkpoint_rows)

    # Build provenance record (Principle 3: reproducibility)
    class_counts = df_chk["failure_mode_gt"].value_counts().to_dict()
    provenance = {
        "dataset_name":       DATASET_NAME,
        "dataset_id":         f"ASQD_v{GENERATOR_VERSION}_{lot_id}",
        "generator_version":  GENERATOR_VERSION,
        "random_seed":        random_seed,
        "lot_id":             lot_id,
        "split":              split,
        "num_components":     num_components,
        "observation_horizon_hours": thermal_profile.observation_horizon_hours,
        "observation_note":   (
            "168h is the AstraGuard prototype observation horizon. "
            "Not a claimed universal qualification duration. "
            "MIL-STD-883 Method 1015 conditions are method-dependent."
        ),
        "sampling_interval_hours": 1,
        "failure_distribution": {CLASS_NAMES[k]: round(v * 100, 2) for k, v in dist.items()},
        "class_counts":       class_counts,
        "defect_rate_pct":    round(df_chk["is_defective_gt"].mean() * 100, 2),
        "temperature_profile": {
            "t_start_c": thermal_profile.t_ambient_start_c,
            "t_peak_c":  thermal_profile.t_ambient_peak_c,
            "ramp_tau_h": thermal_profile.ramp_tau_hours,
            "profile_type": thermal_profile.profile_type,
            "profile_source": thermal_profile.profile_source,
            "qualification_standard": thermal_profile.qualification_standard,
        },
        "measurement_model": {
            "iddq_noise_sigma_ua":       DEFAULT_INSTRUMENT_SPECS["iddq_ua"].noise_sigma,
            "iddq_quantization_lsb_ua":  DEFAULT_INSTRUMENT_SPECS["iddq_ua"].quantization_lsb,
            "voffset_noise_sigma_mv":    DEFAULT_INSTRUMENT_SPECS["v_offset_mv"].noise_sigma,
            "snr_noise_sigma_db":        DEFAULT_INSTRUMENT_SPECS["snr_db"].noise_sigma,
        },
        "synthetic": True,
        "synthetic_note": DATASET_NOTE,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        # Checksum of generator params for reproducibility
        "config_hash": hashlib.md5(
            f"{GENERATOR_VERSION}{random_seed}{lot_id}{num_components}".encode()
        ).hexdigest()[:12],
    }

    return df_tel, df_chk, provenance


def generate_all_lots(
    output_dir: str,
    lots_config: Optional[List[dict]] = None,
    verbose: bool = True,
):
    """
    Generate the full ASQD lot manifest with lot-based train/val/blind splits.
    NEVER produces a row-level random split — entire lots are assigned to splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    tel_dir = os.path.join(output_dir, "telemetry")
    chk_dir = os.path.join(output_dir, "summaries")
    prv_dir = os.path.join(output_dir, "provenance")
    for d in [tel_dir, chk_dir, prv_dir]:
        os.makedirs(d, exist_ok=True)

    if lots_config is None:
        lots_config = [
            *[{"lot_id": f"LOT_{i:02d}", "n": 1000, "seed": 40+i, "split": "TRAIN"} for i in range(1, 8)],
            *[{"lot_id": f"LOT_{i:02d}", "n": 1000, "seed": 40+i, "split": "VAL"}   for i in range(8, 10)],
            *[{"lot_id": f"LOT_{i:02d}", "n": 1000, "seed": 40+i, "split": "BLIND_TEST"} for i in range(10, 12)],
        ]

    manifest_rows = []
    for cfg in lots_config:
        lot_id = cfg["lot_id"]
        split  = cfg["split"]
        n      = cfg.get("n", 1000)
        seed   = cfg.get("seed", 42)
        dist   = cfg.get("distribution", None)

        if verbose:
            print(f"  Generating {lot_id} [{split}]  n={n}  seed={seed}")

        df_tel, df_chk, prov = generate_lot(
            lot_id=lot_id, num_components=n, split=split,
            class_distribution=dist, random_seed=seed,
        )

        df_tel.to_csv(os.path.join(tel_dir, f"{lot_id}_telemetry.csv"), index=False)
        df_chk.to_csv(os.path.join(chk_dir, f"{lot_id}.csv"), index=False)
        with open(os.path.join(prv_dir, f"{lot_id}_provenance.json"), "w") as f:
            json.dump(prov, f, indent=2)

        manifest_rows.append({
            "lot_id": lot_id, "split": split,
            "num_components": n,
            "defect_count": int(df_chk["is_defective_gt"].sum()),
            "defect_rate_pct": round(df_chk["is_defective_gt"].mean() * 100, 2),
            "sensor_faults": int(df_chk["is_sensor_fault"].sum()),
            "config_hash": prov["config_hash"],
        })
        if verbose:
            dc = df_chk["failure_mode_gt"].value_counts().to_dict()
            for k, v in dc.items():
                print(f"    {k:<25}: {v}")
            print()

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(output_dir, "ASQD_manifest.csv"), index=False)
    if verbose:
        print("=" * 60)
        print(f"ASQD Complete. Manifest: {output_dir}/ASQD_manifest.csv")
        total_tel = sum(r["num_components"] for r in manifest_rows) * 169
        print(f"Total telemetry rows: {total_tel:,}")
    return manifest


if __name__ == "__main__":
    print("=" * 60)
    print("AstraGuard 2.1 — ASQD Generator")
    print("=" * 60)
    generate_all_lots(
        output_dir="D:\\SIH 2026\\astraguard_core\\data",
        lots_config=[
            *[{"lot_id": f"LOT_{i:02d}", "n": 500, "seed": 40+i, "split": "TRAIN"} for i in range(1, 8)],
            *[{"lot_id": f"LOT_{i:02d}", "n": 500, "seed": 40+i, "split": "VAL"}   for i in range(8, 10)],
            *[{"lot_id": f"LOT_{i:02d}", "n": 500, "seed": 40+i, "split": "BLIND_TEST"} for i in range(10, 12)],
        ],
    )
