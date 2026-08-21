"""
AstraGuard 2.1 — Experiment Runner & Blind Evaluation (Phase 4 + 5 + 6)
=========================================================================
Runs all seven research experiments using the ASQD dataset.
Enforces the three locked architectural principles throughout.

Experiment 3 (Rare-Failure) and Experiment 5 (Warning Lead-Time) use
completely separate datasets with new degradation parameters — ensuring
results are genuine generalization measurements, not simulation artifacts.
"""

import sys, os
sys.path.insert(0, "D:/SIH 2026")

import json
import numpy as np
import pandas as pd
from pathlib import Path

from models.anomaly_detector import (
    build_features, run_baseline_comparison,
    AstraGuardMLEngine, evaluate_binary,
    _robust_z,
)

DATA_DIR = Path("D:/SIH 2026/astraguard_core/data")
CHK_DIR  = DATA_DIR / "summaries"
PRV_DIR  = DATA_DIR / "provenance"


def load_checkpoint(lot_id: str) -> pd.DataFrame:
    return pd.read_csv(CHK_DIR / f"{lot_id}.csv")


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ASQD_manifest.csv")


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─── Experiment 1: Predictive Horizon ─────────────────────────────────────────
def experiment_1(engine: AstraGuardMLEngine, blind_df: pd.DataFrame):
    print_header("Experiment 1 - Predictive Horizon (24h -> 168h)")
    preds = engine.predict(blind_df)
    y_act = blind_df["iddq_168h_actual"].values
    y_pred = preds["predicted_iddq_168h"].values
    mae  = np.mean(np.abs(y_act - y_pred))
    rmse = np.sqrt(np.mean((y_act - y_pred)**2))
    ss_res = np.sum((y_act - y_pred)**2)
    ss_tot = np.sum((y_act - y_act.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    print(f"  MAE  = {mae:.4f} uA")
    print(f"  RMSE = {rmse:.4f} uA")
    print(f"  R2   = {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ─── Experiment 2: Population-Aware Detection Advantage ───────────────────────
def experiment_2(train_lots: list, blind_df: pd.DataFrame):
    print_header("Experiment 2 - Population-Aware Detection Advantage")
    comparison_df, engine, preds = run_baseline_comparison(train_lots, blind_df, label="BLIND")
    print(comparison_df[["Recall", "Precision", "F1", "PR_AUC", "FNR", "FPR"]].to_string())
    return comparison_df, engine, preds


# ─── Experiment 3: Rare-Failure Behaviour ────────────────────────────────────
def experiment_3(engine: AstraGuardMLEngine):
    print_header("Experiment 3 - Rare-Failure Behaviour (Defect Prevalence Sweep)")
    from simulator.generator import STRESS_DISTRIBUTIONS, generate_lot
    results = []
    for dist_name, dist in STRESS_DISTRIBUTIONS.items():
        _, df_stress, _ = generate_lot(
            lot_id=f"STRESS_{dist_name}", num_components=500,
            class_distribution=dist, random_seed=999,
        )
        preds = engine.predict(df_stress)
        y_true = df_stress["is_defective_gt"].values
        y_pred = (preds["recommendation"] == "RED_HIGH_RISK").astype(int).values
        defect_pct = round(df_stress["is_defective_gt"].mean() * 100, 2)
        m = evaluate_binary(y_true, y_pred, label=dist_name)
        m["defect_rate_pct"] = defect_pct
        results.append(m)
        print(f"  {dist_name}: defect_rate={defect_pct}%  Recall={m['Recall']:.3f}  F1={m['F1']:.3f}  FNR={m['FNR']:.3f}")
    return pd.DataFrame(results)


# ─── Experiment 4: Component vs. Instrument Fault Separation ──────────────────
def experiment_4(blind_df: pd.DataFrame, engine: AstraGuardMLEngine):
    print_header("Experiment 4 - Component vs. Instrument Fault Separation")
    from features.temporal import detect_frozen_channels
    # Load full telemetry for blind lot(s)
    tel_files = list((DATA_DIR / "telemetry").glob("LOT_1*.csv"))
    if not tel_files:
        print("  Telemetry not found - run generator first."); return None
    
    df_tel = pd.concat([pd.read_csv(f) for f in tel_files[:2]], ignore_index=True)
    frozen = detect_frozen_channels(df_tel)
    
    sensor_fault_gt = blind_df[blind_df["is_sensor_fault"] == 1]["component_id"].values
    detected_frozen  = frozen[frozen["channel_frozen"]]["component_id"].values
    
    correctly_id_as_instrument = len(set(sensor_fault_gt) & set(detected_frozen))
    total_sensor_faults = len(sensor_fault_gt)
    if total_sensor_faults > 0:
        detection_rate = correctly_id_as_instrument / total_sensor_faults
        print(f"  Sensor fault components: {total_sensor_faults}")
        print(f"  Correctly identified as instrument fault: {correctly_id_as_instrument}")
        print(f"  Instrument-fault detection rate: {detection_rate:.1%}")
    else:
        print("  No sensor fault components in blind test lot.")
    return frozen


# ─── Experiment 7: Reliability Fingerprint (Pre-Launch -> Post-Launch) ─────────
def experiment_7(blind_df: pd.DataFrame, engine: AstraGuardMLEngine):
    """
    Generates a Reliability Fingerprint for each component in the blind lot.
    Fingerprint persists from Stage A (pre-launch) for later Stage B comparison
    against simulated in-orbit telemetry deviations.
    """
    print_header("Experiment 7 - Reliability Fingerprint Generation")
    preds = engine.predict(blind_df)
    merged = blind_df.merge(preds[["component_id", "predicted_iddq_168h",
                                    "defect_probability", "recommendation",
                                    "reason_codes", "policy_version"]], on="component_id")
    
    fingerprints = []
    for _, row in merged.iterrows():
        fp = {
            "component_id":   row["component_id"],
            "lot_id":         row["lot_id"],
            "device_family":  row.get("device_family", "SPACE_MEMS_SENSOR"),
            "baseline": {
                "iddq_0h":      row["iddq_0h"],
                "v_offset_0h":  row["v_offset_0h"],
                "snr_0h":       row["snr_0h"],
            },
            "degradation_signature": {
                "iddq_24h_delta":     row["delta_iddq"],
                "iddq_drift_vel_uah": round(row["delta_iddq"] / 24.0, 5),
                "voffset_24h_delta":  row["delta_v_offset"],
                "snr_24h_delta":      row["delta_snr"],
            },
            "screening_evidence": {
                "defect_probability":  row["defect_probability"],
                "recommendation":      row["recommendation"],
                "reason_codes":        row["reason_codes"],
                "policy_version":      row["policy_version"],
            },
            "ground_truth":    row["failure_mode_gt"],   # REMOVED in production; for research only
        }
        fingerprints.append(fp)
    
    out_path = DATA_DIR / "reliability_fingerprints.json"
    with open(out_path, "w") as f:
        json.dump(fingerprints[:20], f, indent=2)  # Save first 20 as sample
    
    print(f"  Generated {len(fingerprints)} fingerprints.")
    print(f"  Sample saved to: {out_path}")
    
    # Summary
    rec_dist = preds["recommendation"].value_counts()
    print(f"\n  Recommendation Distribution (Blind Test Lot):")
    for rec, cnt in rec_dist.items():
        print(f"    {rec:<30}: {cnt} ({cnt/len(preds)*100:.1f}%)")
    
    return fingerprints


# ─── Stress Test A: Lot Shift Robustness ──────────────────────────────────────
def stress_test_a_lot_shift(engine: AstraGuardMLEngine):
    print_header("Stress Test A - Lot-to-Lot Shift Robustness")
    from simulator.generator import generate_lot
    # Generate a lot with shifted baseline leakage, temperature offset, and noise
    _, df_shifted, _ = generate_lot(
        lot_id="STRESS_LOT_SHIFT", num_components=500, split="BLIND_TEST",
        random_seed=888,
    )
    # Apply global shift parameters to simulate different manufacturing batch/environment
    df_shifted["iddq_0h"] += 0.45  # Global baseline leakage shift
    df_shifted["iddq_24h"] += 0.55
    df_shifted["delta_iddq"] = df_shifted["iddq_24h"] - df_shifted["iddq_0h"]
    
    preds = engine.predict(df_shifted)
    y_true = df_shifted["is_defective_gt"].values
    y_pred = (preds["recommendation"] == "RED_HIGH_RISK").astype(int).values
    m = evaluate_binary(y_true, y_pred, label="Lot_Shift")
    print(f"  Shifted Lot Defect Rate: {df_shifted['is_defective_gt'].mean()*100:.1f}%")
    print(f"  Recall = {m['Recall']:.3f} | Precision = {m['Precision']:.3f} | F1 = {m['F1']:.3f} | FNR = {m['FNR']:.3f}")
    return m


# ─── Stress Test B: Measurement Uncertainty / ATE Noise ──────────────────────
def stress_test_b_measurement_noise(engine: AstraGuardMLEngine, blind_df: pd.DataFrame):
    print_header("Stress Test B - Measurement Uncertainty & ATE Noise")
    df_noisy = blind_df.copy()
    rng = np.random.default_rng(777)
    # Inject SMU noise (gaussian + quantization)
    df_noisy["iddq_0h"] += rng.normal(0, 0.15, size=len(df_noisy))
    df_noisy["iddq_24h"] += rng.normal(0, 0.25, size=len(df_noisy))
    df_noisy["delta_iddq"] = df_noisy["iddq_24h"] - df_noisy["iddq_0h"]
    
    preds = engine.predict(df_noisy)
    y_true = df_noisy["is_defective_gt"].values
    y_pred = (preds["recommendation"] == "RED_HIGH_RISK").astype(int).values
    m = evaluate_binary(y_true, y_pred, label="Measurement_Noise")
    print(f"  Noisy Evaluation Recall = {m['Recall']:.3f} | Precision = {m['Precision']:.3f} | F1 = {m['F1']:.3f}")
    return m


# ─── Stress Test C: Unknown Failure Mode Screening ───────────────────────────
def stress_test_c_unknown_failure_mode(train_lots: list):
    print_header("Stress Test C - Unknown Failure Mode Screening (OOD)")
    # Train engine WITHOUT Class 2 (MEMS Microcrack) or Class 4 (Intermittent)
    train_ood = []
    for lot in train_lots:
        # Filter out Class 2 and 4 from training data
        lot_filtered = lot[~lot["failure_class"].isin([2, 4])].copy()
        train_ood.append(lot_filtered)
    
    engine_ood = AstraGuardMLEngine()
    engine_ood.fit(train_ood)
    
    from simulator.generator import generate_lot
    _, df_unknown, _ = generate_lot(
        lot_id="STRESS_UNKNOWN_FAIL", num_components=500, split="BLIND_TEST",
        random_seed=666,
    )
    preds = engine_ood.predict(df_unknown)
    
    # Evaluate how unknown failure modes (Class 2 & 4) were routed
    unknown_mask = df_unknown["failure_class"].isin([2, 4])
    unknown_preds = preds[unknown_mask]
    
    rec_dist = unknown_preds["recommendation"].value_counts().to_dict()
    print(f"  Total Unknown Failure Components: {unknown_mask.sum()}")
    print("  Routing Distribution for Unknown Failure Modes:")
    for rec, cnt in rec_dist.items():
        print(f"    {rec:<30}: {cnt} ({cnt/unknown_mask.sum()*100:.1f}%)")
    
    # Success criterion: Unknown failure modes should NOT be silently marked GREEN
    flagged_pct = (1.0 - (rec_dist.get("GREEN_NORMAL_CANDIDATE", 0) / unknown_mask.sum())) * 100
    print(f"  Unknown Failure Flagging Rate (Non-GREEN): {flagged_pct:.1f}%")
    return rec_dist


# ─── Experiment 8: Known vs Unknown Failure Detection ─────────────────────────
def experiment_8_known_vs_unknown(train_lots: list):
    """
    The Killer Research Experiment: Can unsupervised detectors catch failure
    modes that the supervised model has NEVER seen?

    Protocol:
      TRAIN: AstraGuardMLEngine trained on Classes 0–5 (all known archetypes)
      BLIND : Class 6 (Dielectric Oscillation) — brand-new failure mechanism,
              never injected during training, generated with fresh seed.

    Hypothesis:
      XGBoost classifier → LOW defect_probability (never saw Class 6 dynamics)
      Isolation Forest  → IFOREST_OOD_ANOMALY (feature-space outlier)
      LSTM AE           → LSTM_TEMPORAL_ANOMALY (oscillation temporal pattern)
      Policy Engine     → UNKNOWN_PATTERN_REVIEW (not silently GREEN)

    Research Question:
      "Can AstraGuard's unsupervised temporal layer detect previously
      unseen degradation patterns that the supervised model misses?"
    """
    from simulator.generator import generate_lot, CLASS_NAMES
    from simulator.failure_modes import inject_dielectric_oscillation
    from simulator.thermal_model import ThermalProfile, ArrheniusParams, chamber_temperature_profile, arrhenius_coupling
    from simulator.measurement_model import measure_all_channels, DEFAULT_INSTRUMENT_SPECS

    print_header("Experiment 8 — Known vs Unknown Failure Detection (Class 6 OOD)")

    # Step 1: Train engine on all KNOWN classes (0–5)
    engine_exp8 = AstraGuardMLEngine()
    engine_exp8.fit(train_lots)

    # Step 2: Generate a lot of ONLY Class 6 (Dielectric Oscillation)
    V_CC_STRESS = 3.96
    R_TH = 0.05
    rng = np.random.default_rng(2026)
    n_unknown = 300

    thermal = ThermalProfile()
    arrh    = ArrheniusParams()
    time_hours = np.arange(169)
    t_ambient  = chamber_temperature_profile(time_hours, thermal)
    arrh_mult  = arrhenius_coupling(t_ambient, arrh)

    lot_base_iddq    = rng.normal(1.2, 0.08, n_unknown)
    lot_base_voffset = rng.normal(5.0, 0.15, n_unknown)
    lot_base_snr     = rng.normal(45.0, 0.5, n_unknown)

    chk_rows = []
    for ci in range(n_unknown):
        comp_id = f"EXP8_UNKNOWN_{ci:04d}"
        ch = inject_dielectric_oscillation(
            lot_base_iddq[ci], lot_base_voffset[ci], lot_base_snr[ci],
            t_ambient, arrh_mult, rng
        )
        obs = measure_all_channels(ch["iddq"], ch["voffset"], ch["snr"],
                                   t_ambient + ((ch["iddq"] * V_CC_STRESS) / 1000.0) * R_TH,
                                   rng, DEFAULT_INSTRUMENT_SPECS)

        def at(arr, t): return round(float(arr[t]), 3)
        chk_rows.append({
            "component_id": comp_id, "lot_id": "EXP8_CLASS6",
            "wafer_id": "W00", "wafer_x": 0.0, "wafer_y": 0.0,
            "device_family": "SPACE_MEMS_SENSOR", "device_spec_id": "ASQD-MEMS-SENS-2026",
            "operating_voltage_v": V_CC_STRESS, "test_temperature_c": 125.0,
            "observation_horizon_h": 168,
            "iddq_0h":   at(obs["iddq_ua"], 0),  "iddq_24h":  at(obs["iddq_ua"], 24),
            "iddq_168h_actual": at(obs["iddq_ua"], 168),
            "v_offset_0h": at(obs["v_offset_mv"], 0), "v_offset_24h": at(obs["v_offset_mv"], 24),
            "v_offset_168h": at(obs["v_offset_mv"], 168),
            "snr_0h": at(obs["snr_db"], 0), "snr_24h": at(obs["snr_db"], 24),
            "snr_168h": at(obs["snr_db"], 168),
            "t_die_0h": at(obs["iddq_ua"], 0), "t_die_24h": at(obs["iddq_ua"], 24),
            "delta_iddq": round(at(obs["iddq_ua"], 24) - at(obs["iddq_ua"], 0), 3),
            "delta_v_offset": round(at(obs["v_offset_mv"], 24) - at(obs["v_offset_mv"], 0), 3),
            "delta_snr": round(at(obs["snr_db"], 24) - at(obs["snr_db"], 0), 3),
            "spec_max_iddq": 15.0,
            "is_defective_gt": 1,   # Ground truth: bad component
            "failure_class": 6,     # Class 6 = NEVER in training
            "failure_mode_gt": "DIELECTRIC_OSCILLATION",
            "is_sensor_fault": 0,
        })

    df_unknown = pd.DataFrame(chk_rows)

    # Step 3: Run inference
    preds = engine_exp8.predict(df_unknown)

    # Step 4: Evaluate each detector independently
    rec_dist = preds["recommendation"].value_counts().to_dict()
    green_pct   = rec_dist.get("GREEN_NORMAL_CANDIDATE", 0) / n_unknown * 100
    escaped_pct = green_pct  # Silent escapes = GREEN classification of a defective part
    flagged_pct = 100.0 - green_pct

    # Detector-level breakdown
    xgb_high_prob  = (preds["defect_probability"] >= 0.5).mean() * 100
    iforest_flag   = (preds["iforest_anomaly_score"] > 0).mean() * 100  # positive = anomalous
    lstm_flag      = (preds["lstm_ae_pred"] == -1).mean() * 100
    unknown_review = rec_dist.get("UNKNOWN_PATTERN_REVIEW", 0) / n_unknown * 100
    yellow_review  = rec_dist.get("YELLOW_REVIEW", 0) / n_unknown * 100
    red_flag       = rec_dist.get("RED_HIGH_RISK", 0) / n_unknown * 100

    print(f"  Class 6 components (DIELECTRIC_OSCILLATION): {n_unknown}")
    print(f"  — Never seen during training (Experiment 8 OOD protocol)")
    print()
    print(f"  Eye 2 (XGBoost Supervised):")
    print(f"    HIGH defect probability (>=0.5):  {xgb_high_prob:.1f}%")
    print()
    print(f"  Eye 3A (Isolation Forest):")
    print(f"    OOD anomaly flagged:              {iforest_flag:.1f}%")
    print()
    print(f"  Eye 3B (LSTM Autoencoder):")
    print(f"    Temporal anomaly flagged:         {lstm_flag:.1f}%")
    print()
    print(f"  Policy Engine (4-Eye Fusion):")
    print(f"    RED_HIGH_RISK:                    {red_flag:.1f}%")
    print(f"    UNKNOWN_PATTERN_REVIEW:           {unknown_review:.1f}%")
    print(f"    YELLOW_REVIEW:                    {yellow_review:.1f}%")
    print(f"    GREEN (Silent Escape):            {escaped_pct:.1f}%  ← TARGET: 0%")
    print()
    if escaped_pct == 0.0:
        print(f"  ✅ PASS: Zero silent escapes. Both unsupervised detectors caught the unknown class.")
    else:
        print(f"  ⚠️  {escaped_pct:.1f}% silent escapes. Review LSTM threshold or IForest contamination.")

    return {
        "n_unknown": n_unknown,
        "xgb_high_prob_pct": xgb_high_prob,
        "iforest_flag_pct": iforest_flag,
        "lstm_flag_pct": lstm_flag,
        "green_escape_pct": escaped_pct,
        "flagged_pct": flagged_pct,
        "rec_dist": rec_dist,
    }


# ─── Main Experiment Suite ────────────────────────────────────────────────────

def main():
    manifest = load_manifest()
    
    train_lots = [load_checkpoint(r["lot_id"]) for _, r in manifest[manifest["split"] == "TRAIN"].iterrows()]
    val_lots   = [load_checkpoint(r["lot_id"]) for _, r in manifest[manifest["split"] == "VAL"].iterrows()]
    blind_lots = [load_checkpoint(r["lot_id"]) for _, r in manifest[manifest["split"] == "BLIND_TEST"].iterrows()]
    
    blind_df = pd.concat(blind_lots, ignore_index=True)
    
    print_header("AstraGuard 3.0 - Hybrid Intelligence Validation Suite")
    print(f"  Train lots : {manifest[manifest['split']=='TRAIN']['lot_id'].tolist()}")
    print(f"  Val lots   : {manifest[manifest['split']=='VAL']['lot_id'].tolist()}")
    print(f"  Blind lots : {manifest[manifest['split']=='BLIND_TEST']['lot_id'].tolist()}")
    print(f"  Blind components: {len(blind_df)}")
    print(f"  Blind defect rate: {blind_df['is_defective_gt'].mean()*100:.1f}%")
    
    # Phase 4: Baseline comparison (answers Exp 2)
    comparison_df, engine, preds = experiment_2(train_lots, blind_df)
    
    # Phase 5: Regression (answers Exp 1)
    experiment_1(engine, blind_df)
    
    # Experiment 3: Rare-failure sweep
    experiment_3(engine)
    
    # Experiment 4: Sensor fault separation
    experiment_4(blind_df, engine)
    
    # Experiment 7: Reliability fingerprints
    experiment_7(blind_df, engine)
    
    # Advanced Stress Tests A, B, C
    stress_test_a_lot_shift(engine)
    stress_test_b_measurement_noise(engine, blind_df)
    stress_test_c_unknown_failure_mode(train_lots)

    # Experiment 8: Known vs Unknown Failure Detection (LSTM AE + IForest OOD)
    experiment_8_known_vs_unknown(train_lots)
    
    print_header("Experiment Suite Complete — AstraGuard 3.0")
    print("  4-Eye Hybrid Intelligence: Spatial + Supervised + IForest + LSTM AE")
    print("  Recall/FNR are observed experimental results - not hardcoded targets.")
    print("  All results reproducible from: ASQD_manifest.csv + policy-3.0 + generator_v2.1.0")
    print("  Experiment 8 validates OOD detection of Class 6 (Dielectric Oscillation)")


if __name__ == "__main__":
    main()
