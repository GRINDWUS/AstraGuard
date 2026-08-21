"""
AstraGuard 2.0 — Comprehensive Validation Harness
====================================================
Implements a rigorous, reproducible experimental framework across three sub-systems:

  A. Pre-launch burn-in intelligence (prediction accuracy + FNR + chamber savings)
  B. Real-time ATE integration (latency, throughput, resilience)
  C. Post-launch telemetry health monitoring (drift detection lead-time)

DESIGN PRINCIPLES
-----------------
1. Strict information boundary: predictor NEVER sees 96h or 168h data.
2. Lot-based group split — train=LOT_01..05, val=LOT_06, blind_test=LOT_07.
3. All reported numbers are computed from the blind_test set.
4. Threshold sweep instead of a single magic number.
5. Results written to validation_results.json for dashboard consumption.
"""

import pandas as pd
import numpy as np
import json
import time
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from astraguard_core.predictor_fast import AstraGuardPredictorFast

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "validation_results.json")

TRAIN_LOTS     = ["LOT_2026_01", "LOT_2026_02", "LOT_2026_03", "LOT_2026_04", "LOT_2026_05"]
VAL_LOTS       = ["LOT_2026_06"]
BLIND_TEST_LOT = "LOT_2026_07"

# ATE policy: green/red = exit at 24h, yellow = continue to 72h
POLICY_HOURS = {"GREEN_AUTO_PASS": 24, "YELLOW_EXTENDED_TEST": 72, "RED_EARLY_REJECT": 24}
FULL_BURNIN_HOURS = 168

# ─── helpers ─────────────────────────────────────────────────────────────────

def load_lot(lot_id: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{lot_id}.csv")
    return pd.read_csv(path)

def strip_future(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 96h and 168h columns — enforces the information boundary."""
    return df.drop(columns=["iddq_96h", "iddq_168h_actual"], errors="ignore")

def compute_confusion(df_test: pd.DataFrame, predicted_tiers: pd.Series,
                      threshold_168h: float):
    """
    Build a binary 2×2 confusion matrix.
      Positive  = actually defective (iddq_168h_actual > threshold OR is_defective_gt == 1)
      Predicted = risk_tier != GREEN
    """
    y_true_pos = (df_test["is_defective_gt"] == 1).values
    y_pred_pos = (predicted_tiers != "GREEN_AUTO_PASS").values

    TP = int(( y_pred_pos &  y_true_pos).sum())
    FP = int(( y_pred_pos & ~y_true_pos).sum())
    FN = int((~y_pred_pos &  y_true_pos).sum())
    TN = int((~y_pred_pos & ~y_true_pos).sum())
    return TP, FP, FN, TN

def compute_chamber_hours(tier_counts: dict) -> dict:
    """Compute actual vs. traditional chamber-hours from tier distribution."""
    n_total = sum(tier_counts.values())
    traditional = n_total * FULL_BURNIN_HOURS
    astraguard  = sum(cnt * POLICY_HOURS.get(tier, FULL_BURNIN_HOURS)
                      for tier, cnt in tier_counts.items())
    saved_pct   = round((traditional - astraguard) / traditional * 100, 2)
    return {
        "traditional_hours": traditional,
        "astraguard_hours" : astraguard,
        "saved_hours"      : traditional - astraguard,
        "saved_percent"    : saved_pct
    }

# ─── Section A: Ban-in prediction validation ─────────────────────────────────

def section_a(predictor: AstraGuardPredictorFast, df_blind: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("SECTION A — Pre-Launch Burn-In Prediction Accuracy")
    print("="*60)

    df_input  = strip_future(df_blind)
    df_result = predictor.predict_lot(df_input)

    # Regression accuracy on components we have ground truth for
    gt_series  = df_blind["iddq_168h_actual"].values
    pred_series = df_result["predicted_168h_iddq"].values.astype(float)

    # Mask NaN/Inf from pred
    valid_mask = np.isfinite(pred_series) & np.isfinite(gt_series)
    mae  = float(np.mean(np.abs(pred_series[valid_mask] - gt_series[valid_mask])))
    rmse = float(np.sqrt(np.mean((pred_series[valid_mask] - gt_series[valid_mask])**2)))

    print(f"  Regression MAE  : {mae:.4f} µA  (valid pairs: {valid_mask.sum()})")
    print(f"  Regression RMSE : {rmse:.4f} µA")

    # Confusion matrix
    predicted_tiers = df_result["risk_tier"]
    TP, FP, FN, TN  = compute_confusion(df_blind, predicted_tiers, predictor.failure_threshold_168h)

    total       = TP + FP + FN + TN
    recall      = round(TP / (TP + FN) * 100, 4) if (TP + FN) > 0 else 0.0
    fnr         = round(FN / (TP + FN) * 100, 4) if (TP + FN) > 0 else 0.0
    precision   = round(TP / (TP + FP) * 100, 4) if (TP + FP) > 0 else 0.0
    fpr         = round(FP / (FP + TN) * 100, 4) if (FP + TN) > 0 else 0.0
    accuracy    = round((TP + TN) / total * 100, 4) if total > 0 else 0.0

    print(f"  Confusion: TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"  Recall (Sensitivity) : {recall}%")
    print(f"  FNR  (Escapes)       : {fnr}%   ← critical for space-grade")
    print(f"  FPR  (False Alarms)  : {fpr}%")
    print(f"  Precision            : {precision}%")
    print(f"  Accuracy             : {accuracy}%")

    # Tier distribution & chamber savings
    tier_counts = predicted_tiers.value_counts().to_dict()
    tier_counts.setdefault("GREEN_AUTO_PASS",      0)
    tier_counts.setdefault("YELLOW_EXTENDED_TEST", 0)
    tier_counts.setdefault("RED_EARLY_REJECT",     0)
    chamber = compute_chamber_hours(tier_counts)

    print(f"\n  Tier breakdown:")
    for tier, cnt in tier_counts.items():
        print(f"    {tier}: {cnt}  ({cnt/total*100:.1f}%)")
    print(f"\n  Chamber-hours (traditional) : {chamber['traditional_hours']}")
    print(f"  Chamber-hours (AstraGuard)  : {chamber['astraguard_hours']}")
    print(f"  Saved                       : {chamber['saved_percent']}%")

    return {
        "regression": {"mae_ua": round(mae, 4), "rmse_ua": round(rmse, 4), "valid_pairs": int(valid_mask.sum())},
        "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        "classification": {
            "recall_pct": recall, "fnr_pct": fnr, "fpr_pct": fpr,
            "precision_pct": precision, "accuracy_pct": accuracy
        },
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "chamber_savings": chamber
    }

# ─── Section A2: Threshold sweep ─────────────────────────────────────────────

def section_a2_threshold_sweep(predictor: AstraGuardPredictorFast, df_blind: pd.DataFrame) -> list:
    print("\n" + "="*60)
    print("SECTION A2 — Threshold Sensitivity Sweep")
    print("="*60)

    df_input  = strip_future(df_blind)
    df_result = predictor.predict_lot(df_input)
    pred_168  = df_result["predicted_168h_iddq"].values.astype(float)
    y_true    = (df_blind["is_defective_gt"] == 1).values
    n_total   = len(df_blind)

    sweep_results = []
    for threshold in [25, 30, 35, 40, 45, 50, 55, 60, 70]:
        # re-classify using this threshold regardless of original tier
        y_pred = pred_168 > threshold

        TP = int(( y_pred &  y_true).sum())
        FP = int(( y_pred & ~y_true).sum())
        FN = int((~y_pred &  y_true).sum())
        TN = int((~y_pred & ~y_true).sum())

        recall = round(TP / (TP + FN) * 100, 2) if (TP + FN) > 0 else 0.0
        fnr    = round(FN / (TP + FN) * 100, 2) if (TP + FN) > 0 else 0.0
        fpr    = round(FP / (FP + TN) * 100, 2) if (FP + TN) > 0 else 0.0

        n_flagged = int(y_pred.sum())
        saved_pct = round((n_total - n_flagged) * FULL_BURNIN_HOURS /
                          (n_total * FULL_BURNIN_HOURS) * 100, 2)

        row = {"threshold_ua": threshold, "recall_pct": recall, "fnr_pct": fnr,
               "fpr_pct": fpr, "n_flagged": n_flagged, "estimated_saved_pct": saved_pct}
        sweep_results.append(row)
        print(f"  Threshold={threshold:>3} µA | Recall={recall:>6}% | FNR={fnr:>5}% | FPR={fpr:>5}% | Saved≈{saved_pct}%")

    return sweep_results

# ─── Section A3: Noise robustness ────────────────────────────────────────────

def section_a3_noise_robustness(predictor: AstraGuardPredictorFast, df_blind: pd.DataFrame) -> list:
    print("\n" + "="*60)
    print("SECTION A3 — Noise Robustness (ATE Measurement Uncertainty)")
    print("="*60)

    results = []
    rng = np.random.default_rng(42)

    for sigma in [0.0, 0.1, 0.5, 1.0, 2.0]:
        df_noisy = strip_future(df_blind).copy()
        noise = rng.normal(0, sigma, len(df_noisy))
        df_noisy["iddq_24h"] = df_noisy["iddq_24h"] + noise

        df_res   = predictor.predict_lot(df_noisy)
        y_true   = (df_blind["is_defective_gt"] == 1).values
        y_pred   = (df_res["risk_tier"] != "GREEN_AUTO_PASS").values

        TP = int(( y_pred &  y_true).sum())
        FN = int((~y_pred &  y_true).sum())
        recall = round(TP / (TP + FN) * 100, 2) if (TP + FN) > 0 else 0.0
        fnr    = round(FN / (TP + FN) * 100, 2) if (TP + FN) > 0 else 0.0

        results.append({"noise_sigma_ua": sigma, "recall_pct": recall, "fnr_pct": fnr})
        print(f"  σ={sigma} µA → Recall={recall}%  FNR={fnr}%")

    return results

# ─── Section A4: Missing data handling ───────────────────────────────────────

def section_a4_missing_data(predictor: AstraGuardPredictorFast, df_blind: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("SECTION A4 — Missing / Corrupted ATE Data Handling")
    print("="*60)

    # Test 1: NaN in iddq_24h
    df_missing = strip_future(df_blind).copy()
    df_missing.loc[df_missing.sample(frac=0.1, random_state=42).index, "iddq_24h"] = np.nan

    crashed = False
    try:
        df_res = predictor.predict_lot(df_missing)
        n_valid = df_res["predicted_168h_iddq"].notna().sum()
        print(f"  10% NaN in iddq_24h → predictions produced: {n_valid}/{len(df_missing)}")
    except Exception as e:
        crashed = True
        print(f"  10% NaN in iddq_24h → CRASH: {e}")

    # Test 2: Negative IDDQ values (sensor malfunction)
    df_bad = strip_future(df_blind).copy()
    df_bad.loc[df_bad.sample(frac=0.05, random_state=42).index, "iddq_24h"] = -5.0
    crashed2 = False
    try:
        df_res2 = predictor.predict_lot(df_bad)
        print(f"  5% negative IDDQ → engine did not crash ✓")
    except Exception as e:
        crashed2 = True
        print(f"  5% negative IDDQ → CRASH: {e}")

    return {"nan_10pct_crash": crashed, "negative_val_crash": crashed2}

# ─── Section B: ATE integration latency ─────────────────────────────────────

def section_b_latency(predictor: AstraGuardPredictorFast, df_blind: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("SECTION B — ATE Integration Latency Benchmark")
    print("="*60)

    df_input = strip_future(df_blind).head(100)
    latencies_ms = []

    for idx, row in df_input.iterrows():
        single = pd.DataFrame([row])
        t0 = time.perf_counter()
        predictor.predict_lot(single)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    p50  = round(np.percentile(latencies_ms, 50), 2)
    p95  = round(np.percentile(latencies_ms, 95), 2)
    p99  = round(np.percentile(latencies_ms, 99), 2)
    mean = round(np.mean(latencies_ms), 2)

    print(f"  Inference latency (n=100 single-component calls):")
    print(f"    Mean  : {mean} ms")
    print(f"    P50   : {p50} ms")
    print(f"    P95   : {p95} ms")
    print(f"    P99   : {p99} ms")

    # Batch throughput
    t0 = time.perf_counter()
    predictor.predict_lot(strip_future(df_blind))
    t1 = time.perf_counter()
    batch_ms = round((t1 - t0) * 1000, 2)
    throughput = round(len(df_blind) / (t1 - t0), 0)

    print(f"  Batch ({len(df_blind)} components): {batch_ms} ms  |  {throughput} components/sec")

    return {
        "single_inference_ms": {"mean": mean, "p50": p50, "p95": p95, "p99": p99},
        "batch_inference_ms" : batch_ms,
        "batch_throughput_per_sec": int(throughput)
    }

# ─── Section C: Telemetry replay simulator ───────────────────────────────────

def section_c_telemetry_replay() -> dict:
    print("\n" + "="*60)
    print("SECTION C — Post-Launch Telemetry Replay Simulator")
    print("="*60)

    # Simulate three components with known trajectories
    baseline = 11.67  # µA pre-launch fingerprint

    def health_score(current_iddq, baseline_iddq, alpha=0.12):
        ratio = abs(current_iddq - baseline_iddq) / baseline_iddq
        return round(max(0.0, 100 * np.exp(-alpha * ratio * 10)), 1)

    scenarios = {
        "HEALTHY_NOMINAL": [11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4],
        "GRADUAL_DRIFT"  : [11.7, 12.1, 13.0, 14.5, 16.2, 18.8, 22.1, 26.4],
        "RAPID_RUNAWAY"  : [11.7, 13.5, 19.2, 35.0, 68.0, 120.0, 210.0, 380.0],
    }
    mission_days = [1, 30, 60, 90, 120, 150, 180, 210]

    WARNING_THRESHOLD = 60.0   # health score below this → early warning
    CRITICAL_THRESHOLD = 30.0

    results = {}
    for name, trajectory in scenarios.items():
        timeline = []
        warning_day  = None
        critical_day = None
        for day, iddq in zip(mission_days, trajectory):
            h = health_score(iddq, baseline)
            if warning_day is None and h < WARNING_THRESHOLD:
                warning_day = day
            if critical_day is None and h < CRITICAL_THRESHOLD:
                critical_day = day
            timeline.append({"day": day, "iddq_ua": iddq, "health_score": h})

        lead_time = None
        if critical_day and warning_day:
            lead_time = critical_day - warning_day

        results[name] = {
            "timeline"          : timeline,
            "warning_day"       : warning_day,
            "critical_day"      : critical_day,
            "lead_time_days"    : lead_time,
        }
        print(f"  {name}: warning_day={warning_day}  critical_day={critical_day}  lead={lead_time}d")

    return results

# ─── Main orchestrator ───────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ASTRAGUARD 2.0 — COMPREHENSIVE VALIDATION HARNESS      ║")
    print("║  Blind Test Lot: LOT_2026_07  (never used in training)   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Train model on LOT_01..05 only
    print("\n[*] Loading training data (LOT_01 — LOT_05)...")
    train_df = pd.concat([load_lot(l) for l in TRAIN_LOTS], ignore_index=True)
    print(f"    Training set: {len(train_df):,} components")

    predictor = AstraGuardPredictorFast(failure_threshold_168h=45.0)
    predictor.fit(train_df)
    print("    Model trained ✓")

    # Blind test
    print("\n[*] Loading blind test data (LOT_07 — held-out throughout development)...")
    df_blind = load_lot(BLIND_TEST_LOT)
    print(f"    Blind test set: {len(df_blind):,} components  "
          f"(defective: {df_blind['is_defective_gt'].sum()})")

    # Run all sections
    sec_a  = section_a(predictor, df_blind)
    sec_a2 = section_a2_threshold_sweep(predictor, df_blind)
    sec_a3 = section_a3_noise_robustness(predictor, df_blind)
    sec_a4 = section_a4_missing_data(predictor, df_blind)
    sec_b  = section_b_latency(predictor, df_blind)
    sec_c  = section_c_telemetry_replay()

    # Bundle and write results
    output = {
        "meta": {
            "train_lots"    : TRAIN_LOTS,
            "blind_test_lot": BLIND_TEST_LOT,
            "train_n"       : len(train_df),
            "blind_test_n"  : len(df_blind),
            "disclaimer"    : (
                "All numbers produced by running the trained model on a held-out "
                "synthetic dataset. Results reflect prototype feasibility, not "
                "space-grade hardware certification."
            )
        },
        "section_a_burnin_prediction"   : sec_a,
        "section_a2_threshold_sweep"    : sec_a2,
        "section_a3_noise_robustness"   : sec_a3,
        "section_a4_missing_data"       : sec_a4,
        "section_b_ate_latency"         : sec_b,
        "section_c_telemetry_replay"    : sec_c,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY — Blind Test Results (LOT_07)")
    print("="*60)
    cm = sec_a["classification"]
    ch = sec_a["chamber_savings"]
    print(f"  MAE              : {sec_a['regression']['mae_ua']} µA")
    print(f"  RMSE             : {sec_a['regression']['rmse_ua']} µA")
    print(f"  Recall           : {cm['recall_pct']}%")
    print(f"  FNR (Escapes)    : {cm['fnr_pct']}%")
    print(f"  FPR (False alarms): {cm['fpr_pct']}%")
    print(f"  Chamber-hrs saved: {ch['saved_percent']}%  "
          f"({ch['traditional_hours']}h → {ch['astraguard_hours']}h)")
    print(f"\n  Results written to: {OUT_PATH}")
    print("\n  ⚠  These are prototype/simulation results on synthetic data.")
    print("  Label as such in all presentations.\n")

if __name__ == "__main__":
    main()
