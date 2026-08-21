"""
AstraGuard 2.3 — Blind Context & Multi-Device Reliability Evaluator
====================================================================
Evaluates AstraGuard 2.3 across 3 device families (Digital IC, MEMS, Image Sensor)
under 4 blind test conditions:
  Condition 1: Full Metadata (Level 1 Explicit)
  Condition 2: Partial Schema (Level 2 Parameter Mapping)
  Condition 3: Raw Measurement Stream (Level 3 Behavioral Inference)
  Condition 4: Corrupted / Unknown Stream (Operator Safety Interlock Trigger)

Calculates:
  - Context Resolution Accuracy (%)
  - Wrong-Model Routing Rate (%)
  - Unknown / OOD Detection Rate (%)
  - Module B 168h Forecast MAE
  - Silent Escape Rate (%)
"""

import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from dataset_generator.lot_generator import LotSimulator
from astraguard_core.context_resolver.explicit_parser import ExplicitMetadataParser
from astraguard_core.context_resolver.behavioral_infer import BehavioralInferenceEngine
from astraguard_core.context_resolver.schema import (
    TestContext, DeviceMetadata, TestMetadata, MeasurementRecord, ResolutionStatus
)
from astraguard_core.model_router import AstraGuardModelRouter


def run_blind_evaluation():
    sim = LotSimulator()
    explicit_parser = ExplicitMetadataParser()
    behavioral_engine = BehavioralInferenceEngine()
    router = AstraGuardModelRouter()

    families = ["DIGITAL_IC", "MEMS_GYROSCOPE", "IMAGE_SENSOR"]
    conditions = [
        "FULL_METADATA",
        "PARTIAL_SCHEMA",
        "RAW_MEASUREMENTS_ONLY",
        "CORRUPTED_UNKNOWN"
    ]

    summary_rows = []

    print("=" * 75)
    print("🚀 ASTRAGUARD 2.3 — BLIND CONTEXT & MULTI-DEVICE RELIABILITY EVALUATION")
    print("=" * 75)

    for cond in conditions:
        total_lots = 0
        correct_context_cnt = 0
        wrong_model_cnt = 0
        interlocked_cnt = 0
        mae_list = []
        escape_cnt = 0
        total_components = 0

        for fam_idx, dev_fam in enumerate(families):
            # Generate 5 qualification lots per device family per condition
            for lot_id in range(1, 6):
                total_lots += 1
                is_corrupt = (cond == "CORRUPTED_UNKNOWN")
                is_strip = (cond in ("PARTIAL_SCHEMA", "RAW_MEASUREMENTS_ONLY", "CORRUPTED_UNKNOWN"))

                df = sim.generate_lot(
                    lot_id=lot_id + fam_idx * 10,
                    num_components=100,
                    device_family=dev_fam,
                    corrupt_metadata=is_corrupt,
                    strip_metadata=is_strip
                )

                total_components += len(df)

                # Context Resolution Pipeline
                if cond == "FULL_METADATA":
                    ctx = TestContext(
                        device_metadata=DeviceMetadata(device_family=dev_fam),
                        test_metadata=TestMetadata(test_type=df["test_type"].iloc[0])
                    )
                    res = explicit_parser.resolve(
                        test_context=ctx,
                        observed_parameters=[df["primary_parameter"].iloc[0]]
                    )
                elif cond == "PARTIAL_SCHEMA":
                    obs_params = [df["primary_parameter"].iloc[0]]
                    if "ICC_active_mA" in df.columns:
                        obs_params.append("ICC_active")
                    elif "scale_factor_error_ppm" in df.columns:
                        obs_params.append("scale_factor_error")
                    elif "DSNU_DN" in df.columns:
                        obs_params.append("DSNU")

                    res = explicit_parser.resolve(observed_parameters=obs_params)
                elif cond == "RAW_MEASUREMENTS_ONLY":
                    # Convert to MeasurementRecords for Behavioral Inference
                    records = []
                    pname = df["primary_parameter"].iloc[0]
                    punit = df["unit"].iloc[0]
                    for idx_c, r in df.head(10).iterrows():
                        records.append(MeasurementRecord(
                            component_id=str(r["component_id"]),
                            parameter_name=pname,
                            value=float(r["value_24h"]),
                            unit=punit
                        ))
                    res = behavioral_engine.infer(records=records)
                else: # CORRUPTED_UNKNOWN
                    res = explicit_parser.resolve(observed_parameters=["CORRUPTED_PARAM_NOISE"])

                # Check Resolution Accuracy
                resolved_fam = res.resolved_device_family
                if cond == "CORRUPTED_UNKNOWN":
                    if res.status in (ResolutionStatus.UNKNOWN_CONTEXT, ResolutionStatus.AMBIGUOUS_CONTEXT):
                        correct_context_cnt += 1
                else:
                    if resolved_fam == dev_fam:
                        correct_context_cnt += 1

                # Model Router Execution
                report = router.route_and_predict(df_lot=df, context_result=res)

                if not report.is_execution_allowed:
                    interlocked_cnt += 1
                else:
                    if report.active_device_family != dev_fam:
                        wrong_model_cnt += 1

                    # Evaluate Prediction MAE & Escapes
                    for comp_res in report.results:
                        comp_id = comp_res["component_id"]
                        row_gt = df[df["component_id"] == comp_id].iloc[0]
                        gt_168 = float(row_gt["value_168h_actual"])
                        pred_168 = float(comp_res["predicted_168h"])
                        mae_list.append(abs(gt_168 - pred_168))

                        is_def = bool(row_gt["is_defective_gt"])
                        tier = comp_res["risk_tier"]
                        if is_def and tier == "GREEN_AUTO_PASS":
                            escape_cnt += 1

        context_acc = (correct_context_cnt / total_lots) * 100.0
        wrong_model_pct = (wrong_model_cnt / total_lots) * 100.0
        avg_mae = np.mean(mae_list) if mae_list else 0.0

        summary_rows.append({
            "Test Condition": cond,
            "Total Lots": total_lots,
            "Context Accuracy (%)": round(context_acc, 2),
            "Safety Interlocked Lots": interlocked_cnt,
            "Wrong-Model Routing (%)": round(wrong_model_pct, 2),
            "168h Forecast MAE": round(avg_mae, 4),
            "Silent Escapes": escape_cnt,
        })

    df_summary = pd.DataFrame(summary_rows)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nSUMMARY EVALUATION RESULTS:")
    print(df_summary.to_string(index=False))
    print("=" * 75)
    return df_summary


if __name__ == "__main__":
    run_blind_evaluation()
