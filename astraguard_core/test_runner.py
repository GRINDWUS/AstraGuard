"""
AstraGuard Interactive CLI & Test Suite
Allows instant testing of any Lot CSV file or custom component reading.
"""
import sys
import os
import pandas as pd
import numpy as np
from astraguard_core.predictor_fast import AstraGuardPredictorFast
from src.astraguard_lifecycle_engine import AstraGuardLifecycleEngine

def run_single_component_interactive(iddq_0h: float, iddq_24h: float, wafer_x: float = 0.0, wafer_y: float = 0.0):
    print("\n==========================================================================")
    print("           ASTRAGUARD REAL-TIME COMPONENT TEST (SINGLE COMPONENT)")
    print("==========================================================================")
    print(f"Inputs  : IDDQ @ 0h = {iddq_0h:.2f} µA | IDDQ @ 24h = {iddq_24h:.2f} µA | Wafer (X,Y) = ({wafer_x}, {wafer_y})")
    
    # Train predictor on LOT 1-5
    train_df = pd.concat([pd.read_csv(f"D:\\SIH 2026\\astraguard_core\\data\\LOT_2026_0{i}.csv") for i in range(1, 6)])
    predictor = AstraGuardPredictorFast(failure_threshold_168h=45.0)
    predictor.fit(train_df)
    
    single_df = pd.DataFrame([{
        "lot_id": "TEST_LOT",
        "component_id": "TEST_COMP_001",
        "wafer_x": wafer_x,
        "wafer_y": wafer_y,
        "iddq_0h": iddq_0h,
        "iddq_24h": iddq_24h,
        "iddq_96h": 0.0,
        "iddq_168h_actual": 0.0,
        "is_defective_gt": 0,
        "failure_mode_gt": "UNKNOWN"
    }])
    
    res = predictor.predict_lot(single_df).iloc[0]
    
    pred_168 = res["predicted_168h_iddq"]
    tier = res["risk_tier"]
    
    print("\n--- STAGE A PRE-LAUNCH EVALUATION ---")
    print(f"  • Predicted 168h IDDQ Leakage : {pred_168:.2f} µA")
    print(f"  • Spatial Z-Score Anomaly     : {res['spatial_z_score']:.2f}")
    print(f"  • 24h Drift Delta             : {res['delta_24h']:.2f} µA")
    
    if tier == "GREEN_AUTO_PASS":
        print("  • Assigned Risk Tier          : 🟢 GREEN_AUTO_PASS")
        print("  • Operational Decision        : Pass immediately at 24h. Qualify for Space Flight!")
    elif tier == "YELLOW_EXTENDED_TEST":
        print("  • Assigned Risk Tier          : 🟡 YELLOW_EXTENDED_TEST")
        print("  • Operational Decision        : Ambiguous drift. Route component to 48h extra test (72h total).")
    else:
        print("  • Assigned Risk Tier          : 🔴 RED_EARLY_REJECT")
        print("  • Operational Decision        : Critical failure predicted! Scrap component at hour 24.")
        
    print("==========================================================================\n")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        iddq_0 = float(sys.argv[1])
        iddq_24 = float(sys.argv[2])
        wx = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0
        wy = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.0
        run_single_component_interactive(iddq_0, iddq_24, wx, wy)
    else:
        # Default test cases
        print("--- RUNNING SAMPLE COMPONENT TEST CASES ---")
        run_single_component_interactive(iddq_0h=12.5, iddq_24h=13.1)  # Healthy
        run_single_component_interactive(iddq_0h=11.5, iddq_24h=24.8)  # Thermal Runaway Defect
