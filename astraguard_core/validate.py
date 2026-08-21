"""
AstraGuard Core - Step 6: Leave-One-Lot-Out Cross Validation Pipeline (Fast Version)
"""
import os
import glob
import numpy as np
import pandas as pd
from astraguard_core.predictor_fast import AstraGuardPredictorFast

def evaluate_leave_one_lot_out() -> pd.DataFrame:
    all_files = glob.glob("D:\\SIH 2026\\astraguard_core\\data\\LOT_*.csv")
    if not all_files:
        raise FileNotFoundError("No LOT CSV files found in data directory!")
        
    lot_datasets = {os.path.basename(f).replace(".csv", ""): pd.read_csv(f) for f in all_files}
    lot_names = list(lot_datasets.keys())
    
    validation_results = []
    
    for test_lot in lot_names:
        train_lots = [df for name, df in lot_datasets.items() if name != test_lot]
        train_df = pd.concat(train_lots, ignore_index=True)
        test_df = lot_datasets[test_lot]
        
        predictor = AstraGuardPredictorFast(failure_threshold_168h=45.0)
        predictor.fit(train_df)
        pred_df = predictor.predict_lot(test_df)
        
        # 1. Regression Metrics
        y_true_168 = pred_df["iddq_168h_actual"]
        y_pred_168 = pred_df["predicted_168h_iddq"]
        mae = np.mean(np.abs(y_true_168 - y_pred_168))
        rmse = np.sqrt(np.mean((y_true_168 - y_pred_168)**2))
        
        # 2. Classification Metrics (Ground Truth Defective if actual 168h > 45.0 µA)
        gt_defective = (pred_df["iddq_168h_actual"] > 45.0) | (pred_df["is_defective_gt"] == 1)
        pred_defective = pred_df["risk_tier"] == "RED_EARLY_REJECT"
        
        tp = np.sum((gt_defective == True) & (pred_defective == True))
        fp = np.sum((gt_defective == False) & (pred_defective == True))
        fn = np.sum((gt_defective == True) & (pred_defective == False))
        tn = np.sum((gt_defective == False) & (pred_defective == False))
        
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 100.0
        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 100.0
        fnr = (fn / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
        
        # 3. Chamber Hours Saved Calculation
        green_cnt = np.sum(pred_df["risk_tier"] == "GREEN_AUTO_PASS")
        red_cnt = np.sum(pred_df["risk_tier"] == "RED_EARLY_REJECT")
        yellow_cnt = np.sum(pred_df["risk_tier"] == "YELLOW_EXTENDED_TEST")
        
        baseline_chamber_hours = len(pred_df) * 168
        astraguard_chamber_hours = (green_cnt * 24) + (red_cnt * 24) + (yellow_cnt * 72)
        hours_saved_pct = ((baseline_chamber_hours - astraguard_chamber_hours) / baseline_chamber_hours) * 100.0
        
        validation_results.append({
            "test_lot": test_lot,
            "mae_168h_ua": round(mae, 2),
            "rmse_168h_ua": round(rmse, 2),
            "recall_pct": round(recall, 2),
            "precision_pct": round(precision, 2),
            "fnr_escapes_pct": round(fnr, 4),
            "fpr_scrap_pct": round(fpr, 2),
            "chamber_hours_saved_pct": round(hours_saved_pct, 2)
        })
        
    val_df = pd.DataFrame(validation_results)
    return val_df

if __name__ == "__main__":
    print("=== RUNNING LEAVE-ONE-LOT-OUT CROSS VALIDATION (FAST ENGINE) ===")
    results_df = evaluate_leave_one_lot_out()
    print(results_df.to_string(index=False))
    print("\n--- OVERALL EXPERIMENTAL BENCHMARK RESULTS ---")
    print(f"Mean MAE (168h IDDQ Forecast) : {results_df['mae_168h_ua'].mean():.2f} µA")
    print(f"Mean RMSE                      : {results_df['rmse_168h_ua'].mean():.2f} µA")
    print(f"Mean Recall (Sensitivity)      : {results_df['recall_pct'].mean():.2f}%")
    print(f"Mean False Negative Rate (FNR) : {results_df['fnr_escapes_pct'].mean():.4f}%")
    print(f"Mean False Positive Rate (FPR) : {results_df['fpr_scrap_pct'].mean():.2f}%")
    print(f"Actual Chamber Hours Saved     : {results_df['chamber_hours_saved_pct'].mean():.2f}%")
