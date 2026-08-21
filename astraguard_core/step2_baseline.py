"""
AstraGuard Prototype - Step 2: Non-AI Heuristic Baseline Evaluator
Evaluates a simple static limit + drift threshold model to establish the benchmark.
"""
import glob
import pandas as pd
import numpy as np

def run_baseline_experiment(static_limit_24h: float = 20.0, drift_limit_24h: float = 5.0):
    print("========================================================================================")
    print("            ASTRAGUARD PROTOTYPE - STEP 2: NON-AI HEURISTIC BASELINE BENCHMARK")
    print("========================================================================================\n")
    
    files = sorted(glob.glob("D:\\SIH 2026\\astraguard_core\\data\\LOT_*.csv"))
    full_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    # Non-AI Rule Logic
    # IF IDDQ_24h > 20.0 µA -> RED
    # ELSE IF (IDDQ_24h - IDDQ_0h) > 5.0 µA -> YELLOW
    # ELSE -> GREEN
    
    iddq_0 = full_df['iddq_0h'].values
    iddq_24 = full_df['iddq_24h'].values
    delta_24 = iddq_24 - iddq_0
    
    predictions = []
    for i in range(len(full_df)):
        if iddq_24[i] > static_limit_24h:
            predictions.append("RED")
        elif delta_24[i] > drift_limit_24h:
            predictions.append("YELLOW")
        else:
            predictions.append("GREEN")
            
    full_df['baseline_pred'] = predictions
    
    # Evaluation against Ground Truth Defective (actual 168h > 45.0 or defective_gt == 1)
    gt_defective = (full_df['iddq_168h_actual'] > 45.0) | (full_df['is_defective_gt'] == 1)
    pred_defective = full_df['baseline_pred'].isin(['RED', 'YELLOW'])
    
    tp = np.sum((gt_defective == True) & (pred_defective == True))
    fp = np.sum((gt_defective == False) & (pred_defective == True))
    fn = np.sum((gt_defective == True) & (pred_defective == False))
    tn = np.sum((gt_defective == False) & (pred_defective == False))
    
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    fnr = (fn / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("Non-AI Baseline Rule Configuration:")
    print(f"  • Static Limit @ 24h : > {static_limit_24h} µA ➔ RED")
    print(f"  • Drift Threshold    : > {drift_limit_24h} µA ➔ YELLOW")
    print("\nNon-AI Baseline Metrics:")
    print(f"  • True Positives (TP)        : {tp}")
    print(f"  • False Positives (FP)      : {fp} (Scrap)")
    print(f"  • False Negatives (FN)      : {fn} (Escapes!)")
    print(f"  • True Negatives (TN)       : {tn}")
    print(f"  • Recall (Sensitivity)      : {recall:.2f}%")
    print(f"  • Precision                 : {precision:.2f}%")
    print(f"  • False Negative Rate (FNR) : {fnr:.2f}% (CRITICAL: Space escapes)")
    print(f"  • False Positive Rate (FPR) : {fpr:.2f}%")
    print(f"  • F1 Score                  : {f1:.4f}")
    print("========================================================================================\n")

if __name__ == "__main__":
    run_baseline_experiment()
