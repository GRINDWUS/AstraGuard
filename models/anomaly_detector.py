"""
AstraGuard 2.1 — Baseline & ML Model Pipeline (Phase 4 + Phase 5)
=================================================================
Implements the four-tier detection baseline progression:

  Baseline 1: Absolute specification threshold only (I_DDQ_24h > spec_max → RED)
  Baseline 2: Absolute threshold + Standard Z-score
  Baseline 3: Absolute threshold + Robust Z-score (Median + MAD)
  AstraGuard: Robust stats + temporal kinetics + XGBoost + policy engine

This layered baseline design answers the research question:
  "Does each additional intelligence layer actually improve defect detection?"

Dual ML Task:
  Task A: Classification — Is this component showing anomalous behaviour?
          Metrics: Recall, Precision, F1, PR-AUC, FNR, FPR
  Task B: Regression — What will the component's I_DDQ be at 168h?
          Metrics: MAE, RMSE, R²

IMPORTANT: ML recall/precision are observed experimental results at a chosen
decision threshold — NOT hardcoded design targets.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report, precision_recall_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from models.lstm_autoencoder import LSTMAnomalyDetector
import warnings
warnings.filterwarnings("ignore")


# ─── Feature Engineering ─────────────────────────────────────────────────────

def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + 1e-9)

def _standard_z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Stage A feature vector X from the checkpoint summary.
    Feature vector per component:
      [iddq_0h, iddq_24h, Δiddq, drift_velocity, drift_rel,
       robust_z_iddq_0h, robust_z_delta_iddq, modified_z_iddq_24h,
       v_offset_0h, v_offset_24h, Δvoffset, robust_z_voffset,
       snr_0h, snr_24h, Δsnr,
       t_die_24h, iddq_spec_ratio_24h,
       wafer_dist_from_center]
    """
    f = pd.DataFrame(index=df.index)
    f["iddq_0h"]         = df["iddq_0h"]
    f["iddq_24h"]        = df["iddq_24h"]
    f["delta_iddq"]      = df["delta_iddq"]
    f["drift_velocity"]  = df["delta_iddq"] / 24.0
    f["drift_rel"]       = df["delta_iddq"] / (df["iddq_0h"] + 1e-5)
    f["robust_z_iddq"]   = _robust_z(df["iddq_0h"])
    f["robust_z_delta"]  = _robust_z(df["delta_iddq"])
    f["std_z_iddq"]      = _standard_z(df["iddq_0h"])
    f["v_offset_0h"]     = df["v_offset_0h"]
    f["v_offset_24h"]    = df["v_offset_24h"]
    f["delta_voffset"]   = df["delta_v_offset"]
    f["robust_z_voffset"]= _robust_z(df["v_offset_0h"])
    f["snr_0h"]          = df["snr_0h"]
    f["snr_24h"]         = df["snr_24h"]
    f["delta_snr"]       = df["delta_snr"]
    f["robust_z_snr"]    = _robust_z(df["snr_0h"])
    if "t_die_24h" in df.columns:
        f["t_die_24h"]   = df["t_die_24h"]
    if "spec_max_iddq" in df.columns:
        f["spec_ratio_24h"] = df["iddq_24h"] / (df["spec_max_iddq"] + 1e-5)
    if "wafer_x" in df.columns and "wafer_y" in df.columns:
        f["wafer_dist"]  = np.sqrt(df["wafer_x"]**2 + df["wafer_y"]**2)
    return f


# ─── Four Baselines (Phase 4) ─────────────────────────────────────────────────

def baseline_1_absolute(df: pd.DataFrame) -> pd.Series:
    """Baseline 1: Absolute specification limit only."""
    return (df["iddq_24h"] > df.get("spec_max_iddq", 15.0)).astype(int)

def baseline_2_absolute_std_z(df: pd.DataFrame, z_thresh: float = 2.5) -> pd.Series:
    """Baseline 2: Absolute limit + Standard Z-score."""
    b1 = baseline_1_absolute(df)
    z  = _standard_z(df["iddq_0h"])
    return ((b1 == 1) | (z.abs() >= z_thresh)).astype(int)

def baseline_3_absolute_robust_z(df: pd.DataFrame, z_thresh: float = 2.5) -> pd.Series:
    """Baseline 3: Absolute limit + Robust Z-score (Median + MAD)."""
    b1 = baseline_1_absolute(df)
    z  = _robust_z(df["iddq_0h"])
    return ((b1 == 1) | (z.abs() >= z_thresh)).astype(int)

def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[0,0]))
    recall    = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fnr       = fn / (tp + fn + 1e-9)
    fpr       = fp / (fp + tn + 1e-9)
    return {
        "label": label, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Recall": round(recall, 4), "Precision": round(precision, 4),
        "F1": round(f1, 4), "FNR": round(fnr, 4), "FPR": round(fpr, 4),
    }


# ─── AstraGuard ML Model (Phase 5) ────────────────────────────────────────────

class AstraGuardMLEngine:
    """
    AstraGuard 3.0 Hybrid Reliability Intelligence Engine:
      - Task A: Supervised XGBoost Classifier (known defect probability)
      - Task B: Supervised XGBoost Regressor (168h I_DDQ degradation forecast)
      - Task C: Unsupervised Isolation Forest (Out-of-Distribution / Unknown anomaly score)
      - Fusion: Policy Engine (policy-3.0)
    """
    def __init__(self, policy_z_thresh: float = 2.5, policy_spec_ratio: float = 0.85):
        self.z_thresh       = policy_z_thresh
        self.spec_ratio     = policy_spec_ratio
        self.scaler         = StandardScaler()
        
        # Task A: Supervised XGBoost Classifier (known defects)
        self.classifier = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=11,       # ~92/8 class imbalance
            eval_metric="aucpr",       # Precision-Recall AUC loss
            use_label_encoder=False, random_state=42, n_jobs=-1,
        )
        
        # Task B: Supervised XGBoost Regressor (168h forecast)
        self.regressor = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, n_jobs=-1,
        )

        # Task C: Unsupervised Isolation Forest (OOD Anomaly Detector — Static Features)
        from sklearn.ensemble import IsolationForest
        self.iso_forest = IsolationForest(
            n_estimators=150, contamination=0.08, random_state=42, n_jobs=-1
        )

        # Task D: LSTM Autoencoder (OOD Anomaly Detector — Temporal Sequences, Eye 3B)
        self.lstm_detector = LSTMAnomalyDetector(units=32, epochs=30, percentile_threshold=95.0)
        
        self.is_trained = False
        self.feature_cols: list = []

    def fit(self, train_dfs: list, telemetry_dfs: list = None):
        """
        Train on a list of checkpoint DataFrames (one per training lot).
        Fits XGBoost + Isolation Forest + LSTM Autoencoder on healthy/nominal distributions.

        Args:
            train_dfs: List of checkpoint summary DataFrames (required).
            telemetry_dfs: List of hourly telemetry DataFrames (optional).
                           If provided, LSTM AE uses true temporal sequences.
                           If None, LSTM AE uses interpolated pseudo-sequences from checkpoints.
        """
        combined = pd.concat(train_dfs, ignore_index=True)
        X = build_features(combined)
        self.feature_cols = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        
        y_clf = combined["is_defective_gt"].values
        y_reg = combined["iddq_168h_actual"].values
        
        self.classifier.fit(X_scaled, y_clf)
        self.regressor.fit(X_scaled, y_reg)
        
        # Fit Isolation Forest on NOMINAL components only (Eye 3A — static features)
        X_nominal = X_scaled[y_clf == 0] if (y_clf == 0).sum() > 50 else X_scaled
        self.iso_forest.fit(X_nominal)

        # Fit LSTM AE on NOMINAL temporal sequences only (Eye 3B — temporal dynamics)
        nominal_chk = combined[combined["is_defective_gt"] == 0].reset_index(drop=True)
        if telemetry_dfs is not None:
            tel_combined = pd.concat(telemetry_dfs, ignore_index=True)
            nom_ids = set(nominal_chk["component_id"])
            tel_nominal = tel_combined[tel_combined["component_id"].isin(nom_ids)]
            seqs, _ = LSTMAnomalyDetector.build_sequences_from_telemetry(tel_nominal)
        else:
            seqs, _ = LSTMAnomalyDetector.build_sequences_from_checkpoint(nominal_chk)
        print(f"  Fitting LSTM AE on {len(seqs)} nominal sequences...")
        self.lstm_detector.fit(seqs)

        self.is_trained = True
        print(f"  Trained Hybrid Engine (XGB + IsoForest + LSTM AE) on {len(combined)} components.")

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Run full hybrid inference pipeline (Supervised + Unsupervised + Spatial + Policy).
        """
        assert self.is_trained, "Call fit() first."
        X = build_features(df)[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        prob       = self.classifier.predict_proba(X_scaled)[:, 1]
        pred_168h  = self.regressor.predict(X_scaled)
        
        # Eye 3A: Isolation Forest (static feature-space OOD)
        iso_preds  = self.iso_forest.predict(X_scaled)
        iso_scores = -self.iso_forest.decision_function(X_scaled)  # Higher = more anomalous

        # Eye 3B: LSTM AE (temporal sequence OOD)
        # Build pseudo-sequences from checkpoint data (fallback when no telemetry provided)
        lstm_seqs, lstm_ids = LSTMAnomalyDetector.build_sequences_from_checkpoint(df)
        lstm_scores_arr = self.lstm_detector.anomaly_scores(lstm_seqs)   # per component
        lstm_preds_arr  = self.lstm_detector.predict(lstm_seqs)           # -1 or +1
        # Map back to row order (component_id → score)
        lstm_score_map = dict(zip(lstm_ids, lstm_scores_arr))
        lstm_pred_map  = dict(zip(lstm_ids, lstm_preds_arr))
        lstm_scores = np.array([lstm_score_map.get(cid, 0.0) for cid in df["component_id"].values])
        lstm_preds  = np.array([lstm_pred_map.get(cid,  1)   for cid in df["component_id"].values])
        
        out = df[["component_id", "lot_id", "iddq_0h", "iddq_24h",
                  "delta_iddq", "spec_max_iddq", "failure_class",
                  "is_defective_gt", "failure_mode_gt"]].copy()
        out["predicted_iddq_168h"]   = np.round(pred_168h, 3)
        out["defect_probability"]    = np.round(prob, 4)
        out["iforest_anomaly_score"] = np.round(iso_scores, 4)
        out["lstm_ae_anomaly_score"] = np.round(lstm_scores, 6)
        out["lstm_ae_pred"]          = lstm_preds
        
        robust_z = _robust_z(df["iddq_0h"])
        robust_z_delta = _robust_z(df["delta_iddq"])
        pred_spec_ratio = pred_168h / (df["spec_max_iddq"].values + 1e-5)
        
        # ── Evidence Fusion & Policy Engine (policy-3.0) ──
        recommendations, reason_codes = [], []
        for i in range(len(out)):
            codes = []
            
            # Step 1: Instrument/Sensor Fault Check
            if out["failure_class"].iloc[i] == 5:
                codes.append("INSTRUMENT_CHANNEL_FROZEN")
                recommendations.append("INSTRUMENT_REVIEW")
                reason_codes.append(" | ".join(codes))
                continue

            if abs(robust_z.iloc[i]) >= self.z_thresh:
                codes.append("POPULATION_OUTLIER")
            if abs(robust_z_delta.iloc[i]) >= self.z_thresh:
                codes.append("ACCELERATING_KINETICS")
            if pred_spec_ratio[i] >= self.spec_ratio:
                codes.append("FORECAST_NEAR_SPEC_LIMIT")
            if prob[i] >= threshold:
                codes.append("HIGH_DEFECT_PROBABILITY")
            if iso_preds[i] == -1:
                codes.append("IFOREST_OOD_ANOMALY")
            if lstm_preds[i] == -1:
                codes.append("LSTM_TEMPORAL_ANOMALY")

            # Step 2: Novel Anomaly / OOD Check (either unsupervised detector flags it)
            ood_flagged = iso_preds[i] == -1 or lstm_preds[i] == -1
            is_novel_anomaly = (prob[i] < threshold) and (ood_flagged or abs(robust_z.iloc[i]) > 3.0)
            if is_novel_anomaly:
                codes.append("UNKNOWN_DEGRADATION_PATTERN")

            if not codes:
                codes.append("LOW_RISK")
            
            # Step 3: Multi-Eye Decision Tiers (4-Eye Fusion)
            if prob[i] >= threshold or pred_spec_ratio[i] >= self.spec_ratio:
                recommendations.append("RED_HIGH_RISK")
            elif is_novel_anomaly:
                recommendations.append("UNKNOWN_PATTERN_REVIEW")
            elif prob[i] >= 0.25 or abs(robust_z.iloc[i]) >= 1.6 \
                    or iso_preds[i] == -1 or lstm_preds[i] == -1:
                recommendations.append("YELLOW_REVIEW")
            else:
                recommendations.append("GREEN_NORMAL_CANDIDATE")
            reason_codes.append(" | ".join(codes))
        
        out["recommendation"]  = recommendations
        out["reason_codes"]    = reason_codes
        out["policy_version"]  = "policy-3.0"
        return out

    def evaluate(self, df_pred: pd.DataFrame, threshold: float = 0.5) -> dict:
        """Full evaluation including Precision-Recall AUC and regression metrics."""
        y_true = df_pred["is_defective_gt"].values
        y_prob = df_pred["defect_probability"].values
        y_pred = (y_prob >= threshold).astype(int)
        y_reg  = df_pred["predicted_iddq_168h"].values
        y_act  = df_pred["iddq_168h_actual"].values if "iddq_168h_actual" in df_pred.columns else None
        
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        
        clf_metrics = evaluate_binary(y_true, y_pred, label="AstraGuard_ML")
        clf_metrics["PR_AUC"] = round(pr_auc, 4)
        
        reg_metrics = {}
        if y_act is not None:
            reg_metrics = {
                "MAE":  round(mean_absolute_error(y_act, y_reg), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_act, y_reg)), 4),
                "R2":   round(r2_score(y_act, y_reg), 4),
            }
        
        return {"classification": clf_metrics, "regression": reg_metrics, "pr_curve": (prec, rec)}


def run_baseline_comparison(train_lots: list, eval_lot: pd.DataFrame, label: str = "EVAL") -> pd.DataFrame:
    """
    Runs all four detection tiers and returns a comparison DataFrame.
    Phase 4: Establishes baselines BEFORE ML, proving value of each intelligence layer.
    """
    y_true = eval_lot["is_defective_gt"].values
    rows = []
    
    rows.append(evaluate_binary(y_true, baseline_1_absolute(eval_lot).values, "B1_Absolute"))
    rows.append(evaluate_binary(y_true, baseline_2_absolute_std_z(eval_lot).values, "B2_Absolute+StdZ"))
    rows.append(evaluate_binary(y_true, baseline_3_absolute_robust_z(eval_lot).values, "B3_Absolute+RobustZ"))
    
    engine = AstraGuardMLEngine()
    engine.fit(train_lots)
    preds = engine.predict(eval_lot)
    y_ml  = (preds["recommendation"] == "RED_HIGH_RISK").astype(int).values
    ml_metrics = evaluate_binary(y_true, y_ml, "AstraGuard_ML+Policy")
    
    eval_result = engine.evaluate(preds)
    ml_metrics["PR_AUC"] = eval_result["classification"]["PR_AUC"]
    rows.append(ml_metrics)
    
    return pd.DataFrame(rows).set_index("label"), engine, preds