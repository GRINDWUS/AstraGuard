"""
AstraGuard Core — Multi-Payload Adaptive Inference & Anomaly Engine
===================================================================
Strictly maps to ISRO PS #26170 requirements:
  - Multi-payload support: ADITYA_L1_PAPA, ASTROSAT_LAXPC_CZTI, EOS_08_EOIR, CARTOSAT_3_PAN
  - Module A: Dynamic Lot Outlier Detection (Robust Population Z-Score via Median + MAD).
  - Module B: Relative Ratio Regressor (0h + 24h -> Forecasts 168h Normalized Degradation Multiplier).
  - Specification Engine: Evaluates Absolute Max Limit, Population Drift, and Kinetic Velocity.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple

class AstraGuardPredictorFast:
    def __init__(self, failure_threshold_168h: float = 45.0):
        self.failure_threshold_168h = failure_threshold_168h
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering with Device Specification Awareness & Robust Z-Scores.
        """
        feats = pd.DataFrame()
        feats["iddq_0h"] = df["iddq_0h"]
        feats["iddq_24h"] = df["iddq_24h"]
        
        # Spec-Normalized Currents
        spec_max = df.get("spec_max_iddq", pd.Series(50.0, index=df.index))
        feats["iddq_0h_spec_ratio"] = df["iddq_0h"] / (spec_max + 1e-5)
        feats["iddq_24h_spec_ratio"] = df["iddq_24h"] / (spec_max + 1e-5)
        
        # Temporal Kinetics & Degradation Ratios
        delta_24h = df["iddq_24h"] - df["iddq_0h"]
        feats["delta_24h"] = delta_24h
        feats["drift_velocity_24h"] = delta_24h / 24.0
        feats["drift_acceleration_rel"] = delta_24h / (df["iddq_0h"] + 1e-5)
        
        # Environmental / Operational Context
        feats["operating_voltage_v"] = df.get("operating_voltage_v", pd.Series(5.0, index=df.index))
        feats["test_temperature_c"] = df.get("test_temperature_c", pd.Series(25.0, index=df.index))
        feats["clock_freq_mhz"] = df.get("clock_freq_mhz", pd.Series(20.0, index=df.index))
        
        # Spatial Wafer Coordinates
        wafer_x = df.get("wafer_x", pd.Series(0.0, index=df.index))
        wafer_y = df.get("wafer_y", pd.Series(0.0, index=df.index))
        feats["dist_from_center"] = np.sqrt(wafer_x**2 + wafer_y**2)
        
        # Module A: Standard Z-Score
        lot_mean_0h = df["iddq_0h"].mean()
        lot_std_0h = df["iddq_0h"].std() + 1e-5
        feats["spatial_z_score"] = (df["iddq_0h"] - lot_mean_0h) / lot_std_0h
        
        # Robust Population Z-Score (Median + MAD: 1.4826 * MAD)
        lot_median_0h = df["iddq_0h"].median()
        lot_mad_0h = (df["iddq_0h"] - lot_median_0h).abs().median() + 1e-5
        feats["robust_z_score"] = (df["iddq_0h"] - lot_median_0h) / (1.4826 * lot_mad_0h)
        
        # Dynamic 24h Population Drift Outlier Score (Standard & Robust)
        lot_mean_delta = delta_24h.mean()
        lot_std_delta = delta_24h.std() + 1e-5
        feats["drift_z_score"] = (delta_24h - lot_mean_delta) / lot_std_delta
        
        lot_median_delta = delta_24h.median()
        lot_mad_delta = (delta_24h - lot_median_delta).abs().median() + 1e-5
        feats["robust_drift_z_score"] = (delta_24h - lot_median_delta) / (1.4826 * lot_mad_delta)
        
        # Clean up NaNs / Infs for robust ML training & inference
        feats = feats.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return feats

    def fit(self, train_df: pd.DataFrame):
        """Train Module B Regressor using 0h and 24h features to predict 168h multiplier ratio."""
        X_train = self._extract_features(train_df)
        if "iddq_168h_actual" in train_df.columns:
            y_actual = train_df["iddq_168h_actual"]
        elif "iddq_168h" in train_df.columns:
            y_actual = train_df["iddq_168h"]
        else:
            y_actual = train_df["iddq_24h"] * 1.05
            
        y_actual = y_actual.fillna(train_df["iddq_24h"]).replace([np.inf, -np.inf], 0.0)
        # Target: Relative growth ratio relative to 24h measurement
        y_ratio = y_actual / (train_df["iddq_24h"] + 1e-5)
        y_ratio = y_ratio.fillna(1.0).replace([np.inf, -np.inf], 1.0)
        
        self.xgb_model.fit(X_train, y_ratio)
        self.is_trained = True

    def compute_module_a_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Module A: Outlier Detection System (Robust Median/MAD)."""
        feats = self._extract_features(df)
        outlier_df = pd.DataFrame(index=df.index)
        outlier_df["component_id"] = df["component_id"]
        outlier_df["spatial_z_score"] = np.round(feats["robust_z_score"], 2)
        outlier_df["drift_z_score"] = np.round(feats["robust_drift_z_score"], 2)
        outlier_df["is_population_anomaly"] = (abs(feats["robust_z_score"]) > 2.5) | (abs(feats["robust_drift_z_score"]) > 2.5)
        return outlier_df

    def predict_lot(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Full Multi-Factor Inference Pipeline (Specification + Outlier + Forecast)."""
        if not self.is_trained:
            raise ValueError("Model is not trained! Call fit() first.")
            
        X_test = self._extract_features(test_df)
        pred_ratios = self.xgb_model.predict(X_test)
        pred_168h = test_df["iddq_24h"] * pred_ratios
        
        result_df = test_df.copy()
        result_df["predicted_168h_iddq"] = np.round(pred_168h, 2)
        result_df["spatial_z_score"] = np.round(X_test["spatial_z_score"], 2)
        result_df["robust_z_score"] = np.round(X_test["robust_z_score"], 2)
        result_df["drift_z_score"] = np.round(X_test["drift_z_score"], 2)
        result_df["robust_drift_z_score"] = np.round(X_test["robust_drift_z_score"], 2)
        result_df["delta_24h"] = np.round(X_test["delta_24h"], 2)
        
        # Calculate Safety Slope: (Predicted_168h - 24h_IDDQ) / (168 - 24)
        safety_slope = (pred_168h - test_df["iddq_24h"]) / 144.0
        result_df["safety_slope_uA_per_hr"] = np.round(safety_slope, 4)
        
        from sklearn.ensemble import IsolationForest
        if len(X_test) >= 10:
            iforest = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
            iforest.fit(X_test)
            iforest_preds = iforest.predict(X_test)
            result_df["iforest_anomaly"] = iforest_preds == -1
        else:
            result_df["iforest_anomaly"] = False

        tiers = []
        rationales = []
        
        for idx, row in result_df.iterrows():
            pred = row["predicted_168h_iddq"]
            spec_max = row.get("spec_max_iddq", 50.0)
            robust_z = abs(row["robust_z_score"])
            robust_drift_z = abs(row["robust_drift_z_score"])
            slope = row["safety_slope_uA_per_hr"]
            pred_ratio = pred / (spec_max + 1e-5)
            is_iforest_ood = row["iforest_anomaly"]
            
            reasons = []
            if pred >= spec_max:
                reasons.append(f"Predicted leakage ({pred:.1f}µA) exceeds Spec Max ({spec_max:.1f}µA)")
            elif pred_ratio >= 0.85:
                reasons.append(f"Predicted 168h drift exceeds 85% of Spec Limit ({pred_ratio*100:.1f}%)")
            
            if slope > 0.05:
                reasons.append(f"Excessive degradation slope ({slope:.4f} µA/h)")

            if robust_z >= 2.5:
                reasons.append(f"Severe baseline population outlier (Z_robust = {robust_z:.2f}σ > 2.5σ)")
            elif robust_z >= 1.6:
                reasons.append(f"Moderate baseline population deviation (Z_robust = {robust_z:.2f}σ > 1.6σ)")
                
            if robust_drift_z >= 2.5:
                reasons.append(f"Accelerating 24h kinetic drift velocity outlier (Z_drift = {robust_drift_z:.2f}σ)")
            elif robust_drift_z >= 1.6:
                reasons.append(f"Elevated 24h kinetic drift rate")

            if is_iforest_ood:
                reasons.append("Eye 3: High-dimensional OOD anomaly pattern")
            
            if pred_ratio >= 0.85 or pred >= spec_max or robust_z >= 2.5 or robust_drift_z >= 2.5 or slope > 0.05:
                tiers.append("RED_EARLY_REJECT")
                rationales.append(" | ".join(reasons) if reasons else "High-risk kinetic drift anomaly")
            elif pred_ratio >= 0.65 or robust_z >= 1.6 or robust_drift_z >= 1.6 or is_iforest_ood or slope > 0.02:
                tiers.append("YELLOW_EXTENDED_TEST")
                rationales.append(" | ".join(reasons) if reasons else "Marginal drift / OOD pattern — Assigned to extended burn-in")
            else:
                tiers.append("GREEN_AUTO_PASS")
                rationales.append("Nominal population kinetics — Qualified for 24h Early Release")
                
        result_df["risk_tier"] = tiers
        result_df["decision_rationale"] = rationales
        return result_df

# Backward compatibility alias
AstraGuardPredictor = AstraGuardPredictorFast

