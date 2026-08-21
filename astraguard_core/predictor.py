"""
AstraGuard Core - Step 3, 4 & 5: Baseline Evaluator, 168h Predictor & 3-Tier Risk Engine
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import xgboost as xgb
from typing import Dict, Any, Tuple

class AstraGuardPredictor:
    def __init__(self, failure_threshold_168h: float = 45.0):
        self.failure_threshold_168h = failure_threshold_168h
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            random_state=42
        )
        self.is_trained = False

    def _fit_spatial_residuals(self, df: pd.DataFrame) -> np.ndarray:
        """Applies GPR to isolate spatial wafer edge gradients."""
        coords = df[["wafer_x", "wafer_y"]].values
        iddq_0h = df["iddq_0h"].values
        
        kernel = RBF(length_scale=0.8) + WhiteKernel(noise_level=0.5)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
        gpr.fit(coords, iddq_0h)
        
        spatial_trend = gpr.predict(coords)
        residuals = iddq_0h - spatial_trend
        
        # Calculate Robust IQR Spatial Z-Score
        q75, q25 = np.percentile(residuals, [75, 25])
        iqr = q75 - q25
        median_res = np.median(residuals)
        spatial_z = (residuals - median_res) / (0.7413 * iqr if iqr > 0 else 1.0)
        return spatial_z

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts temporal drift and spatial features using ONLY 0h and 24h data."""
        feats = pd.DataFrame()
        feats["iddq_0h"] = df["iddq_0h"]
        feats["iddq_24h"] = df["iddq_24h"]
        feats["delta_24h"] = df["iddq_24h"] - df["iddq_0h"]
        feats["drift_rate_24h"] = feats["delta_24h"] / 24.0
        feats["drift_acceleration"] = feats["delta_24h"] / (df["iddq_0h"] + 1e-5)
        feats["spatial_z_score"] = self._fit_spatial_residuals(df)
        return feats

    def fit(self, train_df: pd.DataFrame):
        """Trains the XGBoost regressor on training lots to predict actual 168h IDDQ."""
        X_train = self._extract_features(train_df)
        y_train = train_df["iddq_168h_actual"]
        self.xgb_model.fit(X_train, y_train)
        self.is_trained = True

    def predict_lot(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts 168h IDDQ and assigns 3-Tier risk classifications."""
        if not self.is_trained:
            raise ValueError("Model is not trained! Call fit() first.")
            
        X_test = self._extract_features(test_df)
        pred_168h = self.xgb_model.predict(X_test)
        
        result_df = test_df.copy()
        result_df["predicted_168h_iddq"] = np.round(pred_168h, 2)
        result_df["spatial_z_score"] = np.round(X_test["spatial_z_score"], 2)
        result_df["delta_24h"] = np.round(X_test["delta_24h"], 2)
        
        # 3-Tier Asymmetric Risk Classification Logic
        tiers = []
        for idx, row in result_df.iterrows():
            pred = row["predicted_168h_iddq"]
            z_score = abs(row["spatial_z_score"])
            drift = row["delta_24h"]
            
            if pred > self.failure_threshold_168h or z_score > 3.2 or drift > 12.0:
                tiers.append("RED_EARLY_REJECT")
            elif pred > (self.failure_threshold_168h * 0.75) or z_score > 2.0 or drift > 5.0:
                tiers.append("YELLOW_EXTENDED_TEST")
            else:
                tiers.append("GREEN_AUTO_PASS")
                
        result_df["risk_tier"] = tiers
        return result_df

# Quick Test
if __name__ == "__main__":
    train_df = pd.read_csv("D:\\SIH 2026\\astraguard_core\\data\\LOT_2026_01.csv")
    test_df = pd.read_csv("D:\\SIH 2026\\astraguard_core\\data\\LOT_2026_06.csv")
    
    predictor = AstraGuardPredictor(failure_threshold_168h=45.0)
    predictor.fit(train_df)
    res = predictor.predict_lot(test_df)
    
    print("Execution Check successful!")
    print(res["risk_tier"].value_counts())
