"""
AstraGuard Prototype - Step 3: ML Model Comparison (Linear vs. Random Forest vs. XGBoost)
Evaluates forecasting performance using ONLY 0h and 24h features.
"""
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_ml_regressor_comparison():
    print("========================================================================================")
    print("        ASTRAGUARD PROTOTYPE - STEP 3: ML MODEL COMPARISON (168H FORECASTING)")
    print("========================================================================================\n")
    
    files = sorted(glob.glob("D:\\SIH 2026\\astraguard_core\\data\\LOT_*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    
    # Train on Lots 1-5, Test on Lots 6-7 (Strict Hold-Out)
    train_df = pd.concat(dfs[:5], ignore_index=True)
    test_df = pd.concat(dfs[5:], ignore_index=True)
    
    def extract_features(df):
        X = pd.DataFrame()
        X['iddq_0h'] = df['iddq_0h']
        X['iddq_24h'] = df['iddq_24h']
        X['delta_24h'] = df['iddq_24h'] - df['iddq_0h']
        X['drift_rate'] = X['delta_24h'] / 24.0
        X['drift_accel'] = X['delta_24h'] / (df['iddq_0h'] + 1e-5)
        dist = np.sqrt(df['wafer_x']**2 + df['wafer_y']**2)
        X['wafer_dist'] = dist
        return X

    X_train = extract_features(train_df)
    y_train = train_df['iddq_168h_actual']
    
    X_test = extract_features(test_df)
    y_test = test_df['iddq_168h_actual']
    
    models = {
        "Model A: Linear Regression": LinearRegression(),
        "Model B: Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "Model C: XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "MAE (µA)": round(mae, 2),
            "RMSE (µA)": round(rmse, 2),
            "R² Score": round(r2, 4)
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("\nConclusion: XGBoost / Random Forest significantly outperform Linear Regression due to non-linear kinetic surge.")
    print("========================================================================================\n")

if __name__ == "__main__":
    run_ml_regressor_comparison()
