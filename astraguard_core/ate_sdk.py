"""
AstraGuard 2.0 ATE SDK Client Module
Provides clean streaming abstraction for Automated Test Equipment (ATE).
"""
import time
import requests
import json
from typing import Dict, Any, Generator, List

class AstraGuardATESDK:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", instrument_id: str = "ISRO_ATE_CHAMBER_01", lot_id: str = "LOT_2026_07"):
        self.base_url = base_url
        self.instrument_id = instrument_id
        self.lot_id = lot_id

    def stream_ate_measurements(self, df_components: pd.DataFrame, interval_sec: float = 0.5) -> Generator[Dict[str, Any], None, None]:
        """Streams ATE measurement events to AstraGuard API mimicking live chamber testing."""
        for idx, row in df_components.iterrows():
            payload = {
                "component_id": str(row["component_id"]),
                "iddq_0h": float(row["iddq_0h"]),
                "iddq_24h": float(row["iddq_24h"]),
                "wafer_x": float(row["wafer_x"]),
                "wafer_y": float(row["wafer_y"])
            }
            
            try:
                res = requests.post(f"{self.base_url}/api/v1/stage-a/predict-single", json=payload)
                if res.status_code == 200:
                    yield res.json()
                else:
                    yield {"error": f"HTTP {res.status_code}: {res.text}"}
            except Exception as e:
                yield {"error": str(e)}
                
            time.sleep(interval_sec)

# Quick Demo Execution
if __name__ == "__main__":
    import pandas as pd
    print("=== ASTRAGUARD ATE SDK STREAMING DEMO ===")
    test_df = pd.read_csv("D:\\SIH 2026\\astraguard_core\\data\\LOT_2026_07.csv").head(5)
    
    sdk = AstraGuardATESDK(base_url="http://127.0.0.1:8000", instrument_id="ISRO_ATE_01", lot_id="LOT_2026_07")
    
    print(f"Connecting to AstraGuard Engine at {sdk.base_url}...")
    print(f"Instrument: {sdk.instrument_id} | Lot: {sdk.lot_id}\n")
    
    for event in sdk.stream_ate_measurements(test_df, interval_sec=0.2):
        comp = event.get("component_id", "N/A")
        pred = event.get("predicted_168h_iddq_ua", "N/A")
        tier = event.get("risk_tier", "N/A")
        print(f"⚡ [ATE EVENT] Comp: {comp:<22} | Predicted 168h IDDQ: {pred:>6} µA | Tier: {tier}")
