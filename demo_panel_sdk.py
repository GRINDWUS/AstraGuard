"""
Panel Demonstration Script: Testing AstraGuard SDK Installation & Stream
========================================================================
Run this script to prove SDK installation, connectivity, and real-time streaming to the panel.
"""

from astraguard_sdk import AstraGuardATESDK

def main():
    print("=================================================================")
    print("   ASTRAGUARD 2.0 SDK LIVE PANEL DEMONSTRATION                   ")
    print("=================================================================")
    
    # 1. Initialize SDK client
    sdk = AstraGuardATESDK(base_url="http://127.0.0.1:8000", instrument_id="ISRO_ATE_DEMO_01")
    
    # 2. Test Connection
    print("\n[Step 1] Checking active connection to AstraGuard backend server...")
    if not sdk.check_connection():
        print("❌ Error: Cannot connect to AstraGuard backend server at http://127.0.0.1:8000!")
        print("   Please start the backend server first by running: python server.py")
        return
    print("✅ Success: Connection established with AstraGuard Engine.")

    # 3. Stream Sample Lot Components via SDK
    csv_path = "D:\\SIH 2026\\astraguard_core\\data\\LOT_2026_07.csv"
    print(f"\n[Step 2] Ingesting ATE Lot Data line-by-line using AstraGuard SDK...")
    print(f"   Source File: {csv_path}")
    print("   Streaming components to ML engine...\n")
    print(f"{'COMPONENT ID':<26} | {'0h IDDQ':<8} | {'24h IDDQ':<9} | {'PREDICTED 168h':<14} | {'RISK TIER'}")
    print("-" * 80)
    
    for event in sdk.stream_lot_csv(csv_path, interval_sec=0.2, max_records=8):
        comp = event.get("component_id", "N/A")
        iddq0 = event.get("iddq_0h", 0.0)
        iddq24 = event.get("iddq_24h", 0.0)
        pred = event.get("predicted_168h_iddq_ua", "N/A")
        tier = event.get("risk_tier", "N/A")
        
        tier_symbol = "🟢" if "GREEN" in tier else "🟡" if "YELLOW" in tier else "🔴"
        print(f"{comp:<26} | {iddq0:>6.2f} µA | {iddq24:>7.2f} µA | {pred:>12.2f} µA | {tier_symbol} {tier}")

    print("-" * 80)
    print("\n✅ Live SDK Panel Verification Complete!")

if __name__ == "__main__":
    main()
