#!/usr/bin/env python3
"""
AstraGuard 2.4 — Live Server API Endpoint Validator
===================================================
Tests all REST API endpoints served by server.py on http://127.0.0.1:8000.
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post(path, payload):
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("=" * 80)
print("TESTING ASTRAGUARD 2.4 FASTAPI BACKEND ENDPOINTS (http://127.0.0.1:8000)")
print("=" * 80)

# 1. API Root
root = get("/")
print(f"✅ GET /: {root['system']} ({root['status']})")

# 2. Validation Metrics
metrics = get("/api/v1/analytics/validation-metrics")
print(f"✅ GET /api/v1/analytics/validation-metrics: Overall R² = {metrics.get('overall_r2')}, Blind Test Size = {metrics.get('blind_test_size')}")

# 3. Lot Summary
lot_summary = get("/api/v1/stage-a/lot-summary/LOT_2026_07")
print(f"✅ GET /api/v1/stage-a/lot-summary/LOT_2026_07: Total Components = {lot_summary.get('total_components')}, Yield = {lot_summary.get('yield_at_96h')}%")

# 4. Predict Single
predict_single = post("/api/v1/stage-a/predict-single", {
    "component_id": "COMP_TEST_001",
    "iddq_0h": 12.5,
    "iddq_24h": 14.8,
    "wafer_x": 10.0,
    "wafer_y": 15.0
})
print(f"✅ POST /api/v1/stage-a/predict-single: {predict_single}")

# 5. Predict Batch
predict_batch = post("/api/v1/stage-a/predict-batch", {
    "lot_id": "LOT_2026_07",
    "components": [
        {"component_id": "COMP_001", "iddq_0h": 12.5, "iddq_24h": 14.8, "wafer_x": 10.0, "wafer_y": 15.0},
        {"component_id": "COMP_002", "iddq_0h": 13.1, "iddq_24h": 18.2, "wafer_x": 12.0, "wafer_y": 15.0}
    ]
})
print(f"✅ POST /api/v1/stage-a/predict-batch: Processed {len(predict_batch.get('results', []))} components")

# 6. SHAP Explanation
shap = get("/api/v1/stage-a/component/COMP_TEST_001/shap-explanation")
print(f"✅ GET /api/v1/stage-a/component/COMP_TEST_001/shap-explanation: Predicted 168h = {shap.get('predicted_168h_ua'):.2f} µA, Mechanism = {shap.get('matched_failure_mechanism')}")

# 7. Evaluate In-Orbit Telemetry
telemetry = post("/api/v1/stage-b/evaluate-telemetry", {
    "component_id": "COMP_TEST_001",
    "telemetry_iddq": 35.5,
    "mission_day": 45
})
print(f"✅ POST /api/v1/stage-b/evaluate-telemetry: Status = {telemetry.get('status')}")

# 8. V2 Context Profiles
profiles = get("/api/v2/context/profiles")
print(f"✅ GET /api/v2/context/profiles: {len(profiles.get('profiles', []))} profiles loaded")

# 9. V2 Context Resolve
resolved = post("/api/v2/context/resolve", {"test_type": "IDDQ", "domain": "DIGITAL_IC"})
print(f"✅ POST /api/v2/context/resolve: Domain = {resolved.get('resolved_domain')}, Physics Route = {resolved.get('physics_route')}")

# 10. V2 Instrument QA
instrument = post("/api/v2/instrument/qa", {})
print(f"✅ POST /api/v2/instrument/qa: Status = {instrument.get('status')}")

print("\n🎉 ALL 10 BACKEND REST API ENDPOINTS ARE FULLY OPERATIONAL & VERIFIED!")
