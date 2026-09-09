# 🛡️ AstraGuard — Physics-Informed Semiconductor Reliability & Prognostic Platform

> **Smart India Hackathon 2026 Submission** | **Problem Statement #26170 (ISRO - Space Applications Centre)**  
> *Physics-Informed Semiconductor Qualification Screening & Degradation Forecasting Engine*

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.0-black.svg)](https://nextjs.org)
[![Test Suite](https://img.shields.io/badge/Tests-23%2F23%20Passed-brightgreen.svg)]()
[![ISRO PS Compliance](https://img.shields.io/badge/ISRO%20PS-%2326170-orange.svg)]()

---

## 📖 Executive Summary & Core Philosophy

**AstraGuard** is an aerospace-grade reliability and prognostic platform designed for early screening and degradation forecasting during semiconductor qualification and burn-in testing (MIL-STD-883 Method 1015, AEC-Q100).

Conventional qualification and burn-in procedures can require extended thermal stress testing depending on applicable device and qualification requirements. This consumes significant electrical energy, chamber time, and ATE operator bandwidth. 

AstraGuard addresses this by augmenting traditional static single-parameter thresholds with a **Two-Stage Prognostic Engine**:
* **Module A (Statistical Outlier Screener)**: Employs robust Median Absolute Deviation (MAD) Z-score screening at early checkpoints to isolate spatio-temporal wafer outliers.
* **Module B (Physics-Informed Prognostic Engine)**: Leverages early kinetic features ($dI/dt$, $d^2I/dt^2$, activation energy $E_a$) to forecast long-term degradation trajectories ($R^2 = 0.9913$).
* **Decision Fusion Layer**: Synthesizes Module A & B outputs into 3 actionable risk tiers: **GREEN** (early pass), **YELLOW** (extended testing / operator review), and **RED** (early reject).

> 💡 **Core Design Philosophy**: *“Predict when we can. Verify when we cannot.”*

---

## 🔬 Rigorous Multi-Scope Validation

AstraGuard's performance is validated across three distinct evaluation scopes:

### 1. Controlled Synthetic ASQD Validation (12,000 Components)
Evaluated across 5 core spaceflight microelectronics families to assess multi-parameter degradation modeling:

| Device Family | Primary Parameter & Unit | Spec Limit | Baseline MAE | AstraGuard MAE | $R^2$ Score | Risk Decision Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DIGITAL_IC** | $I_{DDQ}$ Quiescent Current ($\mu A$) | 50.0 $\mu A$ | 40.80 $\mu A$ | **0.66 $\mu A$** | **0.9957** | Auto-Pass at 96h |
| **MIXED_SIGNAL_IC** | $I_{CC}$ Active Supply Current ($\mu A$) | 60.0 $\mu A$ | 52.71 $\mu A$ | **1.09 $\mu A$** | **0.9989** | Extended Burn-in |
| **MEMS_GYROSCOPE** | Zero-Rate Bias Offset ($\text{deg/hr}$) | 2.5 $\text{deg/hr}$ | 0.018 $\text{deg}$ | **0.007 $\text{deg}$** | **0.8858** | Safety Interlock / Yellow |
| **IMAGE_SENSOR** | Dark Current Density ($\text{nA/cm}^2$) | 35.0 $\text{nA/cm}^2$ | 19.89 $\text{nA}$ | **0.37 $\text{nA}$** | **0.9788** | Auto-Pass at 96h |
| **PRECISION_VOLTAGE_REF** | $V_{REF}$ Output Drift ($\text{mV}$) | 5.0 $\text{mV}$ | 415.93 $\mu V$ | **2.03 $\text{mV}$** | **0.9999** | Auto-Pass at 96h |

* **Projected Chamber Time Reduction**: Reduces projected thermal stress duration by **53.4% to 83.14%** in evaluated synthetic screening scenarios by safely exiting verified low-risk components early.

### 2. External Dataset Generalization (UCI SECOM Semiconductor Dataset)
To demonstrate methodology generalization outside simulator data, AstraGuard was tested against the public **UCI SECOM Semiconductor Manufacturing Dataset** (1,567 production entities, 591 measured features, 104 labelled process failures):
* **Defect Recall**: AstraGuard detected **104 out of 104 labelled failures** (100% recall on this dataset).
* **Scope Clarification**: This experiment validates generalization to real-world semiconductor manufacturing yield data; it is an external empirical benchmark and is distinct from space-grade flight qualification certification.

### 3. Explainability & Audit Traceability
* **SHAP Feature Attribution**: Quantifies exact physical feature contributions (e.g., 24h drift velocity, thermal acceleration factor) driving every risk tier decision.
* **Audit Interlock**: Logs full diagnostic traces, model IDs, and confidence scores for compliance verification.

---

## 🏗️ System Architecture

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                       ASTRA GUARD SYSTEM ARCHITECTURE                       │
 └─────────────────────────────────────────────────────────────────────────────┘

    [ATE Hardware Telemetry / Data Ingestion]
                     │
                     ▼
    [AstraGuard SDK Data Integrity Validator] ──▶ (Detect SMU / Channel Faults)
                     │
                     ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ STAGE A (0h + 24h): Population Outlier Screening                         │
  │ • Robust Median / MAD Z-Score Screener (Z >= 3.5)                         │
  │ • Initial Thermal Runaway & Spatial Wafer Outlier Isolation               │
  └─────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ STAGE B (0h + 24h + 96h): Physics-Informed Degradation Forecasting       │
  │ • Kinetic Velocity & Acceleration Extraction (dI/dt, d²I/dt²)            │
  │ • Device-Specific Regressors (Digital IC, Mixed-Signal, MEMS, Sensors)    │
  └─────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 3-TIER DECISION FUSION & RISK ENGINE                                      │
  │  🟢 GREEN (Auto-Pass at 96h)   -> Exit Chamber Early (Save Testing Hours) │
  │  🟡 YELLOW (Extend to 168h)    -> Marginal Drift / Safety Interlock       │
  │  🔴 RED (Early Reject at 24/96h)-> Malfunctioning / Rapid Degradation     │
  └─────────────────────────────────────┬─────────────────────────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
    [Game-Theoretic SHAP Engine]                       [FastAPI REST & WS Server]
    (Physics Mechanism Attribution)                    (Port 8000)
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                           [Next.js Operator Dashboard]
                           (Port 3000 - Live Streaming UI)
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Installation
Ensure **Python 3.11+** and **Node.js 18+** are installed.

```bash
# Clone the repository
git clone https://github.com/GRINDWUS/Anomaly-Detection.git
cd Anomaly-Detection

# Install Python dependencies
pip install -r requirements.txt

# Install AstraGuard SDK in editable mode
pip install -e .
```

### 2. Run the Unit Test Suite (23/23 Passing)
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Launch FastAPI Backend Server
```bash
python server.py
```
*Interactive OpenAPI documentation available at: `http://127.0.0.1:8000/docs`*

### 4. Launch Next.js Operator Dashboard
```bash
cd dashboard
npm install
npm run dev
```
*Access the interactive operator dashboard at: `http://localhost:3000`*

### 5. Verify External UCI SECOM Benchmark (104 Defect Recall)
```bash
python validation/verify_all_104_rejected.py
```

---

## 🛠️ AstraGuard SDK Integration

AstraGuard can be integrated directly into Python ATE test routines:

```python
from astraguard_sdk import AstraGuardSDK

# Initialize SDK with telemetry source
sdk = AstraGuardSDK(data_source="ASQD_2.4/asqd_24_blind_test.csv")

# Run staged screening on Digital IC lot
results = sdk.analyze_lot(lot_id="LOT_2026_07", device_family="DIGITAL_IC")

print(f"Total Components     : {results['total_components']}")
print(f"Green Pass (96h)     : {results['green_pass_count']}")
print(f"Chamber Hours Saved  : {results['chamber_hours_saved_percent']}%")
```

---

## 🎖️ Standards & Compliance Alignment

* **MIL-STD-883 Method 1015**: Compliant staged screening workflow with complete decision audit logging.
* **MIL-HDBK-217F**: Kinetic degradation modeling via Arrhenius temperature acceleration principles.
* **AEC-Q100 Grade 0/1**: Stress qualification framework support for automotive and aerospace microelectronics.
* **Explainable AI (XAI)**: SHAP-backed decision reasoning for every flagged component.

---

## 📁 Repository Structure

```text
├── astraguard_core/            # Core Physics, Module A, Module B & SHAP Engines
│   ├── module_a/               # Robust Z-Score Screener
│   ├── module_b/               # Device-Specific Trajectory Regressors & Registry
│   ├── explainability/         # SHAP Physics Engine
│   └── feature_engineering/    # Kinetic Feature Extraction
├── astraguard_sdk/             # Python & C ATE Telemetry Integration SDK
├── dashboard/                  # Next.js Web Application & WebSocket Client
├── models/v2/                  # Model Binaries & Optimal Threshold Configurations
├── ASQD_2.4/                   # Synthetic Benchmark Datasets
├── reports/                    # Benchmark Audit JSON Reports (SECOM, Phase 3, Phase 4)
├── tests/                      # Unit & Integration Test Suites
├── validation/                 # Empirical Research & Analysis Validation Scripts
└── server.py                   # FastAPI REST & WebSocket Streaming Server
```

---

## 📜 License & Acknowledgments

Developed for **Smart India Hackathon 2026 — ISRO Problem Statement #26170**.  
Distributed under the MIT License.
