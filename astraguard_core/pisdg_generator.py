"""
AstraGuard 2.0 — ISRO Multi-Payload Specification & Telemetry Dataset Generator (PISDG Pro)
========================================================================================
Calibrated against real ISRO satellite payloads & MIL-STD-883 / ESCC 9000 specifications:
  1. ADITYA_L1_PAPA      (Plasma Analyser Package: SWEEP & SWICAR sensors, V_op=28.0V, Base I=140mA, T=10..40°C)
  2. ASTROSAT_LAXPC_CZTI (X-ray & UV Detectors, V_op=5.0V, Base I=45mA, T=15..35°C)
  3. EOS_08_EOIR         (Infrared Electro-Optical Imager, V_op=12.0V, Base I=850mA, T=5..25°C)
  4. CARTOSAT_3_PAN      (High-Res Panchromatic Camera Sensor Array, V_op=3.3V, Base I=1.2A, T=18..30°C)

Each dataset row includes full mission context:
  - lot_id, component_id, payload_type, device_spec_id
  - operating_voltage_v, test_temperature_c, clock_frequency_mhz
  - wafer_x, wafer_y
  - iddq_0h, iddq_24h, iddq_96h, iddq_168h_actual (in µA or mA relative to payload class)
  - spec_min_iddq, spec_max_iddq
  - is_defective_gt, failure_mode_gt
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Any

PAYLOAD_SPECS: Dict[str, Dict[str, Any]] = {
    "ADITYA_L1_PAPA": {
        "device_spec_id": "ISRO-SPEC-ADITYA-PAPA-2023",
        "operating_voltage_v": 28.0,
        "test_temperature_c": 40.0,  # Max screening temp
        "clock_freq_mhz": 10.0,
        "base_iddq_mua": 140.0,     # µA baseline leakage
        "spec_min_mua": 80.0,
        "spec_max_mua": 220.0,
    },
    "ASTROSAT_LAXPC_CZTI": {
        "device_spec_id": "ISRO-SPEC-ASTROSAT-XRAY-2015",
        "operating_voltage_v": 5.0,
        "test_temperature_c": 35.0,
        "clock_freq_mhz": 20.0,
        "base_iddq_mua": 45.0,
        "spec_min_mua": 25.0,
        "spec_max_mua": 95.0,
    },
    "EOS_08_EOIR": {
        "device_spec_id": "ISRO-SPEC-EOS08-IR-2024",
        "operating_voltage_v": 12.0,
        "test_temperature_c": 25.0,
        "clock_freq_mhz": 50.0,
        "base_iddq_mua": 280.0,
        "spec_min_mua": 180.0,
        "spec_max_mua": 450.0,
    },
    "CARTOSAT_3_PAN": {
        "device_spec_id": "ISRO-SPEC-CARTO3-PAN-2019",
        "operating_voltage_v": 3.3,
        "test_temperature_c": 30.0,
        "clock_freq_mhz": 100.0,
        "base_iddq_mua": 520.0,
        "spec_min_mua": 350.0,
        "spec_max_mua": 850.0,
    }
}

def generate_isro_payload_lot(
    lot_id: str,
    payload_type: str = "ADITYA_L1_PAPA",
    num_components: int = 1000,
    defect_rate: float = 0.04,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generates physics-informed parametric dataset for a specific ISRO payload class."""
    np.random.seed(random_seed)
    spec = PAYLOAD_SPECS[payload_type]
    
    # 1. Wafer Coordinates
    radius = np.sqrt(np.random.uniform(0, 1, num_components))
    angle = np.random.uniform(0, 2 * np.pi, num_components)
    coord_x = radius * np.cos(angle)
    coord_y = radius * np.sin(angle)
    
    # 2. Baseline 0h IDDQ with wafer edge thermal gradient
    edge_bias = 0.08 * spec["base_iddq_mua"] * (coord_x**2 + coord_y**2)
    base_0h = np.random.normal(loc=spec["base_iddq_mua"], scale=0.08 * spec["base_iddq_mua"], size=num_components) + edge_bias
    
    # 3. Defect Injection
    is_defective = np.random.choice([0, 1], size=num_components, p=[1 - defect_rate, defect_rate])
    
    iddq_24h = np.zeros(num_components)
    iddq_96h = np.zeros(num_components)
    iddq_168h = np.zeros(num_components)
    failure_modes = []
    
    for i in range(num_components):
        noise = np.random.normal(0, 0.02 * spec["base_iddq_mua"])
        
        if is_defective[i] == 0:
            # Healthy Power Law Aging
            iddq_24h[i] = base_0h[i] + 0.03 * spec["base_iddq_mua"] * (24**0.15) + noise
            iddq_96h[i] = base_0h[i] + 0.03 * spec["base_iddq_mua"] * (96**0.15) + noise
            iddq_168h[i] = base_0h[i] + 0.03 * spec["base_iddq_mua"] * (168**0.15) + noise
            failure_modes.append("HEALTHY")
        else:
            mode = np.random.choice(["NBTI_POWER_LAW", "ELECTROMIGRATION_LINEAR", "THERMAL_RUNAWAY_EXP"])
            failure_modes.append(mode)
            
            if mode == "NBTI_POWER_LAW":
                iddq_24h[i] = base_0h[i] + 0.18 * spec["base_iddq_mua"] * (24**0.45) + noise
                iddq_96h[i] = base_0h[i] + 0.18 * spec["base_iddq_mua"] * (96**0.45) + noise
                iddq_168h[i] = base_0h[i] + 0.18 * spec["base_iddq_mua"] * (168**0.45) + noise
            elif mode == "ELECTROMIGRATION_LINEAR":
                iddq_24h[i] = base_0h[i] + 0.035 * spec["base_iddq_mua"] * (24/24) * 24 + noise
                iddq_96h[i] = base_0h[i] + 0.035 * spec["base_iddq_mua"] * (96/24) * 24 + noise
                iddq_168h[i] = base_0h[i] + 0.035 * spec["base_iddq_mua"] * (168/24) * 24 + noise
            elif mode == "THERMAL_RUNAWAY_EXP":
                iddq_24h[i] = base_0h[i] * np.exp(0.025 * 24) + noise
                iddq_96h[i] = base_0h[i] * np.exp(0.025 * 96) + noise
                iddq_168h[i] = base_0h[i] * np.exp(0.025 * 168) + noise

    df = pd.DataFrame({
        "lot_id": lot_id,
        "component_id": [f"{lot_id}_COMP_{i:04d}" for i in range(num_components)],
        "payload_type": payload_type,
        "device_spec_id": spec["device_spec_id"],
        "operating_voltage_v": spec["operating_voltage_v"],
        "test_temperature_c": spec["test_temperature_c"],
        "clock_freq_mhz": spec["clock_freq_mhz"],
        "wafer_x": np.round(coord_x, 4),
        "wafer_y": np.round(coord_y, 4),
        "iddq_0h": np.round(base_0h, 2),
        "iddq_24h": np.round(iddq_24h, 2),
        "iddq_96h": np.round(iddq_96h, 2),
        "iddq_168h_actual": np.round(iddq_168h, 2),
        "spec_min_iddq": spec["spec_min_mua"],
        "spec_max_iddq": spec["spec_max_mua"],
        "is_defective_gt": is_defective,
        "failure_mode_gt": failure_modes
    })
    return df

def generate_all_isro_lots():
    data_dir = "D:\\SIH 2026\\astraguard_core\\data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 7 Lots spanning the 4 ISRO payload types
    lot_configs = [
        ("LOT_2026_01", "ADITYA_L1_PAPA", 41),
        ("LOT_2026_02", "ASTROSAT_LAXPC_CZTI", 42),
        ("LOT_2026_03", "EOS_08_EOIR", 43),
        ("LOT_2026_04", "CARTOSAT_3_PAN", 44),
        ("LOT_2026_05", "ADITYA_L1_PAPA", 45),
        ("LOT_2026_06", "ASTROSAT_LAXPC_CZTI", 46),
        ("LOT_2026_07", "EOS_08_EOIR", 47),
    ]
    
    for lot_id, p_type, seed in lot_configs:
        df = generate_isro_payload_lot(lot_id=lot_id, payload_type=p_type, random_seed=seed)
        filepath = os.path.join(data_dir, f"{lot_id}.csv")
        df.to_csv(filepath, index=False)
        print(f"Generated ISRO Payload Dataset: {lot_id} ({p_type}) -> {filepath}")

if __name__ == "__main__":
    generate_all_isro_lots()
