"""
AstraGuard 2.0 — Multi-Parametric High-Fidelity Semiconductor & Sensor Burn-In Dataset Simulator (PISDG Pro 2.0)
================================================================================------------------------------
Implements the multi-channel, time-series burn-in telemetry generation pipeline per ISRO/DoS screening rules,
MIL-STD-883 Method 1015, and NASA EEE-INST-002 standards.

Generates 6 co-dependent physical channels sampled hourly from t=0 to t=168 (169 sample points):
  1. T_ambient (°C) — Exponential chamber ramp up to 125°C/150°C
  2. I_DDQ (µA) — Quiescent leakage current with Arrhenius temperature coupling
  3. V_offset (mV) — Zero-g bias output voltage (MEMS substrate stability)
  4. SNR (dB) — Signal-to-Noise Ratio functional fidelity
  5. T_die (°C) — On-die operating temperature: T_ambient + (I_DDQ * V_CC * R_th)
  6. V_CC (V) — Operating voltage with +20% overvoltage stress (3.96V)

Failure Categories (Target Classes):
  0: Category A — Nominal Survivors (99.5% Lot Coherence)
  1: Category B — Thermal Runaway / Electromigration (Exponential current explosion after onset t in [24, 48])
  2: Category C — Structural MEMS Micro-Cracking (Random walk bias drift & transient voltage spikes)
  3: Category D — Spatial Outlier / "Freak" Part (Hour 0 modified Z-score > 3.5σ)
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Any

class ISROBurnInSimulator:
    def __init__(
        self,
        num_components: int = 200,
        timesteps_hours: int = 168,
        defect_rate: float = 0.05,  # 5% total defects across Category B, C, D for realistic evaluation
        random_seed: int = 42
    ):
        self.num_components = num_components
        self.timesteps = timesteps_hours + 1  # t=0 to t=168 (169 points)
        self.defect_rate = defect_rate
        self.random_seed = random_seed

    def generate_lot(self, lot_id: str = "ISRO_LOT_2026_01") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates:
          1. Time-series hourly telemetry dataframe (169 rows per component, 33,800 rows per lot)
          2. Component summary dataframe at t=24h checkpoint for Module A & B inference
        """
        np.random.seed(self.random_seed)
        
        # 1. Component Health Class Assignment
        # Classes: 0: NOMINAL, 1: THERMAL_RUNAWAY, 2: MEMS_MICROCRACK, 3: FREAK_OUTLIER
        num_defects = max(4, int(self.num_components * self.defect_rate))
        health_classes = np.zeros(self.num_components, dtype=int)
        
        # Distribute defects evenly across Cat B, C, D
        defect_indices = np.random.choice(self.num_components, size=num_defects, replace=False)
        for i, idx in enumerate(defect_indices):
            health_classes[idx] = (i % 3) + 1  # 1, 2, or 3
            
        time_hours = np.arange(self.timesteps)
        
        # Chamber Temperature Profile: Exponential ramp to 125°C in 2 hours
        t_ambient = 25.0 + 100.0 * (1.0 - np.exp(-time_hours / 0.8))
        
        # Base Operating Voltage (+20% stress over 3.3V = 3.96V)
        v_cc = np.full(self.timesteps, 3.96)
        r_th = 0.05  # Thermal resistance °C/mW
        
        telemetry_rows = []
        summary_rows = []
        
        for comp_i in range(self.num_components):
            comp_id = f"{lot_id}_COMP_{comp_i:04d}"
            cat_label = health_classes[comp_i]
            
            # Base parameters
            if cat_label == 3:  # Freak Outlier at Hour 0
                base_iddq = 1.2 + np.random.uniform(0.4, 0.7)  # 4-5 MAD outlier at hour 0
                base_voffset = 5.0 + np.random.uniform(0.8, 1.2)
            else:
                base_iddq = np.random.normal(1.2, 0.08)
                base_voffset = np.random.normal(5.0, 0.15)
                
            base_snr = np.random.normal(45.0, 0.5)
            
            # Time-series generation
            iddq = np.zeros(self.timesteps)
            v_offset = np.zeros(self.timesteps)
            snr = np.zeros(self.timesteps)
            t_die = np.zeros(self.timesteps)
            
            onset_hour = np.random.randint(24, 49) if cat_label in [1, 2] else 169
            
            for t in range(self.timesteps):
                noise_iddq = np.random.normal(0, 0.02)
                noise_voffset = np.random.normal(0, 0.05)
                noise_snr = np.random.normal(0, 0.1)
                
                # Arrhenius thermal leakage coupling multiplier
                thermal_coupling = np.exp(0.012 * (t_ambient[t] - 25.0))
                
                if cat_label == 0:  # Nominal Survivor
                    iddq[t] = base_iddq * thermal_coupling + 0.001 * t + noise_iddq
                    v_offset[t] = base_voffset + 0.002 * (t_ambient[t] - 25.0) + noise_voffset
                    snr[t] = base_snr - 0.02 * (t_ambient[t] - 25.0) + noise_snr
                    
                elif cat_label == 1:  # Thermal Runaway / Electromigration
                    if t < onset_hour:
                        iddq[t] = base_iddq * thermal_coupling + 0.001 * t + noise_iddq
                    else:
                        # Exponential current surge after onset
                        lambda_k = 0.08
                        iddq[t] = base_iddq * thermal_coupling + 0.5 * np.exp(lambda_k * (t - onset_hour)) + noise_iddq
                    v_offset[t] = base_voffset + 0.005 * (t_ambient[t] - 25.0) + noise_voffset
                    snr[t] = base_snr - 0.05 * max(0, t - onset_hour) + noise_snr
                    
                elif cat_label == 2:  # Structural MEMS Micro-Cracking
                    iddq[t] = base_iddq * thermal_coupling + noise_iddq
                    if t < onset_hour:
                        v_offset[t] = base_voffset + noise_voffset
                    else:
                        # Random walk + transient voltage spikes
                        rw = np.sum(np.random.normal(0, 0.15, size=t - onset_hour + 1))
                        spike = 1.5 if (t % 7 == 0) else 0.0
                        v_offset[t] = base_voffset + rw + spike + noise_voffset
                    snr[t] = base_snr - 0.1 * max(0, t - onset_hour) + noise_snr
                    
                elif cat_label == 3:  # Spatial Outlier / Freak Part
                    iddq[t] = base_iddq * thermal_coupling + 0.002 * t + noise_iddq
                    v_offset[t] = base_voffset + noise_voffset
                    snr[t] = base_snr - 1.5 + noise_snr
                    
                # Compute On-Die Temperature (T_die = T_ambient + I_DDQ * V_CC * R_th)
                # Note: I_DDQ in µA, V_CC in V -> Power in µW
                power_mw = (iddq[t] * v_cc[t]) / 1000.0
                t_die[t] = t_ambient[t] + (power_mw * r_th)
                
                telemetry_rows.append({
                    "lot_id": lot_id,
                    "component_id": comp_id,
                    "hour": t,
                    "t_ambient_c": round(t_ambient[t], 2),
                    "iddq_ua": round(iddq[t], 3),
                    "v_offset_mv": round(v_offset[t], 3),
                    "snr_db": round(snr[t], 2),
                    "t_die_c": round(t_die[t], 2),
                    "v_cc_v": v_cc[t],
                    "category_label": cat_label
                })

            # Checkpoint at 24h for summary
            summary_rows.append({
                "lot_id": lot_id,
                "component_id": comp_id,
                "payload_type": "SPACE_MEMS_SENSOR",
                "device_spec_id": "ISRO-MEMS-SENS-2026",
                "operating_voltage_v": 3.96,
                "test_temperature_c": 125.0,
                "iddq_0h": round(iddq[0], 3),
                "iddq_24h": round(iddq[24], 3),
                "iddq_168h_actual": round(iddq[168], 3),
                "v_offset_0h": round(v_offset[0], 3),
                "v_offset_24h": round(v_offset[24], 3),
                "snr_24h": round(snr[24], 2),
                "spec_max_iddq": 15.0,
                "is_defective_gt": 1 if cat_label > 0 else 0,
                "category_label": cat_label,
                "failure_mode_gt": ["HEALTHY", "THERMAL_RUNAWAY", "MEMS_MICROCRACK", "FREAK_OUTLIER"][cat_label]
            })

        df_telemetry = pd.DataFrame(telemetry_rows)
        df_summary = pd.DataFrame(summary_rows)
        return df_telemetry, df_summary

if __name__ == "__main__":
    print("=== ASTRAGUARD 2.0 HIGH-FIDELITY SIMULATOR EXECUTION ===")
    sim = ISROBurnInSimulator(num_components=200, random_seed=42)
    df_tel, df_sum = sim.generate_lot("LOT_2026_01")
    
    out_dir = "D:\\SIH 2026\\astraguard_core\\data"
    os.makedirs(out_dir, exist_ok=True)
    
    tel_path = os.path.join(out_dir, "LOT_2026_01_telemetry.csv")
    sum_path = os.path.join(out_dir, "LOT_2026_01.csv")
    
    df_tel.to_csv(tel_path, index=False)
    df_sum.to_csv(sum_path, index=False)
    
    print(f"Generated High-Fidelity Time-Series Telemetry: {tel_path} ({len(df_tel)} rows)")
    print(f"Generated Lot Summary File: {sum_path} ({len(df_sum)} rows)")
    print("\nClass Distribution in Generated Lot:")
    print(df_sum["failure_mode_gt"].value_counts().to_string())
