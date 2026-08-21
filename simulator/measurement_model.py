"""
AstraGuard 2.1 — ATE Measurement Model
========================================
Models the full realistic ATE measurement acquisition chain — critical for
distinguishing Class 4 (INTERMITTENT_FAULT) from noise and detecting
Class 5 (SENSOR_FAULT) via channel variance analysis.

Physics-grounded measurement chain:
  true_physical_value
        ↓
  signal_conditioning (amplifier gain + offset drift)
        ↓
  instrument_noise (Gaussian noise floor, instrument specification)
        ↓
  ADC_quantization (least significant bit rounding)
        ↓
  observed_measurement

Per-channel ATE instrument specs (prototype model):
  I_DDQ:    ±0.020 µA noise, 0.001 µA LSB quantization
  V_offset: ±0.050 mV noise, 0.010 mV LSB
  SNR:      ±0.100 dB noise, 0.010 dB LSB
  T_die:    ±0.100 °C noise, 0.010 °C LSB
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class InstrumentSpec:
    """Represents ATE instrument measurement uncertainty for a single physical channel."""
    channel_name: str
    noise_sigma: float           # 1σ Gaussian noise (instrument noise floor), in channel units
    quantization_lsb: float     # ADC LSB (least significant bit), in channel units
    gain_error_pct: float = 0.0 # Systematic gain error percentage (optional)
    offset_error: float = 0.0   # Systematic offset bias (optional)


# Default ATE instrument specifications for AstraGuard ASQD channels
DEFAULT_INSTRUMENT_SPECS: Dict[str, InstrumentSpec] = {
    "iddq_ua": InstrumentSpec(
        channel_name="iddq_ua",
        noise_sigma=0.020,       # ±20 nA noise floor
        quantization_lsb=0.001,  # 1 nA ADC resolution
    ),
    "v_offset_mv": InstrumentSpec(
        channel_name="v_offset_mv",
        noise_sigma=0.050,
        quantization_lsb=0.010,
    ),
    "snr_db": InstrumentSpec(
        channel_name="snr_db",
        noise_sigma=0.100,
        quantization_lsb=0.010,
    ),
    "t_die_c": InstrumentSpec(
        channel_name="t_die_c",
        noise_sigma=0.100,
        quantization_lsb=0.010,
    ),
}


def measure(
    true_values: np.ndarray,
    spec: InstrumentSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Full ATE measurement chain: true → signal conditioning → noise → quantization.
    Returns observed values as would be recorded in an ATE datalog (e.g., PTR record).
    """
    # 1. Systematic gain and offset errors (reproducible per-instrument session)
    scaled = true_values * (1.0 + spec.gain_error_pct / 100.0) + spec.offset_error

    # 2. Gaussian instrument noise (random, independent per sample)
    noisy = scaled + rng.normal(0.0, spec.noise_sigma, size=true_values.shape)

    # 3. ADC quantization (round to LSB)
    quantized = np.round(noisy / spec.quantization_lsb) * spec.quantization_lsb

    return quantized


def measure_all_channels(
    true_iddq: np.ndarray,
    true_voffset: np.ndarray,
    true_snr: np.ndarray,
    true_tdie: np.ndarray,
    rng: np.random.Generator,
    specs: Dict[str, InstrumentSpec] = None,
) -> Dict[str, np.ndarray]:
    """Applies measurement model to all four physical channels."""
    if specs is None:
        specs = DEFAULT_INSTRUMENT_SPECS
    return {
        "iddq_ua":    measure(true_iddq,    specs["iddq_ua"],    rng),
        "v_offset_mv": measure(true_voffset, specs["v_offset_mv"], rng),
        "snr_db":     measure(true_snr,     specs["snr_db"],     rng),
        "t_die_c":    measure(true_tdie,    specs["t_die_c"],    rng),
    }
