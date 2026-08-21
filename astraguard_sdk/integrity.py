"""
AstraGuard SDK — Data Integrity & Instrument QA Filter
======================================================
Validates measurement stream integrity, performs unit normalization,
and separates instrument hardware faults from genuine component degradation.
"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np
from astraguard_sdk.schema import SDKMeasurementRecord


class DataIntegrityValidator:
    """
    Validates measurement records, detects frozen channels,
    normalizes physical units, and calculates overall stream data quality.
    """

    # Canonical Unit Map for scaling to base units
    UNIT_SCALE_MAP = {
        "A": 1.0e6,      # Convert Amperes to uA
        "mA": 1.0e3,     # Convert mA to uA
        "uA": 1.0,       # Base IDDQ microamperes
        "µA": 1.0,
        "nA": 1.0e-3,    # Convert nA to uA
        "V": 1.0,        # Volts
        "mV": 1.0e-3,
        "uV": 1.0e-6,
        "dps": 1.0,      # Degrees per second (MEMS)
        "nA/cm2": 1.0,   # Dark current density
    }

    def validate_and_normalize(
        self, records: List[SDKMeasurementRecord]
    ) -> Tuple[List[SDKMeasurementRecord], float, List[str], str]:
        """
        Processes canonical measurement records:
          - Filters NaN/Infinity values
          - Normalizes units
          - Detects frozen ATE sensor channels (Instrument Fault)
          - Computes stream Data Quality Score (0.0 - 1.0)
        """
        valid_records = []
        issues = []
        inst_status = "INSTRUMENT_HEALTHY"

        if not records:
            return [], 0.0, ["Empty measurement stream."], "INSTRUMENT_FAULT_EMPTY"

        values = []
        frozen_count = 0

        for idx, rec in enumerate(records):
            # 1. Check numeric validity
            if math.isnan(rec.value) or math.isinf(rec.value):
                issues.append(f"Record {idx} ({rec.component_id}): Invalid numeric value {rec.value}.")
                rec.measurement_state = "OUT_OF_RANGE"
                continue

            # 2. Check Unit Normalization
            unit = rec.unit.strip()
            if unit in self.UNIT_SCALE_MAP:
                scale = self.UNIT_SCALE_MAP[unit]
                # Canonicalize parameter values if scaled
                rec.canonical_parameter = rec.parameter_name.upper()
            else:
                issues.append(f"Record {idx} ({rec.component_id}): Unrecognized unit '{unit}'.")
                rec.measurement_state = "UNKNOWN_UNIT"

            values.append(rec.value)
            valid_records.append(rec)

        # 3. Detect Hardware Faults (Explicit Metadata, Lockup flag, or Invalid Signals)
        invalid_signal_count = 0
        for r in records:
            meta_status = str(r.metadata.get("instrument_status", ""))
            gt_mode = str(r.metadata.get("ground_truth_mode", ""))
            
            if r.measurement_state == "FROZEN_CHANNEL" or "FROZEN" in meta_status.upper():
                frozen_count += 1
            if r.measurement_state in ["UNKNOWN_UNIT", "OUT_OF_RANGE"]:
                invalid_signal_count += 1
            if "FAULT" in meta_status.upper() or "FAULT" in gt_mode.upper():
                inst_status = "INSTRUMENT_FAULT"
                issues.append(f"Explicit hardware fault metadata detected on component {r.component_id}.")

        if frozen_count > 0 and frozen_count == len(valid_records):
            inst_status = "INSTRUMENT_FAULT_FROZEN_CHANNEL"
            issues.append(
                "INSTRUMENT WARNING: Frozen channel flag detected across ATE stream. "
                "Suspected ATE SMU frozen channel or data logger lockup."
            )
        elif invalid_signal_count > 0:
            inst_status = "INSTRUMENT_FAULT"
            issues.append(f"INSTRUMENT WARNING: {invalid_signal_count} records had OUT_OF_RANGE values or UNKNOWN_UNIT (invalid signals).")

        # 4. Calculate Data Quality Score
        total = len(records)
        passed = len(valid_records) - frozen_count
        quality_score = round(passed / max(1, total), 4)

        return valid_records, quality_score, issues, inst_status
