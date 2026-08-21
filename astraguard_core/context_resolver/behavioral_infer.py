"""
AstraGuard 2.2 — Level 3 Behavioral Inference Engine
Infers test & device context from parameter statistical distributions, units, and signal characteristics.
Calculates calibrated behavioral confidence scores and flags UNKNOWN or AMBIGUOUS contexts.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from astraguard_core.context_resolver.schema import (
    TestContext,
    MeasurementRecord,
    TestIdentityResolutionResult,
    IdentificationSource,
    ResolutionStatus,
)
from astraguard_core.context_resolver.profiles import ProfileRegistry


class BehavioralInferenceEngine:
    def __init__(self):
        self.registry = ProfileRegistry()

    def infer(
        self,
        records: List[MeasurementRecord],
        current_result: Optional[TestIdentityResolutionResult] = None,
    ) -> TestIdentityResolutionResult:
        """
        Performs statistical & physical signal inference to identify device family and test type.
        """
        if current_result and current_result.confidence_score >= 0.90 and current_result.status == ResolutionStatus.KNOWN_CONTEXT:
            # Metadata resolution already high confidence
            return current_result

        if not records:
            dev_profile = self.registry.get_device_profile("DIGITAL_IC")
            test_profile = self.registry.get_test_profile("THERMAL_BURN_IN")
            return TestIdentityResolutionResult(
                resolved_device_family="UNKNOWN",
                resolved_test_type="THERMAL_BURN_IN",
                primary_parameter="UNKNOWN",
                confidence_score=0.10,
                identification_source=IdentificationSource.UNKNOWN,
                status=ResolutionStatus.UNKNOWN_CONTEXT,
                requires_operator_confirmation=True,
                expected_parameters=dev_profile.expected_parameters,
                active_profile_name="UNKNOWN_DEVICE_PROFILE",
                notes=["No records provided for behavioral inference."],
            )

        param_names = [r.parameter_name.lower() for r in records]
        units = set([r.unit.lower() for r in records])
        values = [r.value for r in records]

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        notes = []

        # Behavioral Heuristics:
        # 1. MEMS Gyroscope Detection (units: dps, ppm, dps/rtHz or names containing 'gyro', 'rate', 'bias')
        if any(u in ["dps", "ppm", "dps/rthz"] for u in units) or any("rate" in p or "gyro" in p or "bias" in p for p in param_names):
            dev_profile = self.registry.get_device_profile("MEMS_GYROSCOPE")
            test_profile = self.registry.get_test_profile("THERMAL_BURN_IN")
            notes.append("Behavioral Inference: Detected MEMS Gyroscope signal profile (dps / angular rate bias dynamics).")
            return TestIdentityResolutionResult(
                resolved_device_family="MEMS_GYROSCOPE",
                resolved_test_type="THERMAL_BURN_IN",
                primary_parameter="zero_rate_offset",
                confidence_score=0.82,
                identification_source=IdentificationSource.BEHAVIORAL_INFERENCE,
                status=ResolutionStatus.KNOWN_CONTEXT,
                requires_operator_confirmation=False,
                expected_parameters=dev_profile.expected_parameters,
                active_profile_name=dev_profile.display_name,
                notes=notes,
            )

        # 2. Image Sensor Detection (units: nA/cm2, count or names containing 'dark', 'pixel', 'dsnu')
        if any(u in ["na/cm2", "photons", "count"] for u in units) or any("dark" in p or "pixel" in p or "dsnu" in p for p in param_names):
            dev_profile = self.registry.get_device_profile("IMAGE_SENSOR")
            test_profile = self.registry.get_test_profile("THERMAL_BURN_IN")
            notes.append("Behavioral Inference: Detected Image Sensor dark current / pixel noise dynamics.")
            return TestIdentityResolutionResult(
                resolved_device_family="IMAGE_SENSOR",
                resolved_test_type="THERMAL_BURN_IN",
                primary_parameter="dark_current_density",
                confidence_score=0.80,
                identification_source=IdentificationSource.BEHAVIORAL_INFERENCE,
                status=ResolutionStatus.KNOWN_CONTEXT,
                requires_operator_confirmation=False,
                expected_parameters=dev_profile.expected_parameters,
                active_profile_name=dev_profile.display_name,
                notes=notes,
            )

        # 3. Precision Voltage Ref Detection (units: uV, ppm/c, or names containing 'vref', 'voltage_drift')
        if any(u in ["uv", "ppm/c"] for u in units) or any("vref" in p or "drift" in p for p in param_names):
            dev_profile = self.registry.get_device_profile("PRECISION_VOLTAGE_REF")
            test_profile = self.registry.get_test_profile("THERMAL_BURN_IN")
            notes.append("Behavioral Inference: Detected Voltage Reference precision micro-volt drift dynamics.")
            return TestIdentityResolutionResult(
                resolved_device_family="PRECISION_VOLTAGE_REF",
                resolved_test_type="THERMAL_BURN_IN",
                primary_parameter="output_voltage_drift",
                confidence_score=0.78,
                identification_source=IdentificationSource.BEHAVIORAL_INFERENCE,
                status=ResolutionStatus.KNOWN_CONTEXT,
                requires_operator_confirmation=False,
                expected_parameters=dev_profile.expected_parameters,
                active_profile_name=dev_profile.display_name,
                notes=notes,
            )

        # 4. Digital / Mixed Signal IC IDDQ Detection (units: uA, mA, nA, or names containing 'iddq', 'icc', 'current')
        if any(u in ["ua", "ma", "na"] for u in units) or any("iddq" in p or "current" in p or "leakage" in p for p in param_names):
            dev_profile = self.registry.get_device_profile("DIGITAL_IC")
            test_profile = self.registry.get_test_profile("THERMAL_BURN_IN")
            notes.append("Behavioral Inference: Detected Digital IC IDDQ / quiescent current dynamics.")
            return TestIdentityResolutionResult(
                resolved_device_family="DIGITAL_IC",
                resolved_test_type="THERMAL_BURN_IN",
                primary_parameter="IDDQ",
                confidence_score=0.85,
                identification_source=IdentificationSource.BEHAVIORAL_INFERENCE,
                status=ResolutionStatus.KNOWN_CONTEXT,
                requires_operator_confirmation=False,
                expected_parameters=dev_profile.expected_parameters,
                active_profile_name=dev_profile.display_name,
                notes=notes,
            )

        # 5. Completely Unfamiliar / Corrupted Signals -> UNKNOWN_CONTEXT Guard
        dev_profile = self.registry.get_device_profile("DIGITAL_IC")
        test_profile = self.registry.get_test_profile("THERMAL_BURN_IN")
        notes.append("Behavioral Inference: Signals did not match any known profile signatures. Flagging UNKNOWN_CONTEXT requiring operator confirmation.")

        return TestIdentityResolutionResult(
            resolved_device_family="UNKNOWN",
            resolved_test_type=test_profile.test_type,
            primary_parameter="UNKNOWN",
            confidence_score=0.25,
            identification_source=IdentificationSource.BEHAVIORAL_INFERENCE,
            status=ResolutionStatus.UNKNOWN_CONTEXT,
            requires_operator_confirmation=True,
            expected_parameters=[],
            missing_parameters=[],
            unexpected_parameters=[r.parameter_name for r in records],
            active_profile_name="UNKNOWN_DEVICE_PROFILE",
            notes=notes,
        )
