"""
AstraGuard 2.3 — Device & Test Profile Resolver Registry
=========================================================
Loads, manages, and resolves DeviceProfile and TestProfile specifications from
ASQD 2.3 schema JSON. Handles nested evidence-tagged fields transparently,
preserving backward compatibility with 2.2 callers.
"""

import json
import os
from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field


def _extract_unit(value: Any) -> str:
    """Handle both legacy 'unit': 'uA' and ASQD 2.3 'unit': {'unit': 'uA', 'evidence': '...'}."""
    if isinstance(value, dict):
        return value.get("unit", "")
    return str(value)


def _extract_failure_modes(modes_dict: Any) -> List[str]:
    """Handle both legacy list and ASQD 2.3 dict of failure modes."""
    if isinstance(modes_dict, list):
        return modes_dict
    if isinstance(modes_dict, dict):
        return list(modes_dict.keys())
    return []


def _extract_stress_value(value: Any) -> Any:
    """Handle both legacy scalar and ASQD 2.3 {'value': ..., 'evidence': ...}."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


class DeviceProfile(BaseModel):
    device_family: str
    display_name: str
    expected_parameters: List[str]
    primary_parameter: str
    parameter_units: Dict[str, str] = Field(default_factory=dict)
    physical_failure_modes: List[str] = Field(default_factory=list)
    applicable_test_types: List[str] = Field(default_factory=list)
    model_routing: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_v23_dict(cls, d: Dict[str, Any]) -> "DeviceProfile":
        """Parse ASQD 2.3 nested JSON with evidence tags into a flat DeviceProfile."""

        # Parameter units: extract 'unit' string from nested or flat
        raw_units = d.get("parameter_units", {})
        flat_units = {k: _extract_unit(v) for k, v in raw_units.items()}

        # Failure modes: extract keys from dict or list
        flat_failures = _extract_failure_modes(d.get("physical_failure_modes", []))

        # Model routing: strip 'evidence' key
        raw_routing = d.get("model_routing", {})
        flat_routing = {k: v for k, v in raw_routing.items() if k != "evidence"}

        return cls(
            device_family=d["device_family"],
            display_name=d.get("display_name", d["device_family"]),
            expected_parameters=d.get("expected_parameters", []),
            primary_parameter=d.get("primary_parameter", ""),
            parameter_units=flat_units,
            physical_failure_modes=flat_failures,
            applicable_test_types=d.get("applicable_test_types", []),
            model_routing=flat_routing,
        )


class TestProfile(BaseModel):
    test_type: str
    display_name: str
    standard_reference: str
    is_time_series: bool
    default_checkpoints_hours: List[float]
    stress_conditions: Dict[str, Any] = Field(default_factory=dict)
    required_channels: List[str] = Field(default_factory=list)
    observable_signature: Dict[str, Any] = Field(default_factory=dict)
    applicable_device_families: List[str] = Field(default_factory=list)

    @classmethod
    def from_v23_dict(cls, d: Dict[str, Any]) -> "TestProfile":
        """Parse ASQD 2.3 nested JSON into a flat TestProfile."""

        # Stress conditions: flatten evidence-wrapping
        raw_stress = d.get("stress_conditions", {})
        flat_stress = {k: _extract_stress_value(v) for k, v in raw_stress.items()}

        return cls(
            test_type=d["test_type"],
            display_name=d.get("display_name", d["test_type"]),
            standard_reference=d.get("standard_reference", ""),
            is_time_series=d.get("is_time_series", True),
            default_checkpoints_hours=d.get("default_checkpoints_hours", [0.0, 24.0, 96.0, 168.0]),
            stress_conditions=flat_stress,
            required_channels=d.get("required_channels", []),
            observable_signature=d.get("observable_signature", {}),
            applicable_device_families=d.get("applicable_device_families", []),
        )


class ProfileRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProfileRegistry, cls).__new__(cls)
            cls._instance._load_profiles()
        return cls._instance

    def _load_profiles(self):
        self.device_profiles: Dict[str, DeviceProfile] = {}
        self.test_profiles: Dict[str, TestProfile] = {}

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        device_cfg_path = os.path.join(base_dir, "config", "device_profiles.json")
        test_cfg_path = os.path.join(base_dir, "config", "test_profiles.json")

        if os.path.exists(device_cfg_path):
            with open(device_cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, dev_dict in data.get("device_profiles", {}).items():
                    try:
                        self.device_profiles[key] = DeviceProfile.from_v23_dict(dev_dict)
                    except Exception:
                        # Fallback: try legacy flat load
                        self.device_profiles[key] = DeviceProfile(**{
                            k: v for k, v in dev_dict.items()
                            if k in DeviceProfile.model_fields
                        })

        if os.path.exists(test_cfg_path):
            with open(test_cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, test_dict in data.get("test_profiles", {}).items():
                    try:
                        self.test_profiles[key] = TestProfile.from_v23_dict(test_dict)
                    except Exception:
                        self.test_profiles[key] = TestProfile(**{
                            k: v for k, v in test_dict.items()
                            if k in TestProfile.model_fields
                        })

        # Hard fallbacks if config files were missing entirely
        if "DIGITAL_IC" not in self.device_profiles:
            self.device_profiles["DIGITAL_IC"] = DeviceProfile(
                device_family="DIGITAL_IC",
                display_name="Digital Microcircuit / FPGA",
                expected_parameters=["IDDQ", "ICC_active", "input_leakage"],
                primary_parameter="IDDQ",
                parameter_units={"IDDQ": "uA"},
                physical_failure_modes=["GATE_OXIDE_BREAKDOWN", "ELECTROMIGRATION"],
                model_routing={"anomaly_detector": "RobustZScorePopulationDetector"}
            )
        if "THERMAL_BURN_IN" not in self.test_profiles:
            self.test_profiles["THERMAL_BURN_IN"] = TestProfile(
                test_type="THERMAL_BURN_IN",
                display_name="High-Temperature Operating Life / Burn-In",
                standard_reference="MIL-STD-883 Method 1015",
                is_time_series=True,
                default_checkpoints_hours=[0.0, 24.0, 96.0, 168.0],
                stress_conditions={"temperature_c": 125.0},
                required_channels=["DUT_MEASUREMENT", "CHAMBER_TEMP"],
            )

    def reload(self):
        """Force a reload of profile configs (useful for hot-swap testing)."""
        ProfileRegistry._instance = None
        self._load_profiles()

    def get_device_profile(self, device_family: str) -> DeviceProfile:
        return self.device_profiles.get(device_family, self.device_profiles.get("DIGITAL_IC"))

    def get_test_profile(self, test_type: str) -> TestProfile:
        return self.test_profiles.get(test_type, self.test_profiles.get("THERMAL_BURN_IN"))

    def list_device_families(self) -> List[str]:
        return list(self.device_profiles.keys())

    def list_test_types(self) -> List[str]:
        return list(self.test_profiles.keys())

    def get_applicable_test_types(self, device_family: str) -> List[str]:
        """Return test types valid for a specific device family."""
        profile = self.get_device_profile(device_family)
        if profile.applicable_test_types:
            return profile.applicable_test_types
        # Fallback: search test profiles' applicable_device_families
        return [
            tt for tt, tp in self.test_profiles.items()
            if device_family in tp.applicable_device_families or not tp.applicable_device_families
        ]

    def get_observable_signature(self, test_type: str) -> Dict[str, Any]:
        """Return observable signature dict for behavioral inference."""
        tp = self.get_test_profile(test_type)
        return tp.observable_signature
