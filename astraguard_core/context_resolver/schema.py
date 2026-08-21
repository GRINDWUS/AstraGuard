"""
AstraGuard 2.2 — Test Context & Measurement Schema Definitions
Canonical data abstractions for ATE Domain Intelligence & Discovery.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ParameterCategory(str, Enum):
    ELECTRICAL = "ELECTRICAL"
    THERMAL = "THERMAL"
    MECHANICAL = "MECHANICAL"
    OPTICAL = "OPTICAL"
    PRESSURE = "PRESSURE"
    FUNCTIONAL = "FUNCTIONAL"
    TIMING = "TIMING"
    QUALITY = "QUALITY"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    INSTRUMENT_HEALTH = "INSTRUMENT_HEALTH"


class IdentificationSource(str, Enum):
    EXPLICIT_METADATA = "EXPLICIT_METADATA"
    METADATA_MAPPING = "METADATA_MAPPING"
    BEHAVIORAL_INFERENCE = "BEHAVIORAL_INFERENCE"
    UNKNOWN = "UNKNOWN"


class ResolutionStatus(str, Enum):
    KNOWN_CONTEXT = "KNOWN_CONTEXT"
    PARTIAL_CONTEXT = "PARTIAL_CONTEXT"
    AMBIGUOUS_CONTEXT = "AMBIGUOUS_CONTEXT"
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"


class DeviceMetadata(BaseModel):
    device_family: str = Field(default="DIGITAL_IC", description="Target device family e.g. DIGITAL_IC, MEMS_GYROSCOPE")
    part_number: Optional[str] = Field(default=None, description="Part number e.g. AG-IC-883B")
    technology: Optional[str] = Field(default=None, description="Silicon technology node e.g. CMOS_180nm")
    package_type: Optional[str] = Field(default=None, description="Package type e.g. CQFP-64, LCC-20")
    lot_id: Optional[str] = Field(default=None, description="Screening lot ID")
    wafer_id: Optional[str] = Field(default=None, description="Wafer ID")


class TestMetadata(BaseModel):
    __test__ = False
    test_program_id: Optional[str] = Field(default=None, description="ATE Test Program identifier")
    procedure_id: Optional[str] = Field(default=None, description="Standard Operating Procedure ID")
    test_step_id: Optional[str] = Field(default=None, description="Specific step ID e.g. STEP_168H_FINAL")
    test_type: str = Field(default="THERMAL_BURN_IN", description="Screening/Qualification test family")


class HardwareContext(BaseModel):
    station_id: Optional[str] = Field(default=None, description="ATE station identifier")
    instrument_id: Optional[str] = Field(default=None, description="Primary SMU/Instrument ID")
    chamber_id: Optional[str] = Field(default=None, description="Burn-in or TVAC chamber ID")
    board_id: Optional[str] = Field(default=None, description="Burn-in board (BIB) or Load board ID")


class EnvironmentContext(BaseModel):
    ambient_temperature_c: float = Field(default=25.0, description="Ambient temperature in degrees Celsius")
    chamber_pressure_torr: Optional[float] = Field(default=760.0, description="Chamber pressure in Torr")
    relative_humidity_pct: Optional[float] = Field(default=0.0, description="Relative humidity percentage")


class StressConditions(BaseModel):
    bias_voltage_v: Optional[float] = Field(default=5.0, description="Applied bias voltage")
    stress_duration_hours: float = Field(default=168.0, description="Target total stress duration")
    clock_frequency_mhz: Optional[float] = Field(default=None, description="Dynamic burn-in clock frequency")


class TestContext(BaseModel):
    __test__ = False
    device_metadata: DeviceMetadata = Field(default_factory=DeviceMetadata)
    test_metadata: TestMetadata = Field(default_factory=TestMetadata)
    hardware_context: HardwareContext = Field(default_factory=HardwareContext)
    environment: EnvironmentContext = Field(default_factory=EnvironmentContext)
    stress_conditions: StressConditions = Field(default_factory=StressConditions)


class MeasurementRecord(BaseModel):
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")
    component_id: str = Field(description="Unique DUT component identifier")
    checkpoint_name: str = Field(default="0h", description="Checkpoint tag e.g. 0h, 24h, 96h, 168h")
    parameter_name: str = Field(description="Parameter name e.g. IDDQ, zero_rate_offset")
    parameter_category: ParameterCategory = Field(default=ParameterCategory.ELECTRICAL)
    value: float = Field(description="Numerical measured value")
    unit: str = Field(default="uA", description="Measurement unit")
    channel_id: Optional[str] = Field(default=None, description="Channel ID on load board / ATE")
    spec_low: Optional[float] = Field(default=None, description="Lower specification limit")
    spec_high: Optional[float] = Field(default=None, description="Upper specification limit")
    is_instrument_channel: bool = Field(default=False, description="True if channel measures instrument health")


class TestIdentityResolutionResult(BaseModel):
    resolved_device_family: str
    resolved_test_type: str
    primary_parameter: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    identification_source: IdentificationSource
    status: ResolutionStatus = Field(default=ResolutionStatus.KNOWN_CONTEXT)
    requires_operator_confirmation: bool = Field(default=False)
    expected_parameters: List[str]
    missing_parameters: List[str] = Field(default_factory=list)
    unexpected_parameters: List[str] = Field(default_factory=list)
    active_profile_name: str
    notes: List[str] = Field(default_factory=list)
