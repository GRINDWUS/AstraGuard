"""
AstraGuard 2.3/2.4 SDK — Canonical Measurement Schema & Data Structures
========================================================================
Read-only, non-invasive schema definitions for external ATE data streams.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel, Field


class SDKMeasurementRecord(BaseModel):
    """Canonical representation of a single measurement sample from any ATE format."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    component_id: str
    lot_id: Optional[str] = None
    device_family: Optional[str] = None
    test_id: Optional[str] = None
    test_type: Optional[str] = None
    parameter_name: str
    canonical_parameter: Optional[str] = None
    value: float
    unit: str
    temperature_c: Optional[float] = 25.0
    operating_voltage_v: Optional[float] = 5.0
    channel_id: Optional[str] = "CH_01"
    instrument_id: Optional[str] = "ATE_SMU_01"
    measurement_state: str = "VALID"  # VALID, FROZEN, OUT_OF_RANGE, MISSING
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisSessionMetadata(BaseModel):
    """Traceability & Audit Metadata for an AstraGuard SDK analysis session."""
    session_id: str = Field(default_factory=lambda: f"AG-SDK-{uuid.uuid4().hex[:8].upper()}")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    operator_id: Optional[str] = "QA_OPERATOR_DEFAULT"
    analysis_mode: str = "FULL_SCREENING_ANALYSIS"
    system_version: str = "AstraGuard-2.4"
    model_version: str = "Module-B-v2"
    feature_engine_version: str = "v2"
    threshold_config_version: str = "phase3"
    decision_policy_version: str = "2.4"
    read_only_guarantee: bool = True


class SDKAnalysisResult(BaseModel):
    """Structured end-to-end output returned by AstraGuard Client."""
    session: AnalysisSessionMetadata
    is_execution_allowed: bool
    context_status: str
    resolved_device_family: str
    resolved_test_type: str
    resolved_primary_parameter: str
    context_confidence: float
    data_quality_score: float
    instrument_health_status: str
    total_records_processed: int
    recommendation: str  # GREEN_NORMAL_CANDIDATE, YELLOW_REVIEW, RED_HIGH_RISK, HOLD_OPERATOR_REVIEW
    evidence_trail: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    components_summary: Dict[str, int] = Field(default_factory=dict)
    detailed_components: List[Dict[str, Any]] = Field(default_factory=list)
    audit_id: str = Field(default_factory=lambda: f"AUDIT-{uuid.uuid4().hex[:12].upper()}")
