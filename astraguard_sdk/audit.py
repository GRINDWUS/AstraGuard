"""
AstraGuard SDK — Audit Trail & Security Logger
==============================================
Logs reproducible audit records for every analysis session.
"""

import os
import json
from datetime import datetime
from astraguard_sdk.schema import SDKAnalysisResult


class AstraGuardAuditLogger:
    """Logs auditable JSON records to disk for every SDK analysis session."""

    def __init__(self, log_dir: str = "astraguard_sdk/audit_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_session(self, result: SDKAnalysisResult) -> str:
        filename = f"{result.session.session_id}_{result.audit_id}.json"
        filepath = os.path.join(self.log_dir, filename)

        record = {
            "session_id": result.session.session_id,
            "audit_id": result.audit_id,
            "timestamp": datetime.utcnow().isoformat(),
            "operator_id": result.session.operator_id,
            "read_only_guarantee": result.session.read_only_guarantee,
            "context": {
                "status": result.context_status,
                "resolved_device": result.resolved_device_family,
                "resolved_test": result.resolved_test_type,
                "resolved_param": result.resolved_primary_parameter,
                "confidence": result.context_confidence,
            },
            "data_quality_score": result.data_quality_score,
            "instrument_health": result.instrument_health_status,
            "recommendation": result.recommendation,
            "evidence_trail": result.evidence_trail,
            "reasons": result.reasons,
            "components_summary": result.components_summary,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        latest_path = os.path.join(self.log_dir, "latest_audit.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return filepath
