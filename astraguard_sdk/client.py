"""
AstraGuard 2.4 — Main Read-Only Python SDK Client
=================================================
Provides a non-invasive, read-only analytics client for external ATE measurement data.

Usage:
  from astraguard_sdk.client import AstraGuardClient

  client = AstraGuardClient()
  result = client.analyze("path/to/ate_data.csv")
  print(result.recommendation)
"""

from typing import Union, List, Dict, Any, Optional
import pandas as pd

from astraguard_sdk.schema import (
    SDKMeasurementRecord,
    AnalysisSessionMetadata,
    SDKAnalysisResult,
)
from astraguard_sdk.adapters import CSVATEAdapter, JSONATEAdapter
from astraguard_sdk.integrity import DataIntegrityValidator
from astraguard_sdk.policy import AstraGuardPolicyEngine
from astraguard_sdk.audit import AstraGuardAuditLogger

from astraguard_core.context_resolver.explicit_parser import ExplicitMetadataParser
from astraguard_core.context_resolver.behavioral_infer import BehavioralInferenceEngine
from astraguard_core.context_resolver.schema import (
    TestContext, DeviceMetadata, TestMetadata, MeasurementRecord
)
from astraguard_core.model_router import AstraGuardModelRouter


class AstraGuardClient:
    """
    AstraGuard Safe ATE Integration Client.
    Guarantees 100% read-only, non-invasive, auditable analytics.
    """

    def __init__(self, operator_id: str = "QA_OPERATOR_DEFAULT"):
        from astraguard_core.context_resolver import profiles as _p
        _p.ProfileRegistry._instance = None
        self.operator_id = operator_id
        self.csv_adapter = CSVATEAdapter()
        self.json_adapter = JSONATEAdapter()
        self.integrity_validator = DataIntegrityValidator()
        self.explicit_parser = ExplicitMetadataParser()
        self.behavioral_infer = BehavioralInferenceEngine()
        self.model_router = AstraGuardModelRouter()
        self.policy_engine = AstraGuardPolicyEngine()
        self.audit_logger = AstraGuardAuditLogger()

    def analyze(
        self,
        data: Union[str, pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        analysis_mode: str = "FULL_SCREENING_ANALYSIS"
    ) -> SDKAnalysisResult:
        """
        Executes end-to-end read-only screening analysis on external ATE measurement data.
        """
        session = AnalysisSessionMetadata(
            operator_id=self.operator_id,
            analysis_mode=analysis_mode,
            read_only_guarantee=True
        )

        # Step 1: Adapt input payload to canonical MeasurementRecords
        if isinstance(data, (str, pd.DataFrame)):
            records = self.csv_adapter.parse(data)
            df_lot = self.csv_adapter.to_dataframe(records)
        elif isinstance(data, (dict, list)):
            records = self.json_adapter.parse(data)
            df_lot = self.json_adapter.to_dataframe(records)
        else:
            raise ValueError("Unsupported data payload. Provide CSV path, DataFrame, or JSON.")

        # Step 2: Data Integrity & Instrument QA Filtering
        valid_records, quality_score, issues, inst_status = self.integrity_validator.validate_and_normalize(records)

        # Step 3: Multi-Level Context Resolution
        first_rec = valid_records[0] if valid_records else records[0]
        explicit_fam = first_rec.device_family
        explicit_test = first_rec.test_type
        obs_params = list(set([r.parameter_name for r in valid_records]))

        if explicit_fam and explicit_test:
            ctx = TestContext(
                device_metadata=DeviceMetadata(device_family=explicit_fam),
                test_metadata=TestMetadata(test_type=explicit_test)
            )
            context_res = self.explicit_parser.resolve(test_context=ctx, observed_parameters=obs_params)
        elif obs_params:
            context_res = self.explicit_parser.resolve(observed_parameters=obs_params)
        else:
            # Behavioral Inference
            b_recs = [
                MeasurementRecord(
                    component_id=r.component_id,
                    parameter_name=r.parameter_name,
                    value=r.value,
                    unit=r.unit
                ) for r in valid_records[:10]
            ]
            context_res = self.behavioral_infer.infer(records=b_recs)

        # Step 4: Model Routing (Context Safety Interlocked)
        routing_report = self.model_router.route_and_predict(df_lot=df_lot, context_result=context_res)

        # Step 5: Evidence Fusion & Policy Evaluation
        result = self.policy_engine.evaluate(
            session=session,
            is_execution_allowed=routing_report.is_execution_allowed,
            context_status=context_res.status,
            resolved_device=context_res.resolved_device_family,
            resolved_test=context_res.resolved_test_type,
            resolved_param=context_res.primary_parameter,
            context_confidence=context_res.confidence_score,
            quality_score=quality_score,
            inst_status=inst_status,
            routing_report=routing_report
        )

        # Step 6: Log Session Audit Trail
        self.audit_logger.log_session(result)

        return result
