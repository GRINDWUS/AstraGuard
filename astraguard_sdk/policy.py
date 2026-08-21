"""
AstraGuard SDK — Policy Engine & Evidence Fusion
=================================================
Translates model predictions, context resolution status, and instrument health QA
into auditable QA screening recommendations.
"""

from typing import List, Dict, Any, Tuple
from astraguard_sdk.schema import SDKAnalysisResult, AnalysisSessionMetadata


class AstraGuardPolicyEngine:
    """
    Evidence Fusion & Policy Engine for AstraGuard SDK.
    Enforces clear separation between raw model output and QA policy recommendation.
    """

    def evaluate(
        self,
        session: AnalysisSessionMetadata,
        is_execution_allowed: bool,
        context_status: str,
        resolved_device: str,
        resolved_test: str,
        resolved_param: str,
        context_confidence: float,
        quality_score: float,
        inst_status: str,
        routing_report: Any
    ) -> SDKAnalysisResult:
        evidence = []
        reasons = []

        # 1. Check Safety Interlock
        if not is_execution_allowed or "FAULT" in inst_status:
            rec = "HOLD_OPERATOR_REVIEW"
            evidence.append(f"[SAFETY_INTERLOCK] Context Status: {context_status}")
            evidence.append(f"[QA_ALERT] Instrument QA Status: {inst_status}")
            reasons.append(
                "Automated screening model interlocked: Context is ambiguous/unknown "
                "or test equipment hardware fault was detected."
            )
            return SDKAnalysisResult(
                session=session,
                is_execution_allowed=False,
                context_status=context_status,
                resolved_device_family=resolved_device,
                resolved_test_type=resolved_test,
                resolved_primary_parameter=resolved_param,
                context_confidence=context_confidence,
                data_quality_score=quality_score,
                instrument_health_status=inst_status,
                total_records_processed=0,
                recommendation=rec,
                evidence_trail=evidence,
                reasons=reasons,
                components_summary={"HOLD": 0, "GREEN": 0, "YELLOW": 0, "RED": 0},
                detailed_components=[]
            )

        # 2. Evaluate Executed Model Results
        green = routing_report.green_pass_count
        yellow = routing_report.yellow_extended_count
        red = routing_report.red_reject_count
        total = routing_report.total_components_evaluated

        evidence.append(f"[CONTEXT] Resolved Device: {resolved_device} via {context_status} (Conf: {context_confidence:.2f})")
        evidence.append(f"[ROUTER] Active Anomaly Model: {routing_report.active_anomaly_detector}")
        evidence.append(f"[ROUTER] Active Forecaster: {routing_report.active_forecaster}")
        evidence.append(f"[METRICS] Evaluated {total} components | Green: {green}, Yellow: {yellow}, Red: {red}")
        evidence.append(f"[BENCHMARK] Chamber Hours Saved: {routing_report.chamber_hours_saved_pct:.2f}%")

        if red > 0:
            rec = "RED_HIGH_RISK"
            reasons.append(f"{red} components exhibited severe population drift or predicted 168h specification breach.")
        elif yellow > 0:
            rec = "YELLOW_REVIEW"
            reasons.append(f"{yellow} components assigned to +48h extended observation due to minor kinetic drift.")
        else:
            rec = "GREEN_NORMAL_CANDIDATE"
            reasons.append("100% of evaluated components meet population stability and 168h forecast thresholds.")

        return SDKAnalysisResult(
            session=session,
            is_execution_allowed=True,
            context_status=context_status,
            resolved_device_family=resolved_device,
            resolved_test_type=resolved_test,
            resolved_primary_parameter=resolved_param,
            context_confidence=context_confidence,
            data_quality_score=quality_score,
            instrument_health_status=inst_status,
            total_records_processed=total,
            recommendation=rec,
            evidence_trail=evidence,
            reasons=reasons,
            components_summary={"GREEN": green, "YELLOW": yellow, "RED": red},
            detailed_components=routing_report.results
        )
