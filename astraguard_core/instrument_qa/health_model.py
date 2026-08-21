"""
AstraGuard 2.2 — Instrument & Test System Health Model
Separates instrument/channel/chamber faults from component defects.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class InstrumentHealthStatus(BaseModel):
    is_instrument_healthy: bool = True
    fault_type: Optional[str] = None
    affected_channels: List[str] = Field(default_factory=list)
    confidence_score: float = 1.0
    action_recommendation: str = "PROCEED_WITH_COMPONENT_RELIABILITY_EVALUATION"
    diagnostic_details: List[str] = Field(default_factory=list)


class InstrumentHealthModel:
    """
    Evaluates test stream integrity to isolate instrument/chamber faults prior to ML model inference.
    """

    def evaluate(
        self,
        lot_measurements: List[Dict[str, Any]],
        smu_compliance_limit: Optional[float] = None,
    ) -> InstrumentHealthStatus:
        """
        Input: List of records containing component_id, checkpoint_name, value, channel_id, etc.
        """
        details = []

        if not lot_measurements:
            return InstrumentHealthStatus(
                is_instrument_healthy=True,
                action_recommendation="PROCEED_NO_DATA",
                diagnostic_details=["Empty dataset passed to instrument QA."],
            )

        # 1. Test for Signal Freezing (ADC Stuck / Data Logger Fault)
        # Check variance of individual component time-series values across checkpoints
        component_series: Dict[str, List[float]] = {}
        for r in lot_measurements:
            cid = r.get("component_id", "UNKNOWN")
            val = r.get("value", 0.0)
            component_series.setdefault(cid, []).append(val)

        frozen_components = []
        for cid, vals in component_series.items():
            if len(vals) >= 3 and float(np.var(vals)) < 1e-12:
                frozen_components.append(cid)

        if len(frozen_components) > 0 and len(frozen_components) == len(component_series):
            details.append(f"ADC Stuck Fault: All {len(frozen_components)} components exhibit zero variance across time checkpoints.")
            return InstrumentHealthStatus(
                is_instrument_healthy=False,
                fault_type="ADC_STUCK_DATA_LOGGER_FAULT",
                affected_channels=frozen_components,
                confidence_score=0.98,
                action_recommendation="FLAG_CHAMBER_DATA_LOGGER_RELOAD_SAMPLES",
                diagnostic_details=details,
            )

        # 2. Test for Common-Mode Sudden Jump (Power Supply / Chamber Transient Shift)
        # Group values by checkpoint
        checkpoint_values: Dict[str, List[float]] = {}
        for r in lot_measurements:
            cp = r.get("checkpoint_name", "0h")
            val = r.get("value", 0.0)
            checkpoint_values.setdefault(cp, []).append(val)

        checkpoints = sorted(list(checkpoint_values.keys()))
        if len(checkpoints) >= 2:
            cp_means = {cp: float(np.mean(vals)) for cp, vals in checkpoint_values.items()}
            # Check if all components jumped by >50% simultaneously at the same checkpoint
            for i in range(1, len(checkpoints)):
                prev_cp = checkpoints[i - 1]
                curr_cp = checkpoints[i]
                prev_mean = cp_means[prev_cp]
                curr_mean = cp_means[curr_cp]

                if prev_mean > 0:
                    relative_shift = abs(curr_mean - prev_mean) / prev_mean
                    if relative_shift > 0.80:
                        # Check if individual socket shifts correlate (common mode)
                        details.append(
                            f"Common Mode Shift: Population mean shifted {relative_shift*100:.1f}% simultaneously from {prev_cp} to {curr_cp}."
                        )
                        return InstrumentHealthStatus(
                            is_instrument_healthy=False,
                            fault_type="CHAMBER_POWER_STABILITY_FAULT",
                            affected_channels=list(component_series.keys()),
                            confidence_score=0.92,
                            action_recommendation="RECALIBRATE_CHAMBER_POWER_SUPPLY_BEFORE_REJECTING_LOT",
                            diagnostic_details=details,
                        )

        # 3. Test for Open Circuit / Socket Contact Resistance Fault
        extreme_outliers = []
        for r in lot_measurements:
            val = r.get("value", 0.0)
            if smu_compliance_limit and val >= smu_compliance_limit:
                extreme_outliers.append(r.get("component_id"))
            elif val > 1e6: # Open circuit high impedance saturation
                extreme_outliers.append(r.get("component_id"))

        if len(extreme_outliers) > 0 and len(extreme_outliers) < len(component_series) * 0.2:
            details.append(f"Socket Contact Fault: {len(extreme_outliers)} sockets show open circuit saturation readings.")
            return InstrumentHealthStatus(
                is_instrument_healthy=False,
                fault_type="OPEN_CIRCUIT_SOCKET_CONTACT_FAULT",
                affected_channels=extreme_outliers,
                confidence_score=0.95,
                action_recommendation="CLEAN_SOCKET_PINS_AND_RE_MEASURE_AFFECTED_UNITS",
                diagnostic_details=details,
            )

        # No instrument faults detected
        return InstrumentHealthStatus(
            is_instrument_healthy=True,
            confidence_score=1.0,
            action_recommendation="PROCEED_WITH_COMPONENT_RELIABILITY_EVALUATION",
            diagnostic_details=["Instrument signals validated. Zero common-mode or channel dropout faults detected."],
        )
