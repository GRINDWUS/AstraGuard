"""
AstraGuard 2.2 — Explicit Metadata & Schema Resolver
Parses Level 1 (Explicit Metadata) and Level 2 (Measurement Schema Mapping).
Now features dynamic confidence calibration, ambiguity detection, and operator confirmation flags.
"""

from typing import Dict, List, Any, Optional
from astraguard_core.context_resolver.schema import (
    TestContext,
    MeasurementRecord,
    TestIdentityResolutionResult,
    IdentificationSource,
    ResolutionStatus,
)
from astraguard_core.context_resolver.profiles import ProfileRegistry


class ExplicitMetadataParser:
    def __init__(self):
        self.registry = ProfileRegistry()

    def resolve(
        self,
        test_context: Optional[TestContext] = None,
        observed_parameters: Optional[List[str]] = None,
        metadata_dict: Optional[Dict[str, Any]] = None,
    ) -> TestIdentityResolutionResult:
        """
        Attempts Level 1 (Explicit metadata) and Level 2 (Schema mapping) resolution.
        Calculates calibrated confidence and assigns ResolutionStatus (KNOWN, PARTIAL, AMBIGUOUS, UNKNOWN).
        """
        notes = []
        observed_params = observed_parameters or []

        explicit_device = None
        explicit_test = None

        if test_context:
            explicit_device = test_context.device_metadata.device_family
            explicit_test = test_context.test_metadata.test_type
        elif metadata_dict:
            explicit_device = metadata_dict.get("device_family") or metadata_dict.get("device_type")
            explicit_test = metadata_dict.get("test_type") or metadata_dict.get("procedure_type")

        # Level 1 Check: Explicit Device Family Metadata
        if explicit_device and explicit_device in self.registry.list_device_families():
            dev_profile = self.registry.get_device_profile(explicit_device)
            test_type = explicit_test if (explicit_test and explicit_test in self.registry.list_test_types()) else "THERMAL_BURN_IN"
            test_profile = self.registry.get_test_profile(test_type)

            missing = [p for p in dev_profile.expected_parameters if p not in observed_params] if observed_params else []
            extra = [p for p in observed_params if p not in dev_profile.expected_parameters] if observed_params else []

            # Calibrated Level 1 Confidence
            confidence = 1.0 if explicit_test else 0.90
            status = ResolutionStatus.KNOWN_CONTEXT if explicit_test else ResolutionStatus.PARTIAL_CONTEXT
            requires_operator = False

            if missing and observed_params:
                confidence -= 0.05 * (len(missing) / float(len(dev_profile.expected_parameters)))
                status = ResolutionStatus.PARTIAL_CONTEXT
                notes.append(f"Level 1 Resolution: Explicit metadata found for {explicit_device} with {len(missing)} unobserved parameters.")
            else:
                notes.append(f"Level 1 Resolution: Confirmed explicit metadata for {explicit_device} / {test_type}.")

            return TestIdentityResolutionResult(
                resolved_device_family=dev_profile.device_family,
                resolved_test_type=test_profile.test_type,
                primary_parameter=dev_profile.primary_parameter,
                confidence_score=round(max(0.0, min(1.0, confidence)), 3),
                identification_source=IdentificationSource.EXPLICIT_METADATA,
                status=status,
                requires_operator_confirmation=requires_operator,
                expected_parameters=dev_profile.expected_parameters,
                missing_parameters=missing,
                unexpected_parameters=extra,
                active_profile_name=dev_profile.display_name,
                notes=notes,
            )

        # Level 2 Check: Schema/Parameter Matching & Ambiguity Check
        if observed_params:
            family_scores = {}
            family_missing = {}
            family_extra = {}

            for dev_fam in self.registry.list_device_families():
                profile = self.registry.get_device_profile(dev_fam)
                expected = set(profile.expected_parameters)
                observed_set = set(observed_params)

                intersection = expected.intersection(observed_set)
                if intersection:
                    # Primary parameter match bonus
                    primary_bonus = 0.50 if profile.primary_parameter in observed_set else 0.0
                    overlap_ratio = len(intersection) / float(len(expected))
                    score = min(1.0, primary_bonus + (0.50 * overlap_ratio))
                    
                    family_scores[dev_fam] = score
                    family_missing[dev_fam] = list(expected - observed_set)
                    family_extra[dev_fam] = list(observed_set - expected)

            if family_scores:
                sorted_families = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
                top_family, top_score = sorted_families[0]
                second_score = sorted_families[1][1] if len(sorted_families) > 1 else 0.0

                ambiguity_margin = top_score - second_score

                # Ambiguity Guard: if top score is close to second score, flag as AMBIGUOUS_CONTEXT
                if top_score >= 0.40 and ambiguity_margin < 0.10 and len(sorted_families) > 1:
                    dev_profile = self.registry.get_device_profile(top_family)
                    test_type = explicit_test or "THERMAL_BURN_IN"
                    test_profile = self.registry.get_test_profile(test_type)

                    notes.append(
                        f"Level 2 Ambiguity Alert: Top match '{top_family}' (score {top_score:.2f}) is ambiguous with '{sorted_families[1][0]}' (score {second_score:.2f})."
                    )

                    return TestIdentityResolutionResult(
                        resolved_device_family="UNKNOWN",
                        resolved_test_type=test_profile.test_type,
                        primary_parameter="UNKNOWN",
                        confidence_score=round(max(0.0, top_score * 0.5), 3),
                        identification_source=IdentificationSource.METADATA_MAPPING,
                        status=ResolutionStatus.AMBIGUOUS_CONTEXT,
                        requires_operator_confirmation=True,
                        expected_parameters=[],
                        missing_parameters=family_missing[top_family],
                        unexpected_parameters=observed_params,
                        active_profile_name="AMBIGUOUS_DEVICE_PROFILE",
                        notes=notes,
                    )

                if top_score >= 0.40:
                    dev_profile = self.registry.get_device_profile(top_family)
                    test_type = explicit_test or "THERMAL_BURN_IN"
                    test_profile = self.registry.get_test_profile(test_type)

                    calibrated_conf = min(0.95, top_score * 0.95)
                    status = ResolutionStatus.KNOWN_CONTEXT if calibrated_conf >= 0.75 else ResolutionStatus.PARTIAL_CONTEXT

                    notes.append(f"Level 2 Resolution: Parameter schema match for '{top_family}' with calibrated score {calibrated_conf:.2f}.")

                    return TestIdentityResolutionResult(
                        resolved_device_family=dev_profile.device_family,
                        resolved_test_type=test_profile.test_type,
                        primary_parameter=dev_profile.primary_parameter,
                        confidence_score=round(calibrated_conf, 3),
                        identification_source=IdentificationSource.METADATA_MAPPING,
                        status=status,
                        requires_operator_confirmation=False,
                        expected_parameters=dev_profile.expected_parameters,
                        missing_parameters=family_missing[top_family],
                        unexpected_parameters=family_extra[top_family],
                        active_profile_name=dev_profile.display_name,
                        notes=notes,
                    )

        # Fallback to UNKNOWN_CONTEXT requiring operator confirmation
        test_type = explicit_test or "THERMAL_BURN_IN"
        test_profile = self.registry.get_test_profile(test_type)
        notes.append("Explicit & Schema Resolution failed. Context remains UNKNOWN, requiring operator confirmation or Level 3 inference.")

        return TestIdentityResolutionResult(
            resolved_device_family="UNKNOWN",
            resolved_test_type=test_profile.test_type,
            primary_parameter="UNKNOWN",
            confidence_score=0.15,
            identification_source=IdentificationSource.UNKNOWN,
            status=ResolutionStatus.UNKNOWN_CONTEXT,
            requires_operator_confirmation=True,
            expected_parameters=[],
            missing_parameters=[],
            unexpected_parameters=observed_params,
            active_profile_name="UNKNOWN_DEVICE_PROFILE",
            notes=notes,
        )
