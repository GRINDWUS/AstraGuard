"""
AstraGuard Phase 2A — Multi-Site Decision Fusion Test Suite
===========================================================
Tests conservative tier selection, confidence weighting, and
evidence trail generation across 1–4 probe site configurations.

Test Coverage:
  M01: Single site → fused tier equals that site's tier
  M02: All GREEN → final tier GREEN
  M03: One YELLOW among GREENs → final tier YELLOW (conservative)
  M04: One RED among GREENs → final tier RED
  M05: HOLD propagates unconditionally
  M06: Driving sites correctly identified (not all sites)
  M07: Confidence is minimum-biased (< plain mean)
  M08: Evidence trail contains per-site lines
  M09: Component summary aggregated across sites
  M10: fuse_tier_only() utility works correctly
  M11: Empty input raises ValueError
"""

import unittest
from unittest.mock import MagicMock
from astraguard_sdk.adapters.stdf_multi_site import MultiSiteDecisionFusion, FusedDecision
from astraguard_sdk.schema import SDKAnalysisResult, AnalysisSessionMetadata


def _make_result(
    recommendation: str,
    confidence: float = 0.95,
    green: int = 10, yellow: int = 0, red: int = 0,
    device: str = "DIGITAL_IC",
) -> SDKAnalysisResult:
    """Helper: create a minimal SDKAnalysisResult for fusion tests."""
    total = green + yellow + red
    session = AnalysisSessionMetadata(operator_id="TEST_OPERATOR")
    return SDKAnalysisResult(
        session=session,
        is_execution_allowed=True,
        context_status="EXPLICIT_MATCH",
        resolved_device_family=device,
        resolved_test_type="BURN_IN",
        resolved_primary_parameter="IDDQ",
        context_confidence=confidence,
        data_quality_score=0.98,
        instrument_health_status="INSTRUMENT_HEALTHY",
        total_records_processed=total,
        recommendation=recommendation,
        evidence_trail=[
            f"[METRICS] Evaluated {total} components",
            f"[BENCHMARK] Chamber Hours Saved: 63.50%",
        ],
        reasons=["Test reason"],
        components_summary={"GREEN": green, "YELLOW": yellow, "RED": red},
    )


class TestMultiSiteDecisionFusion(unittest.TestCase):

    def setUp(self):
        self.fusion = MultiSiteDecisionFusion()

    # -----------------------------------------------------------------------

    def test_m01_single_site_passes_through(self):
        """Single-site result → fused tier equals that site's tier."""
        result = _make_result("GREEN_NORMAL_CANDIDATE", confidence=0.97)
        fused = self.fusion.fuse({1: result})
        self.assertEqual(fused.final_tier, "GREEN")

    def test_m02_all_green_yields_green(self):
        """All 4 sites GREEN → final tier GREEN."""
        site_results = {s: _make_result("GREEN_NORMAL_CANDIDATE") for s in [1, 2, 3, 4]}
        fused = self.fusion.fuse(site_results)
        self.assertEqual(fused.final_tier, "GREEN")

    def test_m03_one_yellow_upgrades_to_yellow(self):
        """Sites: GREEN, GREEN, YELLOW, GREEN → conservative decision = YELLOW."""
        site_results = {
            1: _make_result("GREEN_NORMAL_CANDIDATE"),
            2: _make_result("GREEN_NORMAL_CANDIDATE"),
            3: _make_result("YELLOW_REVIEW", yellow=3, green=7),
            4: _make_result("GREEN_NORMAL_CANDIDATE"),
        }
        fused = self.fusion.fuse(site_results)
        self.assertEqual(fused.final_tier, "YELLOW")

    def test_m04_one_red_upgrades_to_red(self):
        """Sites: GREEN, GREEN, RED, GREEN → conservative decision = RED."""
        site_results = {
            1: _make_result("GREEN_NORMAL_CANDIDATE"),
            2: _make_result("GREEN_NORMAL_CANDIDATE"),
            3: _make_result("RED_HIGH_RISK", red=5, green=5),
            4: _make_result("GREEN_NORMAL_CANDIDATE"),
        }
        fused = self.fusion.fuse(site_results)
        self.assertEqual(fused.final_tier, "RED")

    def test_m05_hold_propagates_unconditionally(self):
        """Any HOLD site → final = HOLD (worst tier)."""
        site_results = {
            1: _make_result("GREEN_NORMAL_CANDIDATE"),
            2: _make_result("HOLD_OPERATOR_REVIEW", green=0, yellow=0, red=0),
        }
        fused = self.fusion.fuse(site_results)
        self.assertEqual(fused.final_tier, "HOLD")

    def test_m06_driving_site_identified_correctly(self):
        """The site that drove conservative decision is in driving_sites."""
        site_results = {
            1: _make_result("GREEN_NORMAL_CANDIDATE"),
            2: _make_result("YELLOW_REVIEW", yellow=5, green=5),
            3: _make_result("GREEN_NORMAL_CANDIDATE"),
        }
        fused = self.fusion.fuse(site_results)
        self.assertIn(2, fused.driving_sites, "Site 2 (YELLOW) should be the driving site")
        self.assertNotIn(1, fused.driving_sites)
        self.assertNotIn(3, fused.driving_sites)

    def test_m07_confidence_is_minimum_biased(self):
        """Fused confidence should be below plain mean (min-biased blend)."""
        site_results = {
            1: _make_result("GREEN_NORMAL_CANDIDATE", confidence=0.99),
            2: _make_result("GREEN_NORMAL_CANDIDATE", confidence=0.60),
        }
        fused = self.fusion.fuse(site_results)
        plain_mean = (0.99 + 0.60) / 2   # 0.795

        # Min-biased blend = 0.70 * min + 0.30 * mean = 0.70*0.60 + 0.30*0.795 = 0.6585
        expected = 0.70 * 0.60 + 0.30 * plain_mean
        self.assertAlmostEqual(fused.confidence, expected, places=3)
        self.assertLess(fused.confidence, plain_mean)

    def test_m08_evidence_trail_has_per_site_lines(self):
        """Evidence trail must include [SITE N] lines for each site."""
        site_results = {s: _make_result("GREEN_NORMAL_CANDIDATE") for s in [1, 2, 3]}
        fused = self.fusion.fuse(site_results)
        for site in [1, 2, 3]:
            site_lines = [e for e in fused.evidence_trail if f"[SITE {site}]" in e]
            self.assertGreater(len(site_lines), 0, f"Missing [SITE {site}] in evidence trail")

    def test_m09_component_summary_aggregated(self):
        """Component counts summed across all sites."""
        site_results = {
            1: _make_result("GREEN_NORMAL_CANDIDATE", green=10, yellow=0, red=0),
            2: _make_result("YELLOW_REVIEW",          green=7,  yellow=3, red=0),
            3: _make_result("RED_HIGH_RISK",           green=5,  yellow=2, red=3),
        }
        fused = self.fusion.fuse(site_results)
        self.assertEqual(fused.components_summary["GREEN"],  22)
        self.assertEqual(fused.components_summary["YELLOW"],  5)
        self.assertEqual(fused.components_summary["RED"],     3)

    def test_m10_fuse_tier_only_utility(self):
        """fuse_tier_only() returns worst tier from flat list."""
        tiers = [
            "GREEN_NORMAL_CANDIDATE",
            "YELLOW_REVIEW",
            "GREEN_NORMAL_CANDIDATE",
        ]
        result = self.fusion.fuse_tier_only(tiers)
        self.assertEqual(result, "YELLOW_REVIEW")

    def test_m11_empty_input_raises(self):
        """Empty site_results dict → ValueError."""
        with self.assertRaises(ValueError):
            self.fusion.fuse({})


if __name__ == "__main__":
    unittest.main()
