"""
AstraGuard SDK — Multi-Site Decision Fusion Engine
===================================================
Combines per-site analysis results from multi-site probe headstations
into a single, conservative, confidence-weighted final decision.

Real ATE probe stations run 1–4 independent sites simultaneously.
Each site has its own thermal zone, SMU, and calibration state.
A component flagged on ANY site should route to YELLOW/RED — this
module enforces that safety-first conservative fusion policy.

Design:
  - Per-site decisions from SDKAnalysisResult objects
  - Conservative tier selection: worst site wins
  - Confidence = weighted average (minimum-biased)
  - Evidence trail records which site drove the conservative decision

Usage:
  from astraguard_sdk.adapters.stdf_multi_site import MultiSiteDecisionFusion
  fusion = MultiSiteDecisionFusion()
  fused  = fusion.fuse(site_results={1: result_s1, 2: result_s2})
  print(fused.final_tier)       # 'GREEN' | 'YELLOW' | 'RED' | 'HOLD'
  print(fused.confidence)       # weighted confidence score
  print(fused.driving_sites)    # e.g. [2] — site that drove conservative call
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field
from astraguard_sdk.schema import SDKAnalysisResult

# Tier rank for conservative comparison (higher = worse)
_TIER_RANK = {
    "GREEN_NORMAL_CANDIDATE": 0,
    "YELLOW_REVIEW":          1,
    "RED_HIGH_RISK":          2,
    "HOLD_OPERATOR_REVIEW":   3,
}

# Human-readable tier short labels
_TIER_LABEL = {
    "GREEN_NORMAL_CANDIDATE": "GREEN",
    "YELLOW_REVIEW":          "YELLOW",
    "RED_HIGH_RISK":          "RED",
    "HOLD_OPERATOR_REVIEW":   "HOLD",
}


class FusedDecision(BaseModel):
    """
    Result of fusing per-site SDK analysis results into a single lot-level decision.
    """
    fused_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Input summary
    site_count: int
    site_results: Dict[int, str]    # {site_num: tier_label}
    site_confidences: Dict[int, float]  # {site_num: confidence}

    # Fused output
    final_tier: str                 # 'GREEN' | 'YELLOW' | 'RED' | 'HOLD'
    final_recommendation: str       # Full recommendation string
    confidence: float               # Weighted confidence (minimum-biased)
    driving_sites: List[int]        # Sites that drove conservative decision
    reasoning: str
    evidence_trail: List[str]       # Audit-ready evidence lines

    # Component-level aggregate
    total_components: int
    components_summary: Dict[str, int]   # {'GREEN': n, 'YELLOW': n, 'RED': n}
    chamber_hours_saved_pct: float


class MultiSiteDecisionFusion:
    """
    Conservative multi-site decision fusion engine.

    Fusion Rules:
      1. Tier: Take the worst tier across all sites (any site RED → final RED).
      2. Confidence: Minimum of site confidences (safety-biased, not mean).
      3. Driving site: Track which site(s) forced upgrade to worse tier.
      4. HOLD propagates unconditionally — any site HOLD → final HOLD.

    These rules mirror MIL-STD-883 Method 1015 conservative screening logic.
    """

    def fuse(self, site_results: Dict[int, SDKAnalysisResult]) -> FusedDecision:
        """
        Fuse per-site SDKAnalysisResult objects into a single FusedDecision.

        Args:
            site_results: Mapping of {site_number: SDKAnalysisResult}.
                          Must contain at least 1 entry.

        Returns:
            FusedDecision with conservative tier, confidence, and audit trail.
        """
        if not site_results:
            raise ValueError("MultiSiteDecisionFusion.fuse() requires at least 1 site result.")

        evidence: List[str] = []
        site_tiers: Dict[int, str] = {}
        site_confs: Dict[int, float] = {}

        # Aggregate component counts across all sites
        total_green = total_yellow = total_red = 0
        total_comps = 0
        total_chamber_saved = 0.0

        for site_num, result in sorted(site_results.items()):
            rec = result.recommendation
            tier_label = _TIER_LABEL.get(rec, "HOLD")
            site_tiers[site_num] = tier_label
            site_confs[site_num] = result.context_confidence

            # Aggregate component summaries
            cs = result.components_summary
            g = cs.get("GREEN", 0)
            y = cs.get("YELLOW", 0)
            r = cs.get("RED", 0)
            total_green  += g
            total_yellow += y
            total_red    += r
            total_comps  += result.total_records_processed

            # Extract chamber hours saved from evidence trail
            for ev_line in result.evidence_trail:
                if "Chamber Hours Saved" in ev_line:
                    try:
                        pct = float(ev_line.split(":")[1].strip().replace("%", ""))
                        total_chamber_saved += pct
                    except (ValueError, IndexError):
                        pass

            evidence.append(
                f"[SITE {site_num}] Tier={tier_label} | Conf={result.context_confidence:.2f} | "
                f"Device={result.resolved_device_family} | "
                f"G:{g} Y:{y} R:{r}"
            )

        # --- Conservative tier selection ---
        worst_tier_key = max(
            site_results.keys(),
            key=lambda s: _TIER_RANK.get(site_results[s].recommendation, 3)
        )
        worst_result     = site_results[worst_tier_key]
        worst_tier_full  = worst_result.recommendation
        worst_tier_label = _TIER_LABEL.get(worst_tier_full, "HOLD")

        # Find all sites that share the worst tier (driving sites)
        driving_sites = [
            s for s, r in site_results.items()
            if _TIER_RANK.get(r.recommendation, 3) == _TIER_RANK.get(worst_tier_full, 3)
        ]

        # --- Confidence: minimum-biased weighted average ---
        confs = list(site_confs.values())
        min_conf = min(confs)
        mean_conf = sum(confs) / len(confs)
        # Blend: 70% min (safety-first) + 30% mean
        fused_confidence = round(0.70 * min_conf + 0.30 * mean_conf, 4)

        # --- Reasoning ---
        if len(driving_sites) == len(site_results):
            reasoning = (
                f"All {len(site_results)} probe sites agree on {worst_tier_label} tier. "
                f"Fused confidence: {fused_confidence:.2f}."
            )
        else:
            reasoning = (
                f"Conservative fusion: site(s) {driving_sites} drove upgrade to "
                f"{worst_tier_label} (worst among {len(site_results)} sites). "
                f"Other sites: {[_TIER_LABEL.get(site_results[s].recommendation) for s in site_results if s not in driving_sites]}. "
                f"Fused confidence: {fused_confidence:.2f}."
            )

        avg_chamber_saved = round(total_chamber_saved / max(len(site_results), 1), 2)

        evidence.append(f"[FUSION] Final Tier: {worst_tier_label} | Driving Sites: {driving_sites}")
        evidence.append(f"[FUSION] Fused Confidence: {fused_confidence:.2f} (min-biased 70/30 blend)")
        evidence.append(f"[FUSION] Total Components Fused: {total_comps} (G:{total_green} Y:{total_yellow} R:{total_red})")

        return FusedDecision(
            site_count=len(site_results),
            site_results=site_tiers,
            site_confidences=site_confs,
            final_tier=worst_tier_label,
            final_recommendation=worst_tier_full,
            confidence=fused_confidence,
            driving_sites=driving_sites,
            reasoning=reasoning,
            evidence_trail=evidence,
            total_components=total_comps,
            components_summary={
                "GREEN":  total_green,
                "YELLOW": total_yellow,
                "RED":    total_red,
            },
            chamber_hours_saved_pct=avg_chamber_saved,
        )

    def fuse_tier_only(self, tiers: List[str]) -> str:
        """
        Lightweight utility: fuse a flat list of tier strings conservatively.

        Args:
            tiers: e.g. ['GREEN_NORMAL_CANDIDATE', 'YELLOW_REVIEW', 'GREEN_NORMAL_CANDIDATE']

        Returns:
            The most conservative tier string.
        """
        if not tiers:
            return "HOLD_OPERATOR_REVIEW"
        return max(tiers, key=lambda t: _TIER_RANK.get(t, 3))
