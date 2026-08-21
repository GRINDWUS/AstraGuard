"""
AstraGuard SDK — Offline Command Line Interface (CLI)
=====================================================
Enables local offline execution of context resolution, data validation,
and screening analysis directly from terminal.

Usage:
  python -m astraguard_sdk.cli analyze dataset.csv
  python -m astraguard_sdk.cli resolve dataset.csv
  python -m astraguard_sdk.cli validate dataset.csv
"""

import sys
import os
import json
import argparse
from astraguard_sdk.client import AstraGuardClient


def main():
    parser = argparse.ArgumentParser(
        description="AstraGuard 2.4 Safe ATE Integration SDK CLI (Read-Only)"
    )
    parser.add_argument(
        "action",
        choices=["analyze", "resolve", "validate", "report"],
        help="Action to perform on input ATE data."
    )
    parser.add_argument(
        "file_path",
        help="Path to input ATE CSV/JSON data file."
    )
    parser.add_argument(
        "--operator",
        default="QA_OPERATOR_CLI",
        help="Operator ID for audit log tracking."
    )

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"❌ Error: Input file '{args.file_path}' does not exist.")
        sys.exit(1)

    client = AstraGuardClient(operator_id=args.operator)
    result = client.analyze(args.file_path)

    if args.action == "resolve":
        print("=" * 60)
        print("🔍 ASTRA GUARD SDK — CONTEXT RESOLUTION REPORT")
        print("=" * 60)
        print(f"Status:             {result.context_status}")
        print(f"Resolved Device:    {result.resolved_device_family}")
        print(f"Resolved Test:      {result.resolved_test_type}")
        print(f"Primary Parameter:  {result.resolved_primary_parameter}")
        print(f"Confidence Score:   {result.context_confidence:.2f}")
        print("=" * 60)

    elif args.action == "validate":
        print("=" * 60)
        print("🛡️ ASTRA GUARD SDK — DATA INTEGRITY & INSTRUMENT QA")
        print("=" * 60)
        print(f"Data Quality Score: {result.data_quality_score:.2f}")
        print(f"Instrument Status:  {result.instrument_health_status}")
        print(f"Processed Records:  {result.total_records_processed}")
        print("=" * 60)

    else:  # analyze or report
        print("=" * 60)
        print("🚀 ASTRA GUARD SDK — READ-ONLY SCREENING ANALYSIS REPORT")
        print("=" * 60)
        print(f"Session ID:         {result.session.session_id}")
        print(f"Audit ID:           {result.audit_id}")
        print(f"Resolved Device:    {result.resolved_device_family}")
        print(f"Context Status:     {result.context_status}")
        print(f"Instrument Health:  {result.instrument_health_status}")
        print(f"Recommendation:     {result.recommendation}")
        print(f"Component Tiers:    {result.components_summary}")
        print("\nDecision Reasons:")
        for r in result.reasons:
            print(f"  • {r}")
        print("\nEvidence Trail:")
        for e in result.evidence_trail:
            print(f"  {e}")
        print("=" * 60)


if __name__ == "__main__":
    main()
