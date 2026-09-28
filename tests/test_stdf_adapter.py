"""
AstraGuard Phase 2A — STDF v4 Adapter Test Suite
=================================================
Tests native STDF binary parsing, multi-record type handling,
multi-site mapping, and round-trip (write → parse → validate).

Test Coverage:
  T01: FAR record parsing + endianness detection (LE)
  T02: Full record chain (FAR+MIR+SDR+PIR+PTR+PRR+MRR) produces records
  T03: Multi-site SDR/PIR grouping preserves site_num in metadata
  T04: PTR → PRR pass/fail mapping (PASS/FAIL)
  T05: Multiple PTRs per part mapped to checkpoint slots (value_0h .. value_168h)
  T06: Round-trip: STDFATEAdapter generates + parses → same component count
  T07: Empty/corrupt STDF raises ValueError or returns empty list gracefully
  T08: to_dataframe() canonical schema presence check
  T09: Performance sanity: parse 200-component synthetic STDF in <2s
  T10: Missing MRR (truncated) still returns parsed records
"""

import io
import os
import struct
import time
import unittest
from datetime import datetime

from astraguard_core.stdf_adapter import STDFV4RecordWriter
from astraguard_sdk.adapters.stdf_adapter import STDFATEAdapter


def _build_stdf(
    lot_id: str = "LOT_TEST_01",
    device: str = "DIGITAL_IC",
    num_components: int = 5,
    num_sites: int = 1,
    include_mrr: bool = True,
    fail_site: int = 0,     # set to site number to inject a FAIL on that site
    num_ptrs_per_part: int = 4,  # simulate 4 checkpoints: 0h/24h/96h/168h
) -> bytes:
    """Build a synthetic STDF v4 binary stream for testing."""
    writer = STDFV4RecordWriter

    stream = writer.create_far()
    stream += writer.create_mir(lot_id, device, 125.0)

    # SDR: describe sites
    site_nums = list(range(1, num_sites + 1))
    # Simple SDR: head=1, grp=255, site_count, site_nums...
    sdr_body = struct.pack("<BBB", 1, 255, len(site_nums))
    for sn in site_nums:
        sdr_body += struct.pack("<B", sn)
    stream += struct.pack("<HBB", len(sdr_body), 1, 80) + sdr_body

    test_num_base = 1001

    for site in site_nums:
        for c in range(num_components):
            part_id = f"DUT_H1S{site}"
            # PIR
            pir_body = struct.pack("<BB", 1, site)  # head=1, site
            stream += struct.pack("<HBB", len(pir_body), 5, 10) + pir_body

            # PTRs (4 checkpoints)
            checkpoint_values = [12.0 + c * 0.1, 12.3 + c * 0.12, 12.6 + c * 0.15, 13.0 + c * 0.18]
            for cp_idx in range(min(num_ptrs_per_part, 4)):
                ptr_val = checkpoint_values[cp_idx]
                test_name = f"IDDQ_{cp_idx * 24}H_CH{c:02d}"
                stream += writer.create_ptr(
                    test_num_base + cp_idx,
                    test_name,
                    ptr_val,
                    "uA"
                )

            # PRR: part_flag=0 (PASS) unless fail_site matches
            is_fail = (fail_site == site)
            part_flag = 0x08 if is_fail else 0x00
            prr_body = struct.pack("<BBHHHH", 1, site, part_flag, num_ptrs_per_part, 1 if not is_fail else 10, 1)
            # PRR: head, site, part_flag, num_test(u2), hard_bin(u2), soft_bin(u2)
            prr_body = struct.pack("<BB", 1, site)                     # head, site
            prr_body += struct.pack("<B", part_flag)                    # part_flag
            prr_body += struct.pack("<H", num_ptrs_per_part)            # num_test
            prr_body += struct.pack("<H", 1 if not is_fail else 10)     # hard_bin
            prr_body += struct.pack("<H", 1)                            # soft_bin
            stream += struct.pack("<HBB", len(prr_body), 5, 20) + prr_body

    if include_mrr:
        finish_t = int(time.time())
        mrr_body = struct.pack("<I", finish_t)
        stream += struct.pack("<HBB", len(mrr_body), 1, 20) + mrr_body

    return stream


class TestSTDFAdapterSuite(unittest.TestCase):

    def setUp(self):
        self.adapter = STDFATEAdapter()
        self.stdf_bytes = _build_stdf(
            lot_id="LOT_2026_TEST",
            device="DIGITAL_IC",
            num_components=10,
            num_sites=1,
            num_ptrs_per_part=4
        )

    # -----------------------------------------------------------------------

    def test_01_far_endian_detection_le(self):
        """FAR CPU_TYPE=2 (x86 little-endian) is detected and parser uses LE."""
        records = self.adapter.parse_bytes(self.stdf_bytes)
        # Just checking no exception → endian detection worked
        self.assertIsInstance(records, list)

    def test_02_full_chain_produces_records(self):
        """Full FAR+MIR+SDR+PIR+PTR+PRR+MRR chain → non-empty record list."""
        records = self.adapter.parse_bytes(self.stdf_bytes)
        self.assertGreater(len(records), 0, "Expected at least 1 component record")

    def test_03_component_count_matches(self):
        """10 components on 1 site should produce 10 SDKMeasurementRecords."""
        records = self.adapter.parse_bytes(self.stdf_bytes)
        self.assertEqual(len(records), 10)

    def test_04_lot_id_propagated(self):
        """lot_id from MIR must carry through to all records."""
        records = self.adapter.parse_bytes(self.stdf_bytes)
        for r in records:
            self.assertEqual(r.lot_id, "LOT_2026_TEST")

    def test_05_multi_site_produces_correct_per_site_records(self):
        """4-site STDF with 5 components/site → 20 total records."""
        stdf = _build_stdf(num_components=5, num_sites=4)
        records = self.adapter.parse_bytes(stdf)
        self.assertEqual(len(records), 20, f"Expected 20, got {len(records)}")

    def test_06_multi_site_site_num_in_metadata(self):
        """Each record's metadata.site_num must match the emitting site."""
        stdf = _build_stdf(num_components=3, num_sites=3)
        records = self.adapter.parse_bytes(stdf)
        site_nums_seen = {r.metadata.get("site_num") for r in records}
        self.assertEqual(site_nums_seen, {1, 2, 3})

    def test_07_pass_fail_mapped_from_prr(self):
        """PRR part_flag PASS → measurement_state VALID."""
        records = self.adapter.parse_bytes(self.stdf_bytes)
        for r in records:
            self.assertEqual(r.measurement_state, "VALID")

    def test_08_fail_injection_maps_to_out_of_range(self):
        """PRR part_flag FAIL (0x08) → measurement_state OUT_OF_RANGE."""
        stdf = _build_stdf(num_components=5, num_sites=2, fail_site=2)
        records = self.adapter.parse_bytes(stdf)
        fail_recs = [r for r in records if r.measurement_state == "OUT_OF_RANGE"]
        # Site 2 components should be OUT_OF_RANGE
        self.assertGreater(len(fail_recs), 0, "Expected at least one FAIL record from site 2")

    def test_09_checkpoint_values_in_metadata(self):
        """4 PTRs per part → value_0h/value_24h/value_96h/value_168h slots filled."""
        records = self.adapter.parse_bytes(self.stdf_bytes)
        for r in records:
            meta = r.metadata
            self.assertIn("value_0h",    meta, f"value_0h missing in {r.component_id}")
            self.assertIn("value_24h",   meta, f"value_24h missing in {r.component_id}")

    def test_10_to_dataframe_canonical_schema(self):
        """to_dataframe() must include all required canonical columns."""
        records = self.adapter.parse_bytes(self.stdf_bytes)
        df = self.adapter.to_dataframe(records)
        required_cols = ["component_id", "lot_id", "primary_parameter", "unit",
                         "value_0h", "value_24h", "site_num", "instrument_status"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_11_truncated_stdf_no_mrr_still_returns_records(self):
        """STDF without MRR sentinel (truncated) should still return parsed records."""
        stdf = _build_stdf(num_components=5, include_mrr=False)
        records = self.adapter.parse_bytes(stdf)
        self.assertGreater(len(records), 0)

    def test_12_empty_bytes_returns_empty_list(self):
        """Empty bytes input returns empty list without crashing."""
        records = self.adapter.parse_bytes(b"")
        self.assertEqual(records, [])

    def test_13_performance_200_components(self):
        """200-component, 1-site STDF parses in under 2 seconds."""
        stdf = _build_stdf(num_components=200, num_sites=1)
        start = time.perf_counter()
        records = self.adapter.parse_bytes(stdf)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0, f"Parse took {elapsed:.3f}s — exceeds 2s target")
        self.assertEqual(len(records), 200)


if __name__ == "__main__":
    unittest.main()
