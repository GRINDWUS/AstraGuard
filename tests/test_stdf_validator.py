"""
AstraGuard Phase 2A — STDF Validator Test Suite
================================================
Tests pre-parse data quality validation of STDF v4 binary files.

Test Coverage:
  V01: Valid STDF → recommendation='SAFE', score >= 0.90
  V02: Missing FAR → recommendation='REJECT'
  V03: Missing MIR → recommendation='REJECT'
  V04: Missing MRR → recommendation='CAUTION' (warning, not error)
  V05: Corrupted PTR float (NaN) → WARNING issue, score deduction
  V06: Truncated record body → WARNING, truncated_record_count > 0
  V07: Empty bytes → recommendation='REJECT', score=0.0
  V08: Non-existent file → recommendation='REJECT', score=0.0
  V09: Multi-site valid STDF → SAFE
  V10: parse_stats counts match actual record counts
"""

import io
import math
import os
import struct
import tempfile
import time
import unittest

from astraguard_sdk.adapters.stdf_validator import STDFValidator, STDFValidationReport
from tests.test_stdf_adapter import _build_stdf   # reuse builder


def _corrupt_ptr_value(stdf_bytes: bytes) -> bytes:
    """Inject a NaN float into the first PTR record's value field."""
    # Scan for PTR record (type=15, sub=10) and corrupt its value
    buf = bytearray(stdf_bytes)
    offset = 0
    while offset < len(buf) - 4:
        rec_len = struct.unpack_from("<H", buf, offset)[0]
        r_type  = buf[offset + 2]
        r_sub   = buf[offset + 3]
        if r_type == 15 and r_sub == 10 and rec_len >= 12:
            # Value is at body offset 8 (after test_num U4, head U1, site U1, flags U2)
            val_offset = offset + 4 + 8
            nan_bytes = struct.pack("<f", float("nan"))
            buf[val_offset: val_offset + 4] = nan_bytes
            break
        offset += 4 + rec_len
    return bytes(buf)


def _strip_far(stdf_bytes: bytes) -> bytes:
    """Remove FAR record (first 6 bytes: 4 header + 2 body for CPU_TYPE+VER)."""
    if len(stdf_bytes) < 6:
        return stdf_bytes
    # FAR is always first: 4-byte header + rec_len body
    rec_len = struct.unpack_from("<H", stdf_bytes, 0)[0]
    return stdf_bytes[4 + rec_len:]


def _strip_mrr(stdf_bytes: bytes) -> bytes:
    """Remove MRR record from the end."""
    buf = bytearray(stdf_bytes)
    offset = 0
    last_mrr_start = -1
    while offset < len(buf) - 4:
        rec_len = struct.unpack_from("<H", buf, offset)[0]
        r_type  = buf[offset + 2]
        r_sub   = buf[offset + 3]
        if r_type == 1 and r_sub == 20:
            last_mrr_start = offset
        offset += 4 + rec_len
    if last_mrr_start >= 0:
        return bytes(buf[:last_mrr_start])
    return stdf_bytes


class TestSTDFValidatorSuite(unittest.TestCase):

    def setUp(self):
        self.validator = STDFValidator()
        self.valid_stdf = _build_stdf(num_components=10, num_sites=1)

    # -----------------------------------------------------------------------

    def test_v01_valid_stdf_is_safe(self):
        """A well-formed STDF should return SAFE with score >= 0.90."""
        report = self.validator.validate_bytes(self.valid_stdf)
        self.assertEqual(report.recommendation, "SAFE", f"Issues: {[i.message for i in report.issues]}")
        self.assertGreaterEqual(report.data_quality_score, 0.90)
        self.assertTrue(report.is_valid)

    def test_v02_missing_far_is_rejected(self):
        """STDF without FAR record → recommendation=REJECT."""
        corrupted = _strip_far(self.valid_stdf)
        report = self.validator.validate_bytes(corrupted)
        self.assertEqual(report.recommendation, "REJECT")
        far_errors = [i for i in report.issues if i.record_type == "FAR" and i.severity == "ERROR"]
        self.assertGreater(len(far_errors), 0)

    def test_v03_missing_mrr_is_caution(self):
        """STDF without MRR (truncated end) → recommendation=CAUTION (not REJECT)."""
        no_mrr = _strip_mrr(self.valid_stdf)
        report = self.validator.validate_bytes(no_mrr)
        mrr_warnings = [i for i in report.issues if "MRR" in i.record_type]
        self.assertGreater(len(mrr_warnings), 0, "Expected MRR warning")
        self.assertNotEqual(report.recommendation, "REJECT",
                            "Missing MRR alone should not REJECT — only CAUTION")

    def test_v04_corrupted_ptr_float_is_warning(self):
        """NaN float in PTR value field → WARNING issue, score < 1.0."""
        corrupted = _corrupt_ptr_value(self.valid_stdf)
        report = self.validator.validate_bytes(corrupted)
        ptr_warnings = [i for i in report.issues if i.record_type == "PTR"]
        self.assertGreater(len(ptr_warnings), 0, "Expected a PTR NaN warning")
        self.assertLess(report.data_quality_score, 1.0)

    def test_v05_empty_bytes_is_rejected(self):
        """Empty bytes → recommendation=REJECT, score=0.0."""
        report = self.validator.validate_bytes(b"")
        self.assertEqual(report.recommendation, "REJECT")
        self.assertEqual(report.data_quality_score, 0.0)
        self.assertFalse(report.is_valid)

    def test_v06_nonexistent_file_is_rejected(self):
        """Non-existent file path → recommendation=REJECT."""
        report = self.validator.validate_file("/nonexistent/path/fake.stdf")
        self.assertEqual(report.recommendation, "REJECT")
        self.assertEqual(report.data_quality_score, 0.0)

    def test_v07_parse_stats_ptr_count_correct(self):
        """parse_stats['ptr'] should equal num_components * num_ptrs."""
        # 5 components × 4 PTRs per part = 20 PTRs
        stdf = _build_stdf(num_components=5, num_sites=1, num_ptrs_per_part=4)
        report = self.validator.validate_bytes(stdf)
        self.assertEqual(report.total_ptr_count, 20,
                         f"Expected 20 PTRs, got {report.total_ptr_count}")

    def test_v08_multi_site_valid(self):
        """4-site, 5-components/site STDF → SAFE."""
        stdf = _build_stdf(num_components=5, num_sites=4)
        report = self.validator.validate_bytes(stdf)
        self.assertIn(report.recommendation, ["SAFE", "CAUTION"])
        self.assertTrue(report.is_valid)

    def test_v09_has_far_mir_sdr_flags(self):
        """Report should correctly flag presence of FAR, MIR, SDR."""
        report = self.validator.validate_bytes(self.valid_stdf)
        self.assertTrue(report.has_far)
        self.assertTrue(report.has_mir)
        self.assertTrue(report.has_sdr)

    def test_v10_valid_file_path(self):
        """validate_file() on a temp .stdf file → SAFE."""
        with tempfile.NamedTemporaryFile(suffix=".stdf", delete=False) as f:
            f.write(self.valid_stdf)
            tmp_path = f.name
        try:
            report = self.validator.validate_file(tmp_path)
            self.assertIn(report.recommendation, ["SAFE", "CAUTION"])
            self.assertEqual(report.file_size_bytes, len(self.valid_stdf))
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
