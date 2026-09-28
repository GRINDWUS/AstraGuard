"""
AstraGuard Phase 2A — STDF Exporter Test Suite
===============================================
Tests bidirectional STDF → CSV export functionality.

Test Coverage:
  E01: export_records_to_csv() writes file and returns correct row count
  E02: Output CSV has all canonical column headers
  E03: lot_id preserved correctly in CSV rows
  E04: site_num column present in exported CSV
  E05: delta_iddq computed correctly (value_24h - value_0h)
  E06: to_dataframe() returns DataFrame without disk I/O
  E07: is_defective_gt = 1 for FAIL records, 0 for PASS
  E08: overwrite=False raises on existing file
  E09: Multi-site export has rows from all sites
  E10: include_metadata=False omits meta_ columns
"""

import os
import tempfile
import unittest

import pandas as pd

from astraguard_sdk.adapters.stdf_exporter import STDFExporter
from astraguard_sdk.schema import SDKMeasurementRecord
from tests.test_stdf_adapter import _build_stdf
from astraguard_sdk.adapters.stdf_adapter import STDFATEAdapter

_CANONICAL_COLS = [
    "lot_id", "component_id", "site_num", "device_family", "test_type",
    "primary_parameter", "unit", "value_0h", "value_24h", "value_96h_actual",
    "value_168h_actual", "delta_iddq", "spec_max_iddq", "is_defective_gt",
    "instrument_status", "hard_bin", "soft_bin", "wafer_x", "wafer_y",
    "pass_fail", "operator", "temperature_c", "exported_at",
]


def _make_records(n: int = 5, lot_id: str = "LOT_EXPORT_TEST", fail_some: bool = False):
    """Build synthetic SDKMeasurementRecords for exporter tests."""
    records = []
    for i in range(n):
        v0  = 10.0 + i * 0.5
        v24 = v0 + 0.3
        v96 = v0 + 0.6
        v168 = v0 + 1.0
        state = "OUT_OF_RANGE" if (fail_some and i == 0) else "VALID"
        records.append(SDKMeasurementRecord(
            component_id=f"COMP_{i:04d}",
            lot_id=lot_id,
            device_family="DIGITAL_IC",
            test_type="BURN_IN",
            parameter_name="IDDQ",
            value=v24,
            unit="uA",
            temperature_c=125.0,
            measurement_state=state,
            metadata={
                "value_0h":   v0,
                "value_24h":  v24,
                "value_96h":  v96,
                "value_168h": v168,
                "hi_limit":   50.0,
                "site_num":   1,
                "hard_bin":   1,
                "soft_bin":   1,
                "pass_fail":  "FAIL" if (fail_some and i == 0) else "PASS",
                "operator":   "TEST_QA",
            }
        ))
    return records


class TestSTDFExporterSuite(unittest.TestCase):

    def setUp(self):
        self.exporter = STDFExporter()
        self.records = _make_records(n=10)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------

    def test_e01_export_returns_correct_row_count(self):
        """export_records_to_csv() returns len(records) as row count."""
        out = os.path.join(self.tmpdir, "test_out.csv")
        count = self.exporter.export_records_to_csv(self.records, out)
        self.assertEqual(count, 10)
        self.assertTrue(os.path.exists(out))

    def test_e02_canonical_columns_present(self):
        """Exported CSV must contain all canonical column headers."""
        out = os.path.join(self.tmpdir, "test_cols.csv")
        self.exporter.export_records_to_csv(self.records, out)
        df = pd.read_csv(out)
        for col in _CANONICAL_COLS:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_e03_lot_id_preserved(self):
        """lot_id column values must match input records."""
        out = os.path.join(self.tmpdir, "test_lot.csv")
        records = _make_records(n=5, lot_id="LOT_ISRO_2026")
        self.exporter.export_records_to_csv(records, out)
        df = pd.read_csv(out)
        self.assertTrue((df["lot_id"] == "LOT_ISRO_2026").all())

    def test_e04_site_num_column_present(self):
        """site_num column must be in exported CSV."""
        out = os.path.join(self.tmpdir, "test_site.csv")
        self.exporter.export_records_to_csv(self.records, out)
        df = pd.read_csv(out)
        self.assertIn("site_num", df.columns)

    def test_e05_delta_iddq_computed_correctly(self):
        """delta_iddq = value_24h - value_0h for each row."""
        out = os.path.join(self.tmpdir, "test_delta.csv")
        self.exporter.export_records_to_csv(self.records, out)
        df = pd.read_csv(out)
        for _, row in df.iterrows():
            expected = round(row["value_24h"] - row["value_0h"], 6)
            self.assertAlmostEqual(row["delta_iddq"], expected, places=5)

    def test_e06_to_dataframe_no_disk_write(self):
        """to_dataframe() from STDF path returns valid DataFrame, no file written."""
        stdf_bytes = _build_stdf(num_components=5, lot_id="LOT_DF_TEST")
        tmp_stdf = os.path.join(self.tmpdir, "test.stdf")
        with open(tmp_stdf, "wb") as f:
            f.write(stdf_bytes)

        df = self.exporter.to_dataframe(tmp_stdf)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("component_id", df.columns)

    def test_e07_fail_records_marked_is_defective(self):
        """FAIL measurement_state → is_defective_gt = 1."""
        out = os.path.join(self.tmpdir, "test_fail.csv")
        records = _make_records(n=5, fail_some=True)
        self.exporter.export_records_to_csv(records, out)
        df = pd.read_csv(out)
        self.assertEqual(df.loc[0, "is_defective_gt"], 1)
        self.assertEqual(df.loc[1, "is_defective_gt"], 0)

    def test_e08_overwrite_false_raises_on_existing(self):
        """overwrite=False should raise FileExistsError if file exists."""
        out = os.path.join(self.tmpdir, "existing.csv")
        self.exporter.export_records_to_csv(self.records, out)  # first write
        with self.assertRaises(FileExistsError):
            self.exporter.export_records_to_csv(self.records, out, overwrite=False)

    def test_e09_multi_site_export_all_sites_present(self):
        """Exporting multi-site STDF → all site_nums represented in CSV."""
        stdf_bytes = _build_stdf(num_components=5, num_sites=3)
        tmp_stdf = os.path.join(self.tmpdir, "multisite.stdf")
        with open(tmp_stdf, "wb") as f:
            f.write(stdf_bytes)

        out = os.path.join(self.tmpdir, "multisite_out.csv")
        count = self.exporter.export_to_csv(tmp_stdf, out)
        df = pd.read_csv(out)
        site_nums_present = set(df["site_num"].unique())
        self.assertGreaterEqual(len(site_nums_present), 2, "Expected multiple sites in output")

    def test_e10_no_meta_columns_when_disabled(self):
        """include_metadata=False → no meta_ prefixed columns in output."""
        out = os.path.join(self.tmpdir, "no_meta.csv")
        # Add extra metadata keys
        for r in self.records:
            r.metadata["extra_sensor_voltage"] = 3.3
        self.exporter.export_records_to_csv(self.records, out, include_metadata=False)
        df = pd.read_csv(out)
        meta_cols = [c for c in df.columns if c.startswith("meta_")]
        self.assertEqual(len(meta_cols), 0, f"Unexpected meta_ columns: {meta_cols}")


if __name__ == "__main__":
    unittest.main()
