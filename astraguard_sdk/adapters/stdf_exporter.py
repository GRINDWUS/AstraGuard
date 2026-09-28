"""
AstraGuard SDK — STDF → CSV Reverse Adapter (Bidirectional Export)
===================================================================
Normalizes parsed STDF data back to CSV format for compatibility with
legacy ATE reporting systems, data lakes, and Excel/JMP analysis tools.

Demonstrates bidirectional thinking:
  ATE STDF → AstraGuard Analysis → Normalized CSV → Legacy Systems

Features:
  - Full canonical column schema matching CSVATEAdapter expectations
  - Per-site rows (site_num preserved as a column)
  - Optional: include AstraGuard quality flags in output
  - Optional: include raw metadata columns
  - Row count returned for audit verification

Output CSV Schema:
  lot_id, component_id, site_num, device_family, test_type,
  primary_parameter, unit, value_0h, value_24h, value_96h_actual,
  value_168h_actual, delta_iddq, spec_max_iddq, is_defective_gt,
  instrument_status, hard_bin, soft_bin, wafer_x, wafer_y,
  pass_fail, operator, temperature_c

Usage:
  from astraguard_sdk.adapters.stdf_exporter import STDFExporter
  exporter = STDFExporter()
  row_count = exporter.export_to_csv("lot.stdf", "lot_normalized.csv")
  print(f"Exported {row_count} rows")
"""

import os
import csv
import logging
from typing import List, Optional
from datetime import datetime

import pandas as pd
from astraguard_sdk.adapters.stdf_adapter import STDFATEAdapter
from astraguard_sdk.schema import SDKMeasurementRecord

logger = logging.getLogger("astraguard.stdf.exporter")

# Canonical output column order (matches CSVATEAdapter to_dataframe schema)
_CSV_COLUMNS = [
    "lot_id",
    "component_id",
    "site_num",
    "device_family",
    "test_type",
    "primary_parameter",
    "unit",
    "value_0h",
    "value_24h",
    "value_96h_actual",
    "value_168h_actual",
    "delta_iddq",
    "spec_max_iddq",
    "is_defective_gt",
    "instrument_status",
    "hard_bin",
    "soft_bin",
    "wafer_x",
    "wafer_y",
    "pass_fail",
    "operator",
    "temperature_c",
    "exported_at",
]


class STDFExporter:
    """
    Bidirectional STDF → CSV reverse adapter.

    Reads a binary STDF file and exports a normalized, analysis-ready
    CSV suitable for legacy ISRO ATE reporting systems or external tools.
    """

    def __init__(self):
        self._adapter = STDFATEAdapter()
        self._exported_at = datetime.utcnow().isoformat()

    def export_to_csv(
        self,
        stdf_path: str,
        output_csv: str,
        include_metadata: bool = True,
        overwrite: bool = True,
    ) -> int:
        """
        Parse an STDF file and write normalized rows to a CSV file.

        Args:
            stdf_path:        Path to input binary STDF file.
            output_csv:       Destination CSV file path.
            include_metadata: If True, include all metadata columns.
            overwrite:        If False, raise if output_csv already exists.

        Returns:
            int: Number of rows written (for audit verification).
        """
        if not overwrite and os.path.exists(output_csv):
            raise FileExistsError(
                f"STDFExporter: '{output_csv}' already exists. Use overwrite=True to replace."
            )

        records = self._adapter.parse(stdf_path)
        df = self._records_to_export_df(records, include_metadata=include_metadata)

        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)

        row_count = len(df)
        logger.info(
            "STDFExporter: exported %d rows from '%s' → '%s'",
            row_count, os.path.basename(stdf_path), output_csv
        )
        return row_count

    def export_records_to_csv(
        self,
        records: List[SDKMeasurementRecord],
        output_csv: str,
        include_metadata: bool = True,
        overwrite: bool = True,
    ) -> int:
        """
        Export already-parsed SDKMeasurementRecords directly to CSV.
        Useful when you've already parsed the STDF and want to avoid re-reading.
        """
        if not overwrite and os.path.exists(output_csv):
            raise FileExistsError(
                f"STDFExporter: '{output_csv}' already exists. Use overwrite=True to replace."
            )
        df = self._records_to_export_df(records, include_metadata=include_metadata)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        return len(df)

    def to_dataframe(
        self,
        stdf_path: str,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Parse an STDF file and return a normalized DataFrame without writing to disk.

        Args:
            stdf_path:        Path to input STDF file.
            include_metadata: If True, include all metadata columns.

        Returns:
            pandas DataFrame in canonical AstraGuard CSV schema.
        """
        records = self._adapter.parse(stdf_path)
        return self._records_to_export_df(records, include_metadata=include_metadata)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _records_to_export_df(
        self,
        records: List[SDKMeasurementRecord],
        include_metadata: bool,
    ) -> pd.DataFrame:
        """Convert SDKMeasurementRecords into the canonical CSV export DataFrame."""
        rows = []
        for r in records:
            meta = r.metadata
            v0   = meta.get("value_0h",  r.value)
            v24  = meta.get("value_24h", r.value)
            v96  = meta.get("value_96h", r.value * 1.02)
            v168 = meta.get("value_168h", r.value * 1.05)

            row = {
                "lot_id":           r.lot_id or "LOT_STDF",
                "component_id":     r.component_id,
                "site_num":         meta.get("site_num", 1),
                "device_family":    r.device_family or "",
                "test_type":        r.test_type or "BURN_IN",
                "primary_parameter": r.parameter_name,
                "unit":             r.unit,
                "value_0h":         round(v0,   6),
                "value_24h":        round(v24,  6),
                "value_96h_actual": round(v96,  6),
                "value_168h_actual": round(v168, 6),
                "delta_iddq":       round(v24 - v0, 6),
                "spec_max_iddq":    meta.get("hi_limit", 50.0),
                "is_defective_gt":  1 if meta.get("pass_fail", "PASS") == "FAIL" else 0,
                "instrument_status": r.measurement_state,
                "hard_bin":         meta.get("hard_bin", 1),
                "soft_bin":         meta.get("soft_bin", 1),
                "wafer_x":          meta.get("wafer_x", 0),
                "wafer_y":          meta.get("wafer_y", 0),
                "pass_fail":        meta.get("pass_fail", "PASS"),
                "operator":         meta.get("operator", "UNKNOWN"),
                "temperature_c":    r.temperature_c or 125.0,
                "exported_at":      self._exported_at,
            }

            if include_metadata:
                # Include any extra metadata that's not already in the row
                for mk, mv in meta.items():
                    col = f"meta_{mk}"
                    if col not in row and mk not in (
                        "value_0h", "value_24h", "value_96h", "value_168h",
                        "hi_limit", "lo_limit", "pass_fail", "hard_bin",
                        "soft_bin", "wafer_x", "wafer_y", "operator",
                        "site_num", "stdf_key"
                    ):
                        row[col] = mv

            rows.append(row)

        df = pd.DataFrame(rows)

        # Enforce canonical column order for non-meta columns
        canonical_cols = [c for c in _CSV_COLUMNS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in canonical_cols]
        df = df[canonical_cols + extra_cols]

        return df
