"""
AstraGuard SDK — CSV & Dataframe Adapter
=========================================
Converts CSV files, DataFrames, and STDF exports into canonical SDKMeasurementRecords.
"""

from typing import List, Union
import pandas as pd
import numpy as np
from astraguard_sdk.adapters.base import BaseATEAdapter
from astraguard_sdk.schema import SDKMeasurementRecord


class CSVATEAdapter(BaseATEAdapter):
    """Adapter for CSV files and pandas DataFrames."""

    def parse(self, raw_data: Union[str, pd.DataFrame]) -> List[SDKMeasurementRecord]:
        if isinstance(raw_data, str):
            df = pd.read_csv(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        else:
            raise ValueError("CSVATEAdapter accepts a CSV file path or pandas DataFrame.")

        records = []
        for idx, row in df.iterrows():
            comp_id = str(row.get("component_id", f"COMP_{idx:04d}"))
            param_name = str(row.get("primary_parameter", row.get("parameter", "IDDQ")))
            val = float(row.get("value_24h", row.get("value", row.get("iddq_24h", 0.0))))
            unit = str(row.get("unit", "uA"))
            dev_fam = row.get("device_family")
            if pd.isna(dev_fam):
                dev_fam = None

            rec = SDKMeasurementRecord(
                component_id=comp_id,
                lot_id=str(row.get("lot_id", "LOT_UNKNOWN")),
                device_family=dev_fam,
                test_type=str(row.get("test_type", "BURN_IN")),
                parameter_name=param_name,
                value=val,
                unit=unit,
                temperature_c=float(row.get("stress_temperature_c", row.get("temperature_C", row.get("temperature", 25.0)))),
                operating_voltage_v=float(row.get("stress_voltage_v", row.get("voltage_V", row.get("voltage", 5.0)))),
                metadata={
                    "value_0h": float(row.get("iddq_0h", row.get("value_0h", val))),
                    "value_24h": val,
                    "value_96h": float(row.get("iddq_96h_actual", row.get("value_96h", val * 1.02))),
                    "value_168h": float(row.get("iddq_168h_actual", row.get("value_168h", val * 1.05))),
                    "wafer_x": float(row.get("wafer_x", 0.0)),
                    "wafer_y": float(row.get("wafer_y", 0.0)),
                    **{k: v for k, v in row.items() if str(v) != "nan"}, # Add all extra columns to metadata
                }
            )
            records.append(rec)
        return records

    def to_dataframe(self, records: List[SDKMeasurementRecord]) -> pd.DataFrame:
        rows = []
        for r in records:
            meta = r.metadata
            rows.append({
                "component_id": r.component_id,
                "lot_id": r.lot_id or "LOT_UNKNOWN",
                "device_family": r.device_family,
                "test_type": r.test_type,
                "primary_parameter": r.parameter_name,
                "unit": r.unit,
                "iddq_0h": meta.get("value_0h", r.value),
                "iddq_24h": meta.get("value_24h", r.value),
                "iddq_96h_actual": meta.get("value_96h", r.value * 1.02),
                "iddq_168h_actual": meta.get("value_168h", r.value * 1.05),
                "value_0h": meta.get("value_0h", r.value),
                "value_24h": meta.get("value_24h", r.value),
                "value_96h_actual": meta.get("value_96h", r.value * 1.02),
                "value_168h_actual": meta.get("value_168h", r.value * 1.05),
                "delta_iddq": round(meta.get("value_24h", r.value) - meta.get("value_0h", r.value), 6),
                "delta_24h_ua": round(meta.get("value_24h", r.value) - meta.get("value_0h", r.value), 6),
                "spec_max_iddq": 50.0,
                "is_defective_gt": meta.get("is_defective_gt", False),
                "instrument_status": r.measurement_state,
                "wafer_x": meta.get("wafer_x", 0.0),
                "wafer_y": meta.get("wafer_y", 0.0),
            })
        return pd.DataFrame(rows)
