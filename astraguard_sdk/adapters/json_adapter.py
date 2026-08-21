"""
AstraGuard SDK — JSON & Streaming API Adapter
==============================================
Converts JSON payloads and REST/WebSocket streams into canonical SDKMeasurementRecords.
"""

from typing import List, Dict, Any, Union
import json
import pandas as pd
from astraguard_sdk.adapters.base import BaseATEAdapter
from astraguard_sdk.schema import SDKMeasurementRecord


class JSONATEAdapter(BaseATEAdapter):
    """Adapter for JSON payloads or lists of dictionaries."""

    def parse(self, raw_data: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> List[SDKMeasurementRecord]:
        if isinstance(raw_data, str):
            parsed = json.loads(raw_data)
        else:
            parsed = raw_data

        if isinstance(parsed, dict):
            items = parsed.get("records", parsed.get("components", [parsed]))
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError("JSONATEAdapter requires a valid JSON string, dict, or list.")

        records = []
        for idx, item in enumerate(items):
            comp_id = str(item.get("component_id", f"COMP_{idx:04d}"))
            param_name = str(item.get("primary_parameter", item.get("parameter", "IDDQ")))
            val = float(item.get("value_24h", item.get("value", item.get("iddq_24h", 0.0))))
            unit = str(item.get("unit", "uA"))

            rec = SDKMeasurementRecord(
                component_id=comp_id,
                lot_id=str(item.get("lot_id", "LOT_UNKNOWN")),
                device_family=item.get("device_family"),
                test_type=str(item.get("test_type", "BURN_IN")),
                parameter_name=param_name,
                value=val,
                unit=unit,
                temperature_c=float(item.get("temperature_c", item.get("stress_temperature_c", 25.0))),
                operating_voltage_v=float(item.get("operating_voltage_v", item.get("stress_voltage_v", 5.0))),
                metadata=item
            )
            records.append(rec)
        return records

    def to_dataframe(self, records: List[SDKMeasurementRecord]) -> pd.DataFrame:
        from astraguard_sdk.adapters.csv_adapter import CSVATEAdapter
        return CSVATEAdapter().to_dataframe(records)
