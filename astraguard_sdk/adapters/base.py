"""
AstraGuard SDK — Base Adapter Interface
========================================
All data adapters convert external ATE formats into canonical SDKMeasurementRecords.
"""

from abc import ABC, abstractmethod
from typing import List, Any
import pandas as pd
from astraguard_sdk.schema import SDKMeasurementRecord


class BaseATEAdapter(ABC):
    """Abstract base class for all ATE data format adapters."""

    @abstractmethod
    def parse(self, raw_data: Any) -> List[SDKMeasurementRecord]:
        """Convert raw external payload into canonical SDKMeasurementRecords."""
        pass

    @abstractmethod
    def to_dataframe(self, records: List[SDKMeasurementRecord]) -> pd.DataFrame:
        """Convert canonical records into pandas DataFrame for ML models."""
        pass
