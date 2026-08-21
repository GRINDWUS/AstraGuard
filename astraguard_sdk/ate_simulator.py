"""
AstraGuard SDK — Non-Invasive ATE Simulator
=============================================
Simulates external ATE measurement feeds across 4 device families.
IMPORTANT: The simulator generates measurement streams only.
It carries NO control-plane capabilities and cannot alter ATE hardware settings.
"""

from typing import List, Dict, Any
import pandas as pd
from dataset_generator.lot_generator import LotSimulator


class NonInvasiveATESimulator:
    """Simulates ATE test data exports for AstraGuard SDK demonstration."""

    def __init__(self):
        self.lot_sim = LotSimulator()

    def generate_ate_stream(
        self,
        device_family: str = "DIGITAL_IC",
        lot_id: int = 1,
        num_components: int = 50,
        corrupt_metadata: bool = False,
        strip_metadata: bool = False
    ) -> pd.DataFrame:
        """Emits an ATE measurement stream dataframe for the specified device family."""
        return self.lot_sim.generate_lot(
            lot_id=lot_id,
            num_components=num_components,
            device_family=device_family,
            corrupt_metadata=corrupt_metadata,
            strip_metadata=strip_metadata
        )
