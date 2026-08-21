"""
AstraGuard 2.4 — Safe ATE Integration SDK
==========================================
Non-invasive, read-only analytics SDK for semiconductor and aerospace test environments.
"""

from astraguard_sdk.client import AstraGuardClient
from astraguard_sdk.schema import SDKMeasurementRecord, SDKAnalysisResult

# Backward compatibility alias
AstraGuardATESDK = AstraGuardClient

__version__ = "2.4.0-safe-ate"
__all__ = ["AstraGuardClient", "AstraGuardATESDK", "SDKMeasurementRecord", "SDKAnalysisResult"]
