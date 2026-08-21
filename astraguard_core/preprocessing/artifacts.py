"""
AstraGuard 2.4 — Preprocessing Artifact Manager
================================================
Manages serialization, deserialization, and cryptographic hashing of learned
preprocessing pipeline artifacts (scaler_v1.json / preprocessor.pkl).

Ensures complete scientific traceability and reproducible ISRO panel audit verification.
"""

import os
import sys
import json
import pickle
import hashlib
from typing import Dict, Any, Tuple
import numpy as np

# NumPy 2.x -> 1.x unpickling backward compatibility shim
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    if hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray


class ArtifactManager:
    """Manages saving, loading, and hash verification of preprocessing artifacts."""

    @staticmethod
    def save_json(artifact_data: Dict[str, Any], filepath: str) -> str:
        """Save artifact dictionary to JSON file and return SHA-256 hash string."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        content_bytes = json.dumps(artifact_data, indent=2).encode("utf-8")
        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        
        artifact_data["artifact_sha256"] = sha256_hash
        with open(filepath, "w") as f:
            json.dump(artifact_data, f, indent=2)

        return sha256_hash

    @staticmethod
    def load_json(filepath: str) -> Tuple[Dict[str, Any], str]:
        """Load artifact dictionary from JSON file and verify SHA-256 integrity."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Artifact file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        stored_hash = data.get("artifact_sha256", "")
        return data, stored_hash

    @staticmethod
    def save_pickle(obj: Any, filepath: str) -> str:
        """Save object instance to pickle file and return SHA-256 hash."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data_bytes = pickle.dumps(obj)
        sha256_hash = hashlib.sha256(data_bytes).hexdigest()

        with open(filepath, "wb") as f:
            f.write(data_bytes)

        return sha256_hash

    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """Load object instance from pickle file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pickle artifact not found: {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)
