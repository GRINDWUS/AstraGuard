"""
AstraGuard 2.4 — Formal Model Registry Builder
==============================================
Establishes model lineage, metadata versioning, lot assignments, and SHA-256 hashes
for all locked Module B regressor models.

Directory structure:
  models/registry/{family_dir}/
    ├── model_v1.pkl
    └── metadata.json
"""

import os
import json
import shutil
import hashlib
from datetime import datetime, timezone
import pandas as pd

REGISTRY_BASE = "models/registry"
MANIFEST_PATH = "ASQD_2.4/manifest.json"
TOURNAMENT_SUMMARY_PATH = "models/module_b/tournament_summary.json"

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

with open(TOURNAMENT_SUMMARY_PATH) as f:
    tournament_summary = json.load(f)

print("=" * 80)
print("ASTRAGUARD 2.4 — FORMAL MODEL REGISTRY BUILDER")
print("=" * 80)

for dev_family, info in tournament_summary.items():
    fam_dir = dev_family.lower()
    fam_registry_path = os.path.join(REGISTRY_BASE, fam_dir)
    os.makedirs(fam_registry_path, exist_ok=True)

    src_model_path = os.path.join("models/module_b", f"{fam_dir}_module_b.pkl")
    winning_alg = info["winning_model"]
    model_filename = f"module_b_{winning_alg.lower()}_v1.pkl"
    dst_model_path = os.path.join(fam_registry_path, model_filename)

    shutil.copy2(src_model_path, dst_model_path)

    # Compute SHA-256 hash of registered model artifact
    with open(dst_model_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    # Retrieve lot assignments from ASQD manifest
    lot_assignments = manifest.get("lot_split_assignments", {}).get(dev_family, {})

    metadata = {
        "model_version": "1.0.0",
        "device_family": dev_family,
        "module": "MODULE_B",
        "algorithm": winning_alg,
        "preprocessor_version": "2.4.0",
        "preprocessor_hash": info.get("preprocessor_sha256", ""),
        "artifact_file": model_filename,
        "artifact_sha256": sha256_hash,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_lots": lot_assignments.get("train", []),
        "validation_lots": lot_assignments.get("validation", []),
        "blind_test_lots": lot_assignments.get("blind_test", []),
        "validation_metrics": {
            "val_mae": info["val_mae"],
            "val_r2": info["val_r2"],
            "ood_generalization_gap": info["ood_generalization_gap"]
        },
        "blind_test_metrics": {
            "blind_test_mae": info["blind_test_mae"],
            "blind_test_rmse": info["blind_test_rmse"],
            "blind_test_mape_pct": info["blind_test_mape_pct"],
            "blind_test_r2": info["blind_test_r2"],
            "blind_test_escaped_defect_rate_pct": info["blind_test_escaped_defect_rate_pct"]
        }
    }

    metadata_path = os.path.join(fam_registry_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Registered {dev_family:25s} -> {dst_model_path} (SHA-256: {sha256_hash[:12]}...)")

print("\n" + "=" * 80)
print("FORMAL MODEL REGISTRY PERSISTED AT models/registry/")
print("=" * 80)
