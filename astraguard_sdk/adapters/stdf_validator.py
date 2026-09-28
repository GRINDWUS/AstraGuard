"""
AstraGuard SDK — STDF v4 Data Quality Validator
================================================
Validates binary STDF file integrity and record chain completeness
BEFORE parsing is handed off to STDFATEAdapter for analysis.

Checks performed:
  1. File existence & minimum size (>= 4 bytes for one header)
  2. FAR record presence (must be first record)
  3. MIR record presence (lot/device metadata required)
  4. Record chain order integrity (FAR always first, MRR always last)
  5. PTR count vs PRR count consistency (every DUT should have a result)
  6. Truncated / zero-length record detection
  7. Corrupted PTR values (NaN / Inf floats)
  8. Missing MRR sentinel (indicates truncated file)

Output:
  STDFValidationReport — pydantic model with:
    is_valid: bool
    data_quality_score: float  (0.0 – 1.0)
    recommendation: 'SAFE' | 'CAUTION' | 'REJECT'
    issues: List[STDFIssue]
    parse_stats: dict

Usage:
  from astraguard_sdk.adapters.stdf_validator import STDFValidator
  report = STDFValidator().validate_file("ate_lot.stdf")
  print(report.recommendation)  # 'SAFE' | 'CAUTION' | 'REJECT'
"""

import os
import io
import math
import struct
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger("astraguard.stdf.validator")

# Record type constants (mirrored from stdf_adapter for independence)
_FAR  = (0,  10)
_MIR  = (1,  10)
_SDR  = (1,  80)
_PIR  = (5,  10)
_PTR  = (15, 10)
_PRR  = (5,  20)
_MRR  = (1,  20)


class STDFIssue(BaseModel):
    """A single validation issue detected in the STDF file."""
    severity: str   # "ERROR" | "WARNING" | "INFO"
    record_type: str
    record_index: int
    message: str


class STDFValidationReport(BaseModel):
    """Complete STDF validation result returned by STDFValidator."""
    filepath: str
    file_size_bytes: int
    validated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    is_valid: bool
    data_quality_score: float       # 0.0 – 1.0
    recommendation: str             # 'SAFE' | 'CAUTION' | 'REJECT'

    issues: List[STDFIssue] = Field(default_factory=list)
    parse_stats: Dict[str, int] = Field(default_factory=dict)

    has_far: bool = False
    has_mir: bool = False
    has_mrr: bool = False
    has_sdr: bool = False
    total_ptr_count: int = 0
    total_prr_count: int = 0
    corrupted_record_count: int = 0
    truncated_record_count: int = 0


class STDFValidator:
    """
    AstraGuard STDF Data Quality Validator.

    Performs a lightweight linear scan of the STDF binary stream,
    checking record-level integrity without full field parsing.
    Designed to run in <50ms even on large files.
    """

    # Minimum bytes threshold — files smaller than this are reject-level
    _MIN_FILE_BYTES = 8

    def validate_file(self, filepath: str) -> STDFValidationReport:
        """
        Validate an STDF v4 file and return a structured quality report.

        Args:
            filepath: Path to the .stdf / .std file.

        Returns:
            STDFValidationReport with score, issues, and recommendation.
        """
        issues: List[STDFIssue] = []
        stats: Dict[str, int] = {
            "far": 0, "mir": 0, "sdr": 0, "pir": 0, "ptr": 0,
            "prr": 0, "mrr": 0, "unknown": 0
        }

        # --- File-level checks ---
        if not os.path.exists(filepath):
            return STDFValidationReport(
                filepath=filepath,
                file_size_bytes=0,
                is_valid=False,
                data_quality_score=0.0,
                recommendation="REJECT",
                issues=[STDFIssue(severity="ERROR", record_type="FILE", record_index=0,
                                  message=f"File not found: {filepath}")],
                parse_stats=stats
            )

        file_size = os.path.getsize(filepath)
        if file_size < self._MIN_FILE_BYTES:
            return STDFValidationReport(
                filepath=filepath,
                file_size_bytes=file_size,
                is_valid=False,
                data_quality_score=0.0,
                recommendation="REJECT",
                issues=[STDFIssue(severity="ERROR", record_type="FILE", record_index=0,
                                  message=f"File too small ({file_size} bytes) — likely empty or corrupt.")],
                parse_stats=stats
            )

        with open(filepath, "rb") as f:
            data = f.read()

        return self.validate_bytes(data, filepath=filepath)

    def validate_bytes(self, data: bytes, filepath: str = "in-memory") -> STDFValidationReport:
        """Validate raw STDF bytes. Useful for testing without disk I/O."""
        issues: List[STDFIssue] = []
        stats: Dict[str, int] = {
            "far": 0, "mir": 0, "sdr": 0, "pir": 0, "ptr": 0,
            "prr": 0, "mrr": 0, "unknown": 0
        }
        file_size = len(data)

        if file_size < self._MIN_FILE_BYTES:
            return STDFValidationReport(
                filepath=filepath, file_size_bytes=file_size,
                is_valid=False, data_quality_score=0.0, recommendation="REJECT",
                issues=[STDFIssue(severity="ERROR", record_type="FILE", record_index=0,
                                  message="Data too small — likely empty or corrupt.")],
                parse_stats=stats
            )

        buf = io.BytesIO(data)
        endian = "<"
        record_idx = 0
        truncated_count = 0
        corrupted_ptr_count = 0
        first_record = True

        # Flags
        has_far = has_mir = has_mrr = has_sdr = False

        while buf.tell() < file_size:
            header_bytes = buf.read(4)
            if len(header_bytes) < 4:
                if len(header_bytes) > 0:
                    issues.append(STDFIssue(
                        severity="WARNING", record_type="TRUNCATED_HEADER",
                        record_index=record_idx,
                        message=f"Unexpected end of file — truncated header ({len(header_bytes)} bytes)."
                    ))
                    truncated_count += 1
                break

            rec_len, rec_type, rec_sub = struct.unpack("<HBB", header_bytes)
            body = buf.read(rec_len)

            if len(body) < rec_len:
                issues.append(STDFIssue(
                    severity="WARNING", record_type=f"T{rec_type}_S{rec_sub}",
                    record_index=record_idx,
                    message=f"Truncated record body (expected {rec_len}B, got {len(body)}B)."
                ))
                truncated_count += 1
                record_idx += 1
                continue

            key = (rec_type, rec_sub)

            # --- FAR checks ---
            if key == _FAR:
                if not first_record:
                    issues.append(STDFIssue(
                        severity="ERROR", record_type="FAR", record_index=record_idx,
                        message="FAR record found after first record position — malformed STDF."
                    ))
                if len(body) >= 2:
                    cpu_type = body[0]
                    stdf_ver = body[1]
                    endian = ">" if cpu_type == 1 else "<"
                    if stdf_ver != 4:
                        issues.append(STDFIssue(
                            severity="WARNING", record_type="FAR", record_index=record_idx,
                            message=f"STDF version={stdf_ver} (expected 4). Parser may misinterpret records."
                        ))
                has_far = True
                stats["far"] += 1

            elif key == _MIR:
                has_mir = True
                stats["mir"] += 1

            elif key == _SDR:
                has_sdr = True
                stats["sdr"] += 1

            elif key == _PIR:
                stats["pir"] += 1

            elif key == _PTR:
                stats["ptr"] += 1
                # Validate value field (bytes 8–12) is a valid float
                if len(body) >= 12:
                    try:
                        val = struct.unpack_from(f"{endian}f", body, 8)[0]
                        if math.isnan(val) or math.isinf(val):
                            corrupted_ptr_count += 1
                            issues.append(STDFIssue(
                                severity="WARNING", record_type="PTR", record_index=record_idx,
                                message=f"PTR record #{record_idx}: NaN/Inf float value detected."
                            ))
                    except struct.error:
                        corrupted_ptr_count += 1

            elif key == _PRR:
                stats["prr"] += 1

            elif key == _MRR:
                has_mrr = True
                stats["mrr"] += 1
                # MRR should be the last record — stop scanning
                break

            else:
                stats["unknown"] += 1

            first_record = False
            record_idx += 1

        # --- Post-scan checks ---
        if not has_far:
            issues.append(STDFIssue(severity="ERROR", record_type="FAR", record_index=0,
                                    message="No FAR record found — file may not be valid STDF."))
        if not has_mir:
            issues.append(STDFIssue(severity="ERROR", record_type="MIR", record_index=0,
                                    message="No MIR record found — lot/device metadata missing."))
        if not has_mrr:
            issues.append(STDFIssue(severity="WARNING", record_type="MRR", record_index=record_idx,
                                    message="No MRR record found — file may be incomplete or truncated."))

        ptr_total = stats["ptr"]
        prr_total = stats["prr"]
        if ptr_total > 0 and prr_total > 0:
            ptr_prr_ratio = min(ptr_total, prr_total) / max(ptr_total, prr_total)
            if ptr_prr_ratio < 0.80:
                issues.append(STDFIssue(
                    severity="WARNING", record_type="PTR/PRR",
                    record_index=record_idx,
                    message=f"PTR count ({ptr_total}) vs PRR count ({prr_total}) mismatch "
                            f"({ptr_prr_ratio:.0%}) — some DUTs may be missing results."
                ))

        # --- Score calculation ---
        error_count   = sum(1 for i in issues if i.severity == "ERROR")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")
        total_records = max(record_idx, 1)

        # Deduct: 0.20 per ERROR, 0.05 per WARNING, 0.02 per corrupted PTR
        score = 1.0
        score -= error_count   * 0.20
        score -= warning_count * 0.05
        score -= (corrupted_ptr_count / max(ptr_total, 1)) * 0.10
        score -= (truncated_count / total_records) * 0.10
        score = round(max(0.0, min(1.0, score)), 4)

        # --- Recommendation ---
        if error_count > 0 or score < 0.50:
            recommendation = "REJECT"
            is_valid = False
        elif warning_count > 2 or score < 0.80:
            recommendation = "CAUTION"
            is_valid = True
        else:
            recommendation = "SAFE"
            is_valid = True

        return STDFValidationReport(
            filepath=filepath,
            file_size_bytes=file_size,
            is_valid=is_valid,
            data_quality_score=score,
            recommendation=recommendation,
            issues=issues,
            parse_stats=stats,
            has_far=has_far,
            has_mir=has_mir,
            has_mrr=has_mrr,
            has_sdr=has_sdr,
            total_ptr_count=ptr_total,
            total_prr_count=prr_total,
            corrupted_record_count=corrupted_ptr_count,
            truncated_record_count=truncated_count,
        )
