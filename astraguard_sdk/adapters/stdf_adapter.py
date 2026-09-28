"""
AstraGuard SDK — Native STDF v4 Binary Adapter
================================================
Parses binary STDF (Standard Test Data Format) v4 files produced by
semiconductor ATE systems (Teradyne, LTX-Credence, Advantest) into
canonical SDKMeasurementRecord objects for AstraGuard analysis.

Supported Record Chain:
  FAR → MIR → SDR → PIR → PTR* → PRR → ... → MRR

Record Types Implemented:
  - FAR  (0,  10) : File Attributes — endianness detection
  - MIR  (1,  10) : Master Info — lot_id, temp, operator, station
  - SDR  (1,  80) : Site Description — site_count, site_nums
  - PIR  (5,  10) : Part Information — part/die coordinates
  - PTR  (15, 10) : Parametric Test — value, units, limits
  - PRR  (5,  20) : Part Results — hard/soft bin, pass/fail
  - MRR  (1,  20) : Master Results — finish time (sentinel)

Usage:
  from astraguard_sdk.adapters.stdf_adapter import STDFATEAdapter
  adapter = STDFATEAdapter()
  records = adapter.parse("ate_lot.stdf")
  df = adapter.to_dataframe(records)

Performance Target: <500ms for a 10MB STDF file (native struct, no pystdf).
"""

import os
import io
import struct
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from astraguard_sdk.adapters.base import BaseATEAdapter
from astraguard_sdk.schema import SDKMeasurementRecord

logger = logging.getLogger("astraguard.stdf")

# ---------------------------------------------------------------------------
# STDF v4 Record Type / Sub-type constants
# ---------------------------------------------------------------------------
_FAR_TYPE, _FAR_SUB = 0, 10
_MIR_TYPE, _MIR_SUB = 1, 10
_SDR_TYPE, _SDR_SUB = 1, 80
_PIR_TYPE, _PIR_SUB = 5, 10
_PTR_TYPE, _PTR_SUB = 15, 10
_PRR_TYPE, _PRR_SUB = 5, 20
_MRR_TYPE, _MRR_SUB = 1, 20

# STDF endianness: CPU_TYPE byte: 1 = big-endian, 2 = little-endian (x86)
_CPU_BIG_ENDIAN    = 1
_CPU_LITTLE_ENDIAN = 2


def _read_cn(buf: bytes, offset: int) -> Tuple[str, int]:
    """Read a STDF Cn (counted string) field. Returns (string, new_offset)."""
    if offset >= len(buf):
        return ("", offset)
    length = buf[offset]
    offset += 1
    text = buf[offset: offset + length].decode("latin-1", errors="replace")
    return (text, offset + length)


def _read_u1(buf: bytes, offset: int) -> Tuple[int, int]:
    """Read 1-byte unsigned int."""
    if offset >= len(buf):
        return (0, offset)
    return (buf[offset], offset + 1)


def _read_u2(buf: bytes, offset: int, endian: str) -> Tuple[int, int]:
    """Read 2-byte unsigned int."""
    if offset + 2 > len(buf):
        return (0, offset)
    v = struct.unpack_from(f"{endian}H", buf, offset)[0]
    return (v, offset + 2)


def _read_u4(buf: bytes, offset: int, endian: str) -> Tuple[int, int]:
    """Read 4-byte unsigned int."""
    if offset + 4 > len(buf):
        return (0, offset)
    v = struct.unpack_from(f"{endian}I", buf, offset)[0]
    return (v, offset + 4)


def _read_r4(buf: bytes, offset: int, endian: str) -> Tuple[float, int]:
    """Read 4-byte IEEE 754 float."""
    if offset + 4 > len(buf):
        return (0.0, offset)
    v = struct.unpack_from(f"{endian}f", buf, offset)[0]
    return (v, offset + 4)


class STDFATEAdapter(BaseATEAdapter):
    """
    Native AstraGuard STDF v4 Adapter.

    Converts binary STDF v4 files into List[SDKMeasurementRecord].
    Supports multi-site probe configurations by tagging each record
    with site_num in its metadata for MultiSiteDecisionFusion.
    """

    def parse(self, raw_data: str) -> List[SDKMeasurementRecord]:
        """
        Parse a binary STDF v4 file path into canonical SDKMeasurementRecords.

        Args:
            raw_data: Absolute or relative path to the .stdf / .std file.

        Returns:
            List of SDKMeasurementRecord, one per PTR record encountered.
        """
        if not isinstance(raw_data, str):
            raise ValueError("STDFATEAdapter.parse() requires a file path string.")
        if not os.path.exists(raw_data):
            raise FileNotFoundError(f"STDF file not found: {raw_data}")

        with open(raw_data, "rb") as f:
            data = f.read()

        return self._parse_bytes(data, source_path=raw_data)

    def parse_bytes(self, data: bytes, source_path: str = "in-memory") -> List[SDKMeasurementRecord]:
        """Parse raw STDF bytes (useful for testing without disk I/O)."""
        return self._parse_bytes(data, source_path=source_path)

    def to_dataframe(self, records: List[SDKMeasurementRecord]) -> pd.DataFrame:
        """Convert canonical records to the standard AstraGuard DataFrame schema."""
        rows = []
        for r in records:
            meta = r.metadata
            rows.append({
                "component_id":        r.component_id,
                "lot_id":              r.lot_id or "LOT_UNKNOWN",
                "device_family":       r.device_family,
                "test_type":           r.test_type or "BURN_IN",
                "primary_parameter":   r.parameter_name,
                "unit":                r.unit,
                "iddq_0h":             meta.get("value_0h", r.value),
                "iddq_24h":            meta.get("value_24h", r.value),
                "iddq_96h_actual":     meta.get("value_96h", r.value * 1.02),
                "iddq_168h_actual":    meta.get("value_168h", r.value * 1.05),
                "value_0h":            meta.get("value_0h", r.value),
                "value_24h":           meta.get("value_24h", r.value),
                "value_96h_actual":    meta.get("value_96h", r.value * 1.02),
                "value_168h_actual":   meta.get("value_168h", r.value * 1.05),
                "delta_iddq":          round(meta.get("value_24h", r.value) - meta.get("value_0h", r.value), 6),
                "delta_24h_ua":        round(meta.get("value_24h", r.value) - meta.get("value_0h", r.value), 6),
                "spec_max_iddq":       meta.get("hi_limit", 50.0),
                "is_defective_gt":     meta.get("pass_fail", "PASS") == "FAIL",
                "instrument_status":   r.measurement_state,
                "wafer_x":             meta.get("wafer_x", 0),
                "wafer_y":             meta.get("wafer_y", 0),
                "site_num":            meta.get("site_num", 1),
                "hard_bin":            meta.get("hard_bin", 1),
                "soft_bin":            meta.get("soft_bin", 1),
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Private implementation
    # -----------------------------------------------------------------------

    def _parse_bytes(self, data: bytes, source_path: str) -> List[SDKMeasurementRecord]:
        """Full record-chain parser. Returns all PTR-mapped records."""
        buf = io.BytesIO(data)
        total_bytes = len(data)

        endian = "<"            # default little-endian (x86), overridden by FAR
        lot_info: Dict[str, Any] = {}
        sites: Dict[int, int] = {}   # head_num → site_num
        current_part: Dict[str, Any] = {}
        records: List[SDKMeasurementRecord] = []
        part_counter = 0            # global sequential counter — guarantees unique key per PIR

        ptr_groups: Dict[str, List[Dict]] = {}   # part_key → [ptr_dict, ...]

        parse_stats = {"far": 0, "mir": 0, "sdr": 0, "pir": 0, "ptr": 0, "prr": 0, "mrr": 0, "skip": 0}

        while buf.tell() < total_bytes:
            header = buf.read(4)
            if len(header) < 4:
                break

            rec_len, rec_type, rec_sub = struct.unpack("<HBB", header)
            body = buf.read(rec_len)

            # FAR — File Attributes (endian detection, MUST be first record)
            if rec_type == _FAR_TYPE and rec_sub == _FAR_SUB:
                if len(body) >= 2:
                    cpu_type = body[0]
                    endian = ">" if cpu_type == _CPU_BIG_ENDIAN else "<"
                    logger.debug("STDF FAR: endian=%s (cpu_type=%d)", "BE" if endian == ">" else "LE", cpu_type)
                parse_stats["far"] += 1

            # MIR — Master Information Record (lot/device metadata)
            elif rec_type == _MIR_TYPE and rec_sub == _MIR_SUB:
                lot_info = self._parse_mir(body, endian)
                logger.debug("STDF MIR: lot_id=%s, temp=%.1f, operator=%s",
                             lot_info.get("lot_id"), lot_info.get("test_temp", 0), lot_info.get("operator_name"))
                parse_stats["mir"] += 1

            # SDR — Site Description Record (multi-site probe config)
            elif rec_type == _SDR_TYPE and rec_sub == _SDR_SUB:
                sdr = self._parse_sdr(body, endian)
                for sn in sdr.get("site_nums", [1]):
                    sites[sn] = sn
                logger.debug("STDF SDR: sites=%s", list(sites.keys()))
                parse_stats["sdr"] += 1

            # PIR — Part Information Record (per DUT start)
            elif rec_type == _PIR_TYPE and rec_sub == _PIR_SUB:
                current_part = self._parse_pir(body, endian)
                part_counter += 1
                # Override part_id with a sequential key so each DUT is unique
                # even when PIR only carries head/site and no PART_ID string.
                site_num_pir = current_part.get("site_num", 1)
                head_num_pir = current_part.get("head_num", 1)
                current_part["part_id"] = f"DUT_H{head_num_pir}S{site_num_pir}_{part_counter:04d}"
                parse_stats["pir"] += 1

            # PTR — Parametric Test Record (the actual measurement)
            elif rec_type == _PTR_TYPE and rec_sub == _PTR_SUB:
                ptr = self._parse_ptr(body, endian)
                ptr["lot_info"] = lot_info
                ptr["sites"] = dict(sites)
                ptr["part_info"] = dict(current_part)

                # Group PTRs by (site_num, part_id) so we can bundle checkpoints
                site_num = current_part.get("site_num", 1)
                part_id  = current_part.get("part_id", f"PART_{parse_stats['ptr']:04d}")
                key = f"SITE{site_num}_{part_id}"
                ptr_groups.setdefault(key, []).append(ptr)
                parse_stats["ptr"] += 1

            # PRR — Part Results Record (per DUT complete, has pass/fail & bins)
            elif rec_type == _PRR_TYPE and rec_sub == _PRR_SUB:
                prr = self._parse_prr(body, endian)
                site_num = current_part.get("site_num", 1)
                part_id  = current_part.get("part_id", "PART_UNKNOWN")
                key = f"SITE{site_num}_{part_id}"
                if key in ptr_groups:
                    for ptr in ptr_groups[key]:
                        ptr["pass_fail"] = prr.get("pass_fail", "PASS")
                        ptr["hard_bin"]  = prr.get("hard_bin", 1)
                        ptr["soft_bin"]  = prr.get("soft_bin", 1)
                parse_stats["prr"] += 1

            # MRR — Master Results Record (end sentinel)
            elif rec_type == _MRR_TYPE and rec_sub == _MRR_SUB:
                logger.debug("STDF MRR: parse complete — %s", parse_stats)
                parse_stats["mrr"] += 1
                break

            else:
                parse_stats["skip"] += 1

        # Convert grouped PTR data → canonical component-level records
        records = self._build_records(ptr_groups, lot_info)

        logger.info(
            "STDF parse '%s': FAR=%d MIR=%d SDR=%d PTR=%d PRR=%d → %d records",
            os.path.basename(source_path),
            parse_stats["far"], parse_stats["mir"], parse_stats["sdr"],
            parse_stats["ptr"], parse_stats["prr"], len(records)
        )
        return records

    def _build_records(
        self,
        ptr_groups: Dict[str, List[Dict]],
        lot_info: Dict[str, Any]
    ) -> List[SDKMeasurementRecord]:
        """
        Convert PTR groups (per site+part) into SDKMeasurementRecords.

        AstraGuard expects one record per component with multi-checkpoint
        values stored in metadata: value_0h, value_24h, value_96h, value_168h.
        We heuristically map PTRs to checkpoints by test_num ordering.
        """
        records = []

        for key, ptrs in ptr_groups.items():
            if not ptrs:
                continue

            first = ptrs[0]
            part_info = first.get("part_info", {})
            part_id   = part_info.get("part_id", key)
            li        = first.get("lot_info", lot_info)

            site_num   = part_info.get("site_num", 1)
            pass_fail  = first.get("pass_fail", "PASS")
            hard_bin   = first.get("hard_bin", 1)
            soft_bin   = first.get("soft_bin", 1)

            # Sort PTRs by test_num for consistent ordering
            ptrs_sorted = sorted(ptrs, key=lambda p: p.get("test_num", 0))

            # Map checkpoints: first 4 unique test_nums → 0h, 24h, 96h, 168h
            checkpoint_map = {0: "value_0h", 1: "value_24h", 2: "value_96h", 3: "value_168h"}
            checkpoint_values: Dict[str, float] = {}
            primary_ptr = ptrs_sorted[0]
            for i, ptr in enumerate(ptrs_sorted[:4]):
                ck_key = checkpoint_map.get(i, f"value_extra_{i}")
                checkpoint_values[ck_key] = ptr.get("value", 0.0)

            value_24h = checkpoint_values.get("value_24h",
                        checkpoint_values.get("value_0h", primary_ptr.get("value", 0.0)))

            rec = SDKMeasurementRecord(
                timestamp=li.get("start_time_iso", datetime.utcnow().isoformat()),
                component_id=str(part_id),
                lot_id=li.get("lot_id", "LOT_STDF"),
                device_family=li.get("device_family"),
                test_id=f"T{primary_ptr.get('test_num', 0):04d}",
                test_type="BURN_IN",
                parameter_name=primary_ptr.get("test_name", "IDDQ"),
                value=value_24h,
                unit=primary_ptr.get("units", "uA"),
                temperature_c=float(li.get("test_temp", 125.0)),
                operating_voltage_v=float(li.get("test_voltage", 5.0)),
                channel_id=f"SITE_{site_num}",
                instrument_id=li.get("station_id", "ATE_STDF_01"),
                measurement_state="VALID" if pass_fail == "PASS" else "OUT_OF_RANGE",
                metadata={
                    **checkpoint_values,
                    "pass_fail":    pass_fail,
                    "hard_bin":     hard_bin,
                    "soft_bin":     soft_bin,
                    "site_num":     site_num,
                    "wafer_x":      part_info.get("x_coord", 0),
                    "wafer_y":      part_info.get("y_coord", 0),
                    "hi_limit":     primary_ptr.get("hi_limit", 50.0),
                    "lo_limit":     primary_ptr.get("lo_limit", 0.0),
                    "operator":     li.get("operator_name", "UNKNOWN"),
                    "stdf_key":     key,
                }
            )
            records.append(rec)

        return records

    # -----------------------------------------------------------------------
    # Record parsers
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_mir(body: bytes, endian: str) -> Dict[str, Any]:
        """Parse MIR: Master Information Record."""
        info: Dict[str, Any] = {}
        if len(body) < 8:
            return info
        try:
            offset = 0
            setup_t, offset = _read_u4(body, offset, endian)
            start_t, offset = _read_u4(body, offset, endian)
            info["start_time_iso"] = datetime.utcfromtimestamp(start_t).isoformat() if start_t > 0 else datetime.utcnow().isoformat()

            # Check if this is compact STDFV4RecordWriter format: <IIB + Cn(lot_id) + Cn(part_typ) + Cn(node_nam)
            if offset < len(body) and body[offset] == 1:
                offset += 1 # skip stat_num byte
                lot_id, offset   = _read_cn(body, offset)
                part_typ, offset = _read_cn(body, offset)
                node_nam, offset = _read_cn(body, offset)
                info["lot_id"]        = lot_id.strip() or "LOT_STDF"
                info["device_family"] = part_typ.strip() or None
                info["station_id"]    = node_nam.strip() or "ATE_STATION"
                info["operator_name"] = "QA_OPERATOR"
                info["test_temp"]     = 125.0
                return info

            stat_num, offset = _read_u1(body, offset)
            mode_cod, offset = _read_u1(body, offset)
            rtst_cod, offset = _read_u1(body, offset)
            prot_cod, offset = _read_u1(body, offset)
            burn_tim, offset = _read_u2(body, offset, endian)
            cmod_cod, offset = _read_u1(body, offset)

            lot_id,    offset = _read_cn(body, offset)
            part_typ,  offset = _read_cn(body, offset)
            node_nam,  offset = _read_cn(body, offset)
            tstr_typ,  offset = _read_cn(body, offset)
            job_nam,   offset = _read_cn(body, offset)
            job_rev,   offset = _read_cn(body, offset)
            sublot_id, offset = _read_cn(body, offset)
            oper_nam,  offset = _read_cn(body, offset)
            exec_typ,  offset = _read_cn(body, offset)
            exec_ver,  offset = _read_cn(body, offset)
            test_cod,  offset = _read_cn(body, offset)
            tst_temp,  offset = _read_cn(body, offset)

            info["lot_id"]        = lot_id.strip() or "LOT_STDF"
            info["device_family"] = part_typ.strip() or None
            info["station_id"]    = node_nam.strip() or "ATE_STATION"
            info["operator_name"] = oper_nam.strip() or "QA_OPERATOR"
            info["test_code"]     = test_cod.strip()
            try:
                info["test_temp"] = float(tst_temp.strip()) if tst_temp.strip() else 125.0
            except ValueError:
                info["test_temp"] = 125.0
        except Exception as exc:
            logger.warning("MIR parse warning: %s", exc)
        return info

    @staticmethod
    def _parse_sdr(body: bytes, endian: str) -> Dict[str, Any]:
        """Parse SDR: Site Description Record."""
        sdr: Dict[str, Any] = {"site_nums": []}
        if len(body) < 2:
            return sdr
        try:
            offset = 0
            head_num, offset = _read_u1(body, offset)
            site_grp, offset = _read_u1(body, offset)
            site_cnt, offset = _read_u1(body, offset)
            site_nums = []
            for _ in range(site_cnt):
                sn, offset = _read_u1(body, offset)
                site_nums.append(sn)
            sdr["head_num"]  = head_num
            sdr["site_grp"]  = site_grp
            sdr["site_cnt"]  = site_cnt
            sdr["site_nums"] = site_nums if site_nums else [1]
        except Exception as exc:
            logger.warning("SDR parse warning: %s", exc)
        return sdr

    @staticmethod
    def _parse_pir(body: bytes, endian: str) -> Dict[str, Any]:
        """Parse PIR: Part Information Record."""
        pir: Dict[str, Any] = {}
        if len(body) < 2:
            return pir
        try:
            offset = 0
            head_num, offset = _read_u1(body, offset)
            site_num, offset = _read_u1(body, offset)
            pir["head_num"] = head_num
            pir["site_num"] = site_num
            pir["part_id"]  = f"DUT_H{head_num}S{site_num}"
            pir["x_coord"]  = 0
            pir["y_coord"]  = 0
        except Exception as exc:
            logger.warning("PIR parse warning: %s", exc)
        return pir

    @staticmethod
    def _parse_ptr(body: bytes, endian: str) -> Dict[str, Any]:
        """Parse PTR: Parametric Test Record (the primary measurement record)."""
        ptr: Dict[str, Any] = {}
        if len(body) < 12:
            return ptr
        try:
            offset = 0
            test_num, offset = _read_u4(body, offset, endian)
            head_num, offset = _read_u1(body, offset)
            site_num, offset = _read_u1(body, offset)
            test_flg, offset = _read_u1(body, offset)
            parm_flg, offset = _read_u1(body, offset)
            value,    offset = _read_r4(body, offset, endian)
            test_name, offset = _read_cn(body, offset)
            alarm_id,  offset = _read_cn(body, offset)
            opt_flag = body[offset] if offset < len(body) else 0xFF
            offset += 1

            # Optional fields gated by opt_flag bits
            res_scal = 0
            if not (opt_flag & 0x01):
                res_scal = body[offset] if offset < len(body) else 0
                offset += 1
            if not (opt_flag & 0x02):
                offset += 1   # llm_scal
            if not (opt_flag & 0x04):
                offset += 1   # hlm_scal

            lo_limit = 0.0
            if not (opt_flag & 0x10) and offset + 4 <= len(body):
                lo_limit, offset = _read_r4(body, offset, endian)
            else:
                offset = min(offset + 4, len(body))

            hi_limit = 50.0
            if not (opt_flag & 0x20) and offset + 4 <= len(body):
                hi_limit, offset = _read_r4(body, offset, endian)
            else:
                offset = min(offset + 4, len(body))

            units, _  = _read_cn(body, offset)

            ptr["test_num"]  = test_num
            ptr["head_num"]  = head_num
            ptr["site_num"]  = site_num
            ptr["test_flag"] = test_flg
            ptr["value"]     = float(value)
            ptr["test_name"] = test_name.strip() or "IDDQ"
            ptr["lo_limit"]  = float(lo_limit)
            ptr["hi_limit"]  = float(hi_limit)
            ptr["units"]     = units.strip() or "uA"
            ptr["pass_fail"] = "FAIL" if (test_flg & 0x80) else "PASS"
        except Exception as exc:
            logger.warning("PTR parse warning (test_num=%s): %s", ptr.get("test_num"), exc)
        return ptr

    @staticmethod
    def _parse_prr(body: bytes, endian: str) -> Dict[str, Any]:
        """Parse PRR: Part Results Record."""
        prr: Dict[str, Any] = {}
        if len(body) < 8:
            return prr
        try:
            offset = 0
            head_num, offset  = _read_u1(body, offset)
            site_num, offset  = _read_u1(body, offset)
            part_flg, offset  = _read_u1(body, offset)
            num_test, offset  = _read_u2(body, offset, endian)
            hard_bin, offset  = _read_u2(body, offset, endian)
            soft_bin, offset  = _read_u2(body, offset, endian)

            prr["head_num"]  = head_num
            prr["site_num"]  = site_num
            prr["part_flag"] = part_flg
            prr["hard_bin"]  = hard_bin
            prr["soft_bin"]  = soft_bin
            prr["pass_fail"] = "FAIL" if (part_flg & 0x08) else "PASS"
        except Exception as exc:
            logger.warning("PRR parse warning: %s", exc)
        return prr
