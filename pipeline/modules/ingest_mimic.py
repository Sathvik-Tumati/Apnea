"""
pipeline/modules/ingest_mimic.py
================================
MIMIC-IV waveform ingestion — downloads records from PhysioNet, extracts
ECG / SpO2 / ABP / Resp signals, computes features, assigns pseudo-labels
from the ground-truth Resp channel, and persists everything to the pipeline DB.

Public API
----------
    ingest_mimic_records(n_records, run_id) -> List[Dict]

    Mirrors ingest_slpdb.ingest_slpdb_records() so train.py can call both
    the same way and merge their outputs cleanly.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import os

import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import resample as scipy_resample

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

from pipeline.modules.config import (
    MIMIC_URL, MIMIC_CACHE_DIR,
    N_MIMIC_RECORDS,
    FS_ECG, FS_PPG, FS_RESP,
    SEGMENT_LEN_S,
    NumpyEncoder,
)
from pipeline.modules.features import (
    _extract_apnea_features,
    _detect_r_peaks,
    _bandpass,
    _compute_edr,
    _resp_flag_edr,
    HAS_EDR_V3,
)
from pipeline.db.database import (
    insert_apnea_ecg_plot,
    insert_apnea_features,
    insert_apnea_preprocessed,
    insert_apnea_raw,
    insert_apnea_segment,
    log_module,
)

logger = logging.getLogger(__name__)

# ── EDR v3 (optional enhanced version) ────────────────────────────────────────
try:
    from pipeline.compute_edr_fixed import compute_edr_v3 as _compute_edr_v3
except ImportError:
    _compute_edr_v3 = None


# ══════════════════════════════════════════════════════════════════════════════
#  MIMIC RECORD LOADER (WITH CACHING)
# ══════════════════════════════════════════════════════════════════════════════

def _load_mimic_records(n: int = N_MIMIC_RECORDS) -> List[str]:
    """
    Fetch the list of valid MIMIC-IV waveform record paths from PhysioNet.
    Returns up to *n* record paths (e.g. 'p10/p1000000/83404622').
    """
    if not HAS_WFDB:
        logger.error("[MIMIC] wfdb not installed — pip install wfdb")
        return []
    try:
        import urllib.request
        with urllib.request.urlopen(MIMIC_URL + "RECORDS", timeout=30) as r:
            lines = r.read().decode().splitlines()
        valid_paths: List[str] = []
        for ln in lines:
            if not ln.strip():
                continue
            dir_path = ln.strip()
            try:
                with urllib.request.urlopen(
                    MIMIC_URL + dir_path + "RECORDS", timeout=10
                ) as ir:
                    for iln in ir.read().decode().splitlines():
                        if "layout" not in iln:
                            valid_paths.append(dir_path + iln.strip())
                            if len(valid_paths) >= n:
                                return valid_paths
            except Exception:
                continue
        return valid_paths
    except Exception as exc:
        logger.error("[MIMIC] Failed to load record list: %s", exc)
        return []


def _load_mimic_record_with_cache(
    record_path: str, pn_dir: str
) -> Optional[Any]:
    """
    Load a single MIMIC-IV waveform record, using the local disk cache first.
    Falls back to streaming from PhysioNet and caching the result.
    Returns a wfdb.Record object, or None on failure.
    """
    record_name = record_path.split("/")[-1]
    local_hea   = os.path.join(MIMIC_CACHE_DIR, f"{record_name}.hea")

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if os.path.exists(local_hea):
        try:
            record = wfdb.rdrecord(os.path.join(MIMIC_CACHE_DIR, record_name))
            logger.debug("[MIMIC] Loaded from cache: %s", record_name)
            return record
        except Exception as e:
            logger.warning("[MIMIC] Cache read failed for %s: %s", record_name, e)

    # ── Download ───────────────────────────────────────────────────────────────
    os.makedirs(MIMIC_CACHE_DIR, exist_ok=True)
    logger.info("[MIMIC] Caching %s ...", record_name)
    try:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=96000)
        try:
            wfdb.wrsamp(
                record_name,
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=record.p_signal,
                fmt=record.fmt,
                write_dir=MIMIC_CACHE_DIR,
            )
            logger.info("[MIMIC] Cached %s", record_name)
        except Exception:
            pass  # Cache write failure is non-fatal
        return record
    except Exception as e:
        logger.error("[MIMIC] Failed to load %s: %s", record_name, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_mimic_baseline(
    ecg_full:      np.ndarray,
    pleth_full:    np.ndarray,
    abp_full:      np.ndarray,
    resp_gt_full:  Optional[np.ndarray],
    n_segs:        int,
) -> Dict:
    """
    Compute per-record physiological baseline from the first clean segments.
    Used to contextualise features (e.g. baseline SpO2, baseline RMSSD).
    Returns a dict with baseline_spo2, baseline_rmssd, baseline_rr_ms,
    baseline_map_std, baseline_sbp.
    """
    spe = FS_ECG  * SEGMENT_LEN_S
    spp = FS_PPG  * SEGMENT_LEN_S
    spr = FS_RESP * SEGMENT_LEN_S

    baseline_windows: List[Dict] = []
    for i in range(n_segs):
        ecg_seg   = _bandpass(ecg_full[i * spe: (i + 1) * spe], FS_ECG)
        pleth_seg = pleth_full[i * spp: (i + 1) * spp]
        abp_seg   = abp_full[i * spe: (i + 1) * spe]

        rp    = _detect_r_peaks(ecg_seg, FS_ECG)
        rr_ms = (
            np.diff(rp) / FS_ECG * 1000.0
            if len(rp) >= 3 else np.array([833.0])
        )
        rmssd_w = (
            float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
            if len(rr_ms) >= 2 else 35.0
        )

        if _compute_edr_v3 is not None:
            resp_seg, _, _ = _compute_edr_v3(ecg_seg, rp, FS_ECG, FS_RESP, SEGMENT_LEN_S)
        else:
            resp_seg = _compute_edr(ecg_seg, rp, FS_ECG, FS_RESP)

        resp_gt_seg_bl = (
            resp_gt_full[i * spr: (i + 1) * spr]
            if resp_gt_full is not None and (i + 1) * spr <= len(resp_gt_full)
            else None
        )
        resp_for_bl = resp_gt_seg_bl if resp_gt_seg_bl is not None else resp_seg
        map_sig     = (
            pd.Series(abp_seg)
            .rolling(int(FS_ECG * 2), min_periods=1)
            .mean()
            .values
        )

        baseline_windows.append({
            "spo2_mean":  float(np.mean(pleth_seg)),
            "rmssd":      rmssd_w,
            "rr_mean":    float(np.mean(rr_ms)),
            "map_std":    float(np.std(map_sig)),
            "sbp_max":    float(np.max(abp_seg)),
            "individually_clean": (
                not _resp_flag_edr(resp_for_bl, FS_RESP)
                and float(np.min(pleth_seg)) > 90.0
            ),
        })

    clean_wins = [w for w in baseline_windows if w["individually_clean"]][:5]
    if clean_wins:
        return {
            "baseline_spo2":    float(np.mean([w["spo2_mean"] for w in clean_wins])),
            "baseline_rmssd":   float(np.mean([w["rmssd"]     for w in clean_wins])),
            "baseline_rr_ms":   float(np.mean([w["rr_mean"]   for w in clean_wins])),
            "baseline_map_std": float(np.mean([w["map_std"]   for w in clean_wins])),
            "baseline_sbp":     float(np.mean([w["sbp_max"]   for w in clean_wins])),
        }
    return {
        "baseline_spo2": 97.0,
        "baseline_rmssd": 35.0,
        "baseline_rr_ms": 833.0,
        "baseline_map_std": 5.0,
        "baseline_sbp": 120.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-RECORD PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def _process_mimic_record(
    rec,
    record_name: str,
    run_id:      str,
    plots_per_record: Dict[str, int],
) -> List[Dict]:
    """
    Process one wfdb.Record object:
      - validate required channels (II, Pleth)
      - resample to pipeline target rates
      - compute per-record physiological baseline
      - extract features per 30-s segment
      - persist raw/preprocessed/features/plots to DB
      - return list of segment dicts

    Returns [] if the record is unusable.
    """
    spe = FS_ECG  * SEGMENT_LEN_S
    spp = FS_PPG  * SEGMENT_LEN_S
    spr = FS_RESP * SEGMENT_LEN_S

    sig_map = {name: idx for idx, name in enumerate(rec.sig_name)}

    # ── Require II (ECG) and Pleth (SpO2 proxy) ───────────────────────────────
    if any(s not in sig_map for s in ["II", "Pleth"]):
        logger.warning("[MIMIC] %s missing II or Pleth — skipping", record_name)
        return []

    signals  = rec.p_signal
    fs_orig  = rec.fs

    ecg_orig   = signals[:, sig_map["II"]]
    pleth_orig = signals[:, sig_map["Pleth"]]
    has_abp    = "ABP" in sig_map
    abp_orig   = signals[:, sig_map["ABP"]] if has_abp else np.zeros(signals.shape[0])
    has_gt_resp = "Resp" in sig_map
    resp_orig   = signals[:, sig_map["Resp"]] if has_gt_resp else None

    if has_gt_resp:
        logger.info("[MIMIC] %s: GT Resp channel present", record_name)

    # ── Fill NaNs ─────────────────────────────────────────────────────────────
    for arr in (ecg_orig, pleth_orig, abp_orig):
        m = np.isnan(arr)
        arr[m] = np.nanmean(arr) if not m.all() else 0.0
    if resp_orig is not None:
        m = np.isnan(resp_orig)
        resp_orig[m] = np.nanmean(resp_orig) if not m.all() else 0.0

    # ── Resample to target rates ───────────────────────────────────────────────
    ecg_full   = scipy_resample(ecg_orig,   int(len(ecg_orig)   * FS_ECG  / fs_orig))
    pleth_full = scipy_resample(pleth_orig, int(len(pleth_orig) * FS_PPG  / fs_orig))
    abp_full   = scipy_resample(abp_orig,   int(len(abp_orig)   * FS_ECG  / fs_orig))
    resp_gt_full = (
        scipy_resample(resp_orig, int(len(resp_orig) * FS_RESP / fs_orig))
        if resp_orig is not None else None
    )

    n_segs = min(len(ecg_full) // spe, 10)
    if n_segs == 0:
        return []

    # ── Compute physiological baseline ────────────────────────────────────────
    baseline = _compute_mimic_baseline(
        ecg_full, pleth_full, abp_full, resp_gt_full, n_segs
    )

    # ── Per-segment processing ─────────────────────────────────────────────────
    rows: List[Dict]  = []
    last_spo2_val     = 97.0

    for i in range(n_segs):
        ecg_seg = _bandpass(ecg_full[i * spe: (i + 1) * spe], FS_ECG)
        abp_seg = abp_full[i * spe: (i + 1) * spe]

        # Simulate intermittent SpO2 readings (as on real monitors)
        take_reading = (i % np.random.randint(6, 11)) == 0
        if take_reading:
            pleth_seg     = pleth_full[i * spp: (i + 1) * spp]
            last_spo2_val = float(np.mean(pleth_seg))
        else:
            pleth_seg = np.full(spp, last_spo2_val)

        r_peaks = _detect_r_peaks(ecg_seg, FS_ECG)

        # EDR (ECG-Derived Respiration)
        if _compute_edr_v3 is not None:
            resp_seg, edr_bpm, edr_quality = _compute_edr_v3(
                ecg_seg, r_peaks, FS_ECG, FS_RESP, SEGMENT_LEN_S
            )
        else:
            resp_seg    = _compute_edr(ecg_seg, r_peaks, FS_ECG, FS_RESP)
            edr_bpm     = None
            edr_quality = None

        resp_gt_seg = (
            resp_gt_full[i * spr: (i + 1) * spr]
            if resp_gt_full is not None and (i + 1) * spr <= len(resp_gt_full)
            else None
        )

        # ── Persist raw signals ────────────────────────────────────────────────
        raw_id = insert_apnea_raw(
            record_name, i, ecg_seg, pleth_seg, resp_seg, abp_seg, FS_ECG
        )

        spo2_smooth = (
            pd.Series(pleth_seg)
            .rolling(int(FS_PPG * 2), center=True, min_periods=1)
            .median()
            .values
        )
        resp_smooth = (
            pd.Series(resp_seg)
            .rolling(FS_RESP, center=True, min_periods=1)
            .median()
            .values
        )
        rr_ms  = (
            np.diff(r_peaks) / FS_ECG * 1000.0
            if len(r_peaks) >= 2 else np.array([0.0])
        )
        pre_id = insert_apnea_preprocessed(
            raw_id, ecg_seg, r_peaks, spo2_smooth, resp_smooth,
            float(np.mean(rr_ms)), float(np.std(rr_ms)),
            int(len(r_peaks)), float(np.median(rr_ms)),
        )

        # ── Feature extraction ─────────────────────────────────────────────────
        feats = _extract_apnea_features(
            ecg_seg, pleth_seg, resp_seg, abp_seg, r_peaks, baseline,
            edr_bpm=edr_bpm, edr_quality=edr_quality,
            resp_gt=resp_gt_seg, has_abp_signal=has_abp,
        )
        insert_apnea_features(pre_id, json.dumps(feats, cls=NumpyEncoder))

        # ── Persist segment row ────────────────────────────────────────────────
        seg_row = {
            "record":      record_name,
            "segment_idx": i,
            "run_id":      run_id,
            **feats,
        }
        insert_apnea_segment(seg_row)
        rows.append(seg_row)

        # ── Save ECG plot for first 2 labelled segments ────────────────────────
        if plots_per_record.get(record_name, 0) < 2 and feats["true_label"] in (0, 1):
            def _d(sig, n):
                return np.interp(
                    np.linspace(0, 1, n), np.linspace(0, 1, len(sig)), sig
                )
            insert_apnea_ecg_plot(
                record_name, i, ecg_seg, r_peaks,
                _d(spo2_smooth, SEGMENT_LEN_S),
                _d(resp_smooth, SEGMENT_LEN_S),
                _d(abp_seg, SEGMENT_LEN_S),
                FS_ECG, feats["true_label"], feats["label_confidence"],
            )
            plots_per_record[record_name] = plots_per_record.get(record_name, 0) + 1

    logger.info("[MIMIC] %s: %d segments ingested", record_name, n_segs)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — mirrors ingest_slpdb.ingest_slpdb_records()
# ══════════════════════════════════════════════════════════════════════════════

def ingest_mimic_records(
    n_records: int = N_MIMIC_RECORDS,
    run_id:    str = "default",
) -> List[Dict]:
    """
    Download and process up to *n_records* MIMIC-IV waveform records.

    Returns a list of segment dicts (one per 30-s window), each containing
    all APNEA_FEATURE_COLS plus 'record', 'segment_idx', 'run_id',
    'true_label', and 'label_confidence'.

    Persists raw signals, preprocessed signals, features, and ECG plots
    to the pipeline SQLite DB for traceability.
    """
    if not HAS_WFDB:
        logger.error("[MIMIC] wfdb not installed — pip install wfdb")
        log_module("apnea", "ingest_mimic", "failed", "wfdb not installed", 0)
        return []

    log_module("apnea", "ingest_mimic", "started")
    record_paths = _load_mimic_records(n_records)

    if not record_paths:
        logger.error("[MIMIC] No record paths fetched from PhysioNet")
        log_module("apnea", "ingest_mimic", "failed", "No MIMIC records fetched", 0)
        return []

    logger.info("[MIMIC] Ingesting %d records ...", len(record_paths))

    all_rows:         List[Dict]      = []
    plots_per_record: Dict[str, int]  = {}
    failed_records:   List[str]       = []

    for rec_path in record_paths:
        record_name = rec_path.split("/")[-1]
        pn_dir      = "mimic4wdb/0.1.0/" + "/".join(rec_path.split("/")[:-1])

        rec = _load_mimic_record_with_cache(rec_path, pn_dir)
        if rec is None:
            failed_records.append(record_name)
            continue

        try:
            rows = _process_mimic_record(rec, record_name, run_id, plots_per_record)
            all_rows.extend(rows)
        except Exception as exc:
            logger.error("[MIMIC] Error processing %s: %s", record_name, exc)
            failed_records.append(record_name)
            continue

    if failed_records:
        logger.warning(
            "[MIMIC] Failed to process %d/%d records: %s",
            len(failed_records), len(record_paths), failed_records,
        )

    log_module(
        "apnea", "ingest_mimic", "done",
        f"MIMIC segments ingested ({len(failed_records)} records failed)",
        len(all_rows),
    )
    logger.info("[MIMIC] Total MIMIC segments ingested: %d", len(all_rows))
    return all_rows
