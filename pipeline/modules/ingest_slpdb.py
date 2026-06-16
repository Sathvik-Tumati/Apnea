from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
from pipeline.modules.config import *
from pipeline.modules.features import _extract_features_slpdb, _bandpass
from pipeline.db.database import fetch_apnea_segments, insert_apnea_segment, log_module
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  SLPDB DATA LOADING (WITH CACHING & IMPROVED ERROR HANDLING)
# ══════════════════════════════════════════════════════════════════════════════

def _load_slpdb_record(record_name: str) -> Tuple[Optional[np.ndarray], Any, int]:
    """
    Load one SLPDB record, using local cache first.
    Returns (ecg_125hz_bandpassed, annotation, fs_orig) or (None, None, 0).
    """
    if not HAS_WFDB:
        return None, None, 0

    os.makedirs(SLPDB_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(SLPDB_CACHE_DIR, record_name)
    
    # Check if we already have a valid cache
    cache_valid = False
    record = None
    
    if os.path.exists(local_path + ".hea") and os.path.exists(local_path + ".dat"):
        # Verify the .dat file has reasonable size (> 1MB)
        dat_size = os.path.getsize(local_path + ".dat")
        if dat_size > 1024 * 1024:  # At least 1MB
            try:
                record = wfdb.rdrecord(local_path)
                cache_valid = True
                logger.info(f"[SLPDB] Loaded from cache: {record_name} ({dat_size/1024/1024:.1f}MB)")
            except Exception as e:
                logger.warning(f"[SLPDB] Cache corrupted for {record_name}: {e}, will re-download")
                # Remove corrupted files
                try:
                    os.remove(local_path + ".hea")
                    os.remove(local_path + ".dat")
                except:
                    pass
    
    if cache_valid and record is not None:
        sig_map = SLPDB_SIGNAL_MAP.get(record_name, SLPDB_SIGNAL_MAP["default"])
        ecg_idx = sig_map["ecg"]
        if ecg_idx >= record.p_signal.shape[1]:
            logger.warning("[SLPDB] %s: ECG index %d out of range", record_name, ecg_idx)
            return None, None, 0
        
        ecg_raw = record.p_signal[:, ecg_idx].astype(float)
        fs_orig = record.fs
        
        if fs_orig != FS_ECG:
            n_target = int(len(ecg_raw) * FS_ECG / fs_orig)
            ecg_raw = scipy_resample(ecg_raw, n_target)
        
        ecg = _bandpass(ecg_raw, FS_ECG)
        
        # Load .st annotations
        try:
            if os.path.exists(local_path + ".st"):
                ann = wfdb.rdann(local_path, "st")
            else:
                ann = wfdb.rdann(record_name, "st", pn_dir="slpdb/1.0.0")
        except Exception as exc:
            logger.warning("[SLPDB] %s: no .st annotations (%s)", record_name, exc)
            ann = None
        
        return ecg, ann, fs_orig
    
    # Try downloading with retries and fallback
    logger.info(f"[SLPDB] Attempting to download {record_name}...")
    
    # Method 1: Try direct wfdb streaming (original method)
    try:
        logger.info(f"[SLPDB] Method 1: Streaming {record_name} ...")
        record = wfdb.rdrecord(record_name, pn_dir="slpdb/1.0.0")
        if record is not None:
            # Save to cache for future use
            try:
                wfdb.wrsamp(record_name, fs=record.fs, units=record.units,
                           sig_name=record.sig_name, p_signal=record.p_signal,
                           fmt=record.fmt, write_dir=SLPDB_CACHE_DIR)
                logger.info(f"[SLPDB] Cached {record_name}")
            except Exception as e:
                logger.warning(f"[SLPDB] Failed to cache {record_name}: {e}")
    except Exception as e:
        logger.warning(f"[SLPDB] Method 1 failed for {record_name}: {e}")
    
    # Method 2: Try downloading just the needed samples (sampto parameter)
    if record is None:
        try:
            logger.info(f"[SLPDB] Method 2: Streaming limited samples for {record_name}...")
            # Limit to first 10 minutes (FS_ECG * 600 samples at original fs)
            max_samples = FS_ECG * 600  # 10 minutes at 125Hz = 75,000 samples
            record = wfdb.rdrecord(record_name, pn_dir="slpdb/1.0.0", sampto=max_samples)
            if record is not None:
                logger.info(f"[SLPDB] Successfully loaded {record_name} (limited to {max_samples} samples)")
        except Exception as e:
            logger.warning(f"[SLPDB] Method 2 failed for {record_name}: {e}")
    
    if record is None:
        logger.error(f"[SLPDB] All methods failed for {record_name}")
        return None, None, 0
    
    # Process the record
    sig_map = SLPDB_SIGNAL_MAP.get(record_name, SLPDB_SIGNAL_MAP["default"])
    ecg_idx = sig_map["ecg"]
    if ecg_idx >= record.p_signal.shape[1]:
        logger.warning("[SLPDB] %s: ECG index %d out of range", record_name, ecg_idx)
        return None, None, 0
    
    ecg_raw = record.p_signal[:, ecg_idx].astype(float)
    fs_orig = record.fs
    
    if fs_orig != FS_ECG:
        n_target = int(len(ecg_raw) * FS_ECG / fs_orig)
        ecg_raw = scipy_resample(ecg_raw, n_target)
    
    ecg = _bandpass(ecg_raw, FS_ECG)
    
    # Load .st annotations
    try:
        if os.path.exists(local_path + ".st"):
            ann = wfdb.rdann(local_path, "st")
        else:
            ann = wfdb.rdann(record_name, "st", pn_dir="slpdb/1.0.0")
    except Exception as exc:
        logger.warning("[SLPDB] %s: no .st annotations (%s)", record_name, exc)
        ann = None
    
    return ecg, ann, fs_orig


def _get_slpdb_segment_labels(
    annotation, n_segments: int, fs_orig: int,
) -> np.ndarray:
    """
    Build binary label array from SLPDB .st aux_note annotations.
    Uses exact token matching to avoid 'O' matching inside 'NORMAL'.
    """
    labels = np.zeros(n_segments, dtype=int)
    if annotation is None:
        return labels
    try:
        samples    = np.array(annotation.sample, dtype=float)
        aux_notes  = annotation.aux_note if hasattr(annotation, "aux_note") else []
        if not aux_notes:
            return labels

        ann_samples_125 = (samples * FS_ECG / fs_orig).astype(int)
        for ann_s, aux in zip(ann_samples_125, aux_notes):
            # Exact token matching — avoids 'O' hitting 'NORMAL'
            tokens   = set(str(aux).upper().strip().split())
            is_apnea = bool(SLPDB_APNEA_TOKENS.intersection(tokens))
            if not is_apnea:
                continue
            ann_end = ann_s + FS_ECG * 10   # min 10-second event
            seg_s   = max(0, ann_s // SAMPLES_PER_SEG - 1)
            seg_e   = min(n_segments - 1, ann_end // SAMPLES_PER_SEG + 1)
            for si in range(seg_s, seg_e + 1):
                ss = si * SAMPLES_PER_SEG
                se = ss + SAMPLES_PER_SEG
                if ann_s < se and ann_end > ss:
                    labels[si] = 1
    except Exception as exc:
        logger.warning("[SLPDB] Label extraction error: %s", exc)
    return labels


def ingest_slpdb_records(
    records: List[str],
    run_id: str,
    max_segments_per_record: int = 80,
) -> List[Dict]:
    """
    Load SLPDB records, extract ECG-only features + GT labels,
    return list of segment dicts compatible with MIMIC segment rows.
    """
    all_rows: List[Dict] = []
    failed_records = []
    
    for record_name in records:
        try:
            ecg, annotation, fs_orig = _load_slpdb_record(record_name)
            if ecg is None:
                failed_records.append(record_name)
                continue

            n_segs = min(len(ecg) // SAMPLES_PER_SEG, max_segments_per_record)
            if n_segs == 0:
                logger.warning("[SLPDB] %s: not enough samples", record_name)
                continue

            gt_labels = _get_slpdb_segment_labels(annotation, n_segs, fs_orig)
            n_apnea   = int(gt_labels.sum())
            logger.info(
                "[SLPDB] %s: %d segments | apnea=%d normal=%d",
                record_name, n_segs, n_apnea, n_segs - n_apnea,
            )

            for i in range(n_segs):
                s   = i * SAMPLES_PER_SEG
                seg = ecg[s: s + SAMPLES_PER_SEG]
                f   = _extract_features_slpdb(seg)
                row = {
                    "record":      record_name,
                    "segment_idx": i,
                    "run_id":      run_id,
                    "true_label":  int(gt_labels[i]),
                    **f,
                }
                # Columns required by fetch_apnea_segments / DB schema but absent in SLPDB
                row.setdefault("edr_quality_ok", 0)
                row.setdefault("edr_snr",        0.0)
                row.setdefault("resp_flag",       0)
                row.setdefault("spo2_flag",       0)
                row.setdefault("hrv_flag",        0)
                row.setdefault("abp_flag",        0)
                row.setdefault("signals_positive", 0)
                row.setdefault("label_confidence", "slpdb_annotation")

                # Persist to DB for traceability (best-effort)
                try:
                    insert_apnea_segment(row)
                except Exception:
                    pass

                all_rows.append(row)

            logger.info("[SLPDB] %s: ingested %d segments", record_name, n_segs)
        except Exception as e:
            logger.error(f"[SLPDB] Failed to process {record_name}: {e}")
            failed_records.append(record_name)
            continue
    
    if failed_records:
        logger.warning(f"[SLPDB] Failed to load {len(failed_records)}/{len(records)} records: {failed_records}")
    
    logger.info(f"[SLPDB] Total SLPDB segments ingested: {len(all_rows)}")
    return all_rows


