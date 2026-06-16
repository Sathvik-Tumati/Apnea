from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import resample as scipy_resample
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
from pipeline.modules.config import *
from pipeline.modules.features import _extract_apnea_features, _detect_r_peaks
from pipeline.db.database import insert_apnea_ecg_plot, insert_apnea_features, insert_apnea_preprocessed, insert_apnea_raw, insert_apnea_segment, log_module
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  MIMIC RECORD LOADER (WITH CACHING)
# ══════════════════════════════════════════════════════════════════════════════

def _load_mimic_records(n: int = N_MIMIC_RECORDS) -> List[str]:
    """Load MIMIC record paths with caching support."""
    if not HAS_WFDB:
        logger.error("[MIMIC] wfdb not installed")
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
                with urllib.request.urlopen(MIMIC_URL + dir_path + "RECORDS", timeout=10) as ir:
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


def _load_mimic_record_with_cache(record_path: str, pn_dir: str) -> Optional[Any]:
    """Load a MIMIC record, using local cache if available."""
    record_name = record_path.split("/")[-1]
    local_hea = os.path.join(MIMIC_CACHE_DIR, f"{record_name}.hea")
    
    # Check cache
    if os.path.exists(local_hea):
        try:
            record = wfdb.rdrecord(os.path.join(MIMIC_CACHE_DIR, record_name))
            logger.debug(f"[MIMIC] Loaded from cache: {record_name}")
            return record
        except Exception as e:
            logger.warning(f"[MIMIC] Cache read failed for {record_name}: {e}")
    
    # Download to cache
    os.makedirs(MIMIC_CACHE_DIR, exist_ok=True)
    logger.info(f"[MIMIC] Caching {record_name} ...")
    
    try:
        # Stream and cache by reading the record (wfdb may cache automatically)
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=96000)
        
        # Try to save a copy to cache directory
        try:
            wfdb.wrsamp(record_name, fs=record.fs, units=record.units,
                       sig_name=record.sig_name, p_signal=record.p_signal,
                       fmt=record.fmt, write_dir=MIMIC_CACHE_DIR)
            logger.info(f"[MIMIC] Cached {record_name}")
        except Exception:
            pass
        
        return record
    except Exception as e:
        logger.error(f"[MIMIC] Failed to load {record_name}: {e}")
        return None


