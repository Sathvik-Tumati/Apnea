"""
supabase_infer.py
==================
Pulls ECG + SpO2 data for a given admission_id from Supabase (tables
populated by the Supabase importer: ecg_stream, spo2_stream, devices),
assembles 30-second segments, runs the apnea inference pipeline, and
writes results to Supabase (apnea_results).

This is the Supabase-native replacement for mongo_infer.py. The SSH
tunnel / MongoDB plumbing has been removed entirely; everything reads
and writes through the Supabase REST client. Signal processing,
CSV building, and inference/write-back logic are unchanged from
mongo_infer.py.

Usage
-----
  python automation/supabase_infer.py --admission ADM1819906487
  python automation/supabase_infer.py --since 24h
  python automation/supabase_infer.py --from 2026-06-01 --to 2026-06-15
  python automation/supabase_infer.py --since 24h --dry-run
  python automation/supabase_infer.py --since 24h --write-supabase
  python automation/supabase_infer.py --since 24h --write-supabase --reprocess

Environment variables (set in .env at project root)
----------------------------------------------------
SUPABASE_URL   https://your-project.supabase.co
SUPABASE_KEY   your-service-role-or-anon-key
MODEL_PATH     /path/to/apnea_model.keras      (default: apnea_model.keras)
SCALER_PATH    /path/to/apnea_scaler.pkl       (default: apnea_scaler.pkl)
THRESHOLD      0.45                             (default: 0.45 — matches infer.py)
OUTPUT_DIR     /path/to/output/                (default: infer_output/)

NOTE ON SCHEMA ASSUMPTIONS
---------------------------
This file assumes the Supabase schema written by the importer script
(Supabase Medical Data Importer):

  ecg_stream   : admission_id, utc_timestamp, ecg_data (array, ~125 samples/row)
  spo2_stream  : admission_id, utc_timestamp, spo2_value, pulse_rate, pi,
                 spo2_data, pulse_data, pi_data
  devices      : admission_id (unique), facility_id, patient_name, ...

`ecg_stream` and `spo2_stream` do NOT carry facility_id, so facility_id
is looked up from `devices` per admission. If your actual schema differs
(e.g. different column names, different row granularity), update
`_fetch_all_rows`, `extract_ecg_signal`, `extract_spo2_signal`, and
`_get_facility_id` accordingly — these are the only functions that talk
to the database.
"""

# ── .env loading ──────────────────────────────────────────────────────────────
from pathlib import Path as _Path
_env_file = _Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(_env_file)
    except ImportError:
        pass

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Sleep filter ───────────────────────────────────────────────────────────────
try:
    from ecg_sleep_filter import detect_sleep_segments, filter_to_sleep
    HAS_SLEEP_FILTER = True
except ImportError:
    try:
        from automation.ecg_sleep_filter import detect_sleep_segments, filter_to_sleep
        HAS_SLEEP_FILTER = True
    except ImportError:
        HAS_SLEEP_FILTER = False
        logger.warning("[SLEEP] ecg_sleep_filter.py not importable — sleep filtering disabled")

# ── Supabase config ────────────────────────────────────────────────────────────
SUPABASE_CONFIG = {
    "url": os.environ.get("SUPABASE_URL"),
    "key": os.environ.get("SUPABASE_KEY"),
}

# Table names — adjust here if your schema uses different names
ECG_TABLE     = os.environ.get("ECG_TABLE", "ecg_stream")
SPO2_TABLE    = os.environ.get("SPO2_TABLE", "spo2_stream")
DEVICES_TABLE = os.environ.get("DEVICES_TABLE", "devices")
RESULTS_TABLE = os.environ.get("RESULTS_TABLE", "apnea_results")

# Supabase REST caps page size; paginate in chunks of this size
PAGE_SIZE = 1000

# ── Dual-model config ─────────────────────────────────────────────────────────
# IMPORTANT: per pipeline.py, the two trained model pairs are:
#   BiLSTM   → apnea_model.keras          + apnea_scaler.pkl
#   XGBoost  → apnea_model_xgb_seq.pkl    + apnea_scaler_tree.pkl
# These defaults previously both pointed at the *same* .keras/.pkl pair
# (a leftover collision), which meant the "XGBoost" slot silently loaded
# the BiLSTM model instead, and apnea_model_xgb_seq.pkl was never used.
# Do not let these two pairs default to the same path again.
BILSTM_MODEL_PATH  = os.environ.get("BILSTM_MODEL_PATH",  "apnea_model.keras")
BILSTM_SCALER_PATH = os.environ.get("BILSTM_SCALER_PATH", "apnea_scaler.pkl")

# ── Pipeline constants ────────────────────────────────────────────────────────
FS_ECG               = 125
FS_SPO2              = 1      # device-computed SpO2, 1 sample per spo2_stream row
SEGMENT_LEN_S        = 30
SAMPLES_SEG          = FS_ECG  * SEGMENT_LEN_S   # 3750 ECG samples per segment
SAMPLES_SPO2_SEG     = FS_SPO2 * SEGMENT_LEN_S   # 30 SpO2 samples per segment
SAMPLES_PER_ECG_DOC  = 125     # expected length of each ecg_stream.ecg_data row
MIN_SEGMENTS         = 11
MIN_DURATION_MINUTES = 30
COMPLETION_GAP_HOURS = 2

ECG_COLS    = [f"ecgData[{i}]"                              for i in range(SAMPLES_SEG)]
HR_COLS     = [f"analysis.segments[{i}].morphology.hr_bpm" for i in range(6)]
RHYTHM_COLS = [f"analysis.segments[{i}].rhythm_label"      for i in range(6)]
ECTOPY_COLS = [f"analysis.segments[{i}].ectopy_label"      for i in range(6)]

SPO2_FEATURE_COLS = [
    "spo2_mean", "spo2_min", "spo2_delta_index",
    "odi", "t90", "spo2_approx_entropy",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_supabase_client():
    """Return a supabase-py client. Exits on misconfiguration."""
    try:
        from supabase import create_client, Client
    except ImportError:
        logger.error("supabase-py not installed — pip install supabase")
        sys.exit(1)
    url = SUPABASE_CONFIG["url"]
    key = SUPABASE_CONFIG["key"]
    if not url:
        logger.error("[SUPABASE] SUPABASE_URL not set")
        sys.exit(1)
    if not key:
        logger.error("[SUPABASE] SUPABASE_KEY not set")
        sys.exit(1)
    client: Client = create_client(url, key)
    logger.info("[SUPABASE] Client initialised → %s", url)
    return client


def _fetch_all_rows(
    client, table: str, columns: str, admission_id: Optional[str] = None,
    ts_filter: Optional[Dict] = None, order_col: str = "utc_timestamp",
) -> List[Dict]:
    """
    Paginate through a Supabase table using .range(), since the REST API
    caps each response at PAGE_SIZE rows. Returns all matching rows
    ordered by order_col ascending.
    """
    rows: List[Dict] = []
    start = 0
    while True:
        q = client.table(table).select(columns)
        if admission_id is not None:
            q = q.eq("admission_id", admission_id)
        if ts_filter:
            if "$gte" in ts_filter:
                q = q.gte(order_col, ts_filter["$gte"])
            if "$lte" in ts_filter:
                q = q.lte(order_col, ts_filter["$lte"])
        q = q.order(order_col, desc=False).range(start, start + PAGE_SIZE - 1)
        resp = q.execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def _get_facility_id(client, admission_id: str) -> str:
    """devices.admission_id is a unique key (see importer's upload_device)."""
    try:
        resp = (
            client.table(DEVICES_TABLE)
            .select("facility_id")
            .eq("admission_id", admission_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if rows:
            return rows[0].get("facility_id") or ""
    except Exception as e:
        logger.warning("[DEVICES] facility_id lookup failed for %s: %s", admission_id, e)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  ADMISSION DISCOVERY  (Postgres has no Mongo-style aggregation pipeline,
#  so this is done client-side in pandas after pulling timestamp columns.)
# ══════════════════════════════════════════════════════════════════════════════

def find_completed_admissions(
    client,
    since_hours: Optional[float] = None,
    date_from:   Optional[datetime] = None,
    date_to:     Optional[datetime] = None,
) -> List[Dict]:
    """
    Return admissions that have ended (no new ECG data for COMPLETION_GAP_HOURS)
    and have enough data to run inference.

    Pulls (admission_id, utc_timestamp) from ecg_stream for the requested
    window and aggregates min/max/count per admission_id in pandas. This
    avoids relying on a Postgres-side GROUP BY RPC, at the cost of pulling
    more rows than a true aggregation would — fine at moderate table sizes,
    but consider a Postgres function (e.g. via client.rpc(...)) if this
    table grows large.
    """
    since_cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)
                    if since_hours else None)
    from_cutoff  = (date_from.replace(tzinfo=timezone.utc) if date_from else None)

    ts_filter: Dict = {}
    if since_cutoff and from_cutoff:
        chosen = max(since_cutoff, from_cutoff)
        logger.warning(
            "[DISCOVER] Both --since and --from given — using later bound: %s", chosen)
        ts_filter["$gte"] = chosen.isoformat()
    elif since_cutoff:
        ts_filter["$gte"] = since_cutoff.isoformat()
    elif from_cutoff:
        ts_filter["$gte"] = from_cutoff.isoformat()
    if date_to:
        ts_filter["$lte"] = date_to.replace(tzinfo=timezone.utc).isoformat()

    logger.info("[DISCOVER] Pulling admission_id/utc_timestamp from %s ...", ECG_TABLE)
    rows = _fetch_all_rows(
        client, ECG_TABLE, "admission_id,utc_timestamp",
        ts_filter=ts_filter or None,
    )
    if not rows:
        logger.info("[DISCOVER] No rows found in window")
        return []

    df = pd.DataFrame(rows)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)

    grouped = df.groupby("admission_id")["utc_timestamp"].agg(["min", "max", "count"])
    grouped = grouped.rename(columns={"min": "first_ts", "max": "last_ts", "count": "n_docs"})
    grouped["duration_min"] = (
        (grouped["last_ts"] - grouped["first_ts"]).dt.total_seconds() / 60
    )

    now = datetime.now(timezone.utc)
    completed = []
    for admission_id, r in grouped.iterrows():
        last_ts = r["last_ts"].to_pydatetime()
        hours_since_last = (now - last_ts).total_seconds() / 3600
        has_enough_data  = r["duration_min"] >= MIN_DURATION_MINUTES
        recording_ended  = hours_since_last >= COMPLETION_GAP_HOURS
        if recording_ended and has_enough_data:
            facility_id = _get_facility_id(client, admission_id)
            completed.append({
                "admissionId":      admission_id,
                "facilityId":       facility_id,
                "first_ts":         r["first_ts"].to_pydatetime(),
                "last_ts":          last_ts,
                "n_docs":           int(r["n_docs"]),
                "duration_min":     float(r["duration_min"]),
                "hours_since_last": round(hours_since_last, 1),
            })
            logger.info("[DISCOVER] %s  %.0f min  %d docs  %.1fh ago  ✓ ELIGIBLE",
                        admission_id, r["duration_min"], r["n_docs"], hours_since_last)
        else:
            reason = []
            if not recording_ended:
                reason.append(f"still active ({hours_since_last:.1f}h since last)")
            if not has_enough_data:
                reason.append(f"too short ({r['duration_min']:.0f} min)")
            logger.info("[DISCOVER] %s  SKIP — %s", admission_id, " | ".join(reason))

    logger.info("[DISCOVER] %d / %d eligible", len(completed), len(grouped))
    return completed


# ══════════════════════════════════════════════════════════════════════════════
#  ECG EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_ecg_signal(
    client, admission_id: str
) -> Tuple[np.ndarray, List[datetime], List[int]]:
    """
    Pull all ecg_stream rows for this admission ordered by utc_timestamp.
    Returns (signal, doc_timestamps, doc_chunk_lengths).
    """
    logger.info("[ECG] Fetching for %s ...", admission_id)
    rows = _fetch_all_rows(
        client, ECG_TABLE, "utc_timestamp,ecg_data", admission_id=admission_id,
    )
    if not rows:
        logger.error("[ECG] No rows for %s", admission_id)
        return np.array([]), [], []

    logger.info("[ECG] %d rows retrieved", len(rows))
    chunks, timestamps, chunk_lengths, skipped = [], [], [], 0

    for row in rows:
        raw = row.get("ecg_data")
        if isinstance(raw, list) and len(raw) > 0:
            chunk = raw[0] if (len(raw) and isinstance(raw[0], list)) else raw
            if len(chunk) > 0:
                arr = np.array(chunk, dtype=float)
                chunks.append(arr)
                timestamps.append(row.get("utc_timestamp"))
                chunk_lengths.append(len(arr))
                continue
        skipped += 1

    if skipped:
        logger.warning("[ECG] Skipped %d rows with empty/missing ecg_data", skipped)
    if not chunks:
        return np.array([]), [], []

    lengths_arr = np.array(chunk_lengths)
    modal_len   = int(np.bincount(lengths_arr).argmax())
    if modal_len != SAMPLES_PER_ECG_DOC:
        logger.warning(
            "[ECG] Modal row length for %s is %d, expected %d. "
            "Proceeding using each row's actual length rather than a fixed "
            "assumption — if this is unexpected, check the importer's "
            "ecg_data format.", admission_id, modal_len, SAMPLES_PER_ECG_DOC)

    n_odd = int(np.sum(lengths_arr != modal_len))
    if n_odd:
        logger.warning("[ECG] %d / %d rows have non-standard length (edge packets)",
                       n_odd, len(lengths_arr))

    signal = np.concatenate(chunks)
    logger.info("[ECG] Assembled %d samples (%.1f min at %d Hz)",
                len(signal), len(signal) / FS_ECG / 60, FS_ECG)
    return signal, timestamps, chunk_lengths


# ══════════════════════════════════════════════════════════════════════════════
#  SPO2 EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_spo2_signal(
    client, admission_id: str
) -> Tuple[np.ndarray, List[datetime], List[int]]:
    """
    Pull device-computed SpO2 readings from spo2_stream (one spo2_value per
    row, ~1 Hz). Returns (spo2_signal, doc_timestamps, doc_chunk_lengths);
    doc_chunk_lengths is always a list of 1s (one sample per row).

    Returns (empty, [], []) if no data is found.
    """
    rows = _fetch_all_rows(
        client, SPO2_TABLE, "utc_timestamp,spo2_value", admission_id=admission_id,
    )
    if not rows:
        logger.warning("[SPO2] No %s rows for %s", SPO2_TABLE, admission_id)
        return np.array([]), [], []

    values, timestamps, chunk_lengths = [], [], []
    for row in rows:
        val = row.get("spo2_value")
        if val is not None:
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if 30.0 <= fval <= 105.0:      # physiological guard
                values.append(fval)
                timestamps.append(row.get("utc_timestamp"))
                chunk_lengths.append(1)

    if not values:
        logger.warning("[SPO2] All rows had out-of-range SpO2 for %s", admission_id)
        return np.array([]), [], []

    signal = np.array(values, dtype=float)
    logger.info("[SPO2] %d readings (~1 Hz, %.1f min)  range %.0f–%.0f%%",
                len(signal), len(signal) / FS_SPO2 / 60,
                signal.min(), signal.max())
    return signal, timestamps, chunk_lengths


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL HELPERS  (unchanged from mongo_infer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _bandpass(sig: np.ndarray, fs: int,
              lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    nyq = fs / 2.0
    hi  = min(hi, nyq - 0.1)
    lo  = min(lo, hi - 0.1)
    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    if not np.isfinite(ecg).all():
        ecg = np.where(np.isfinite(ecg), ecg, 0.0)
    try:
        import neurokit2 as nk
        _, info = nk.ecg_process(ecg, sampling_rate=fs)
        peaks   = info["ECG_R_Peaks"]
        if len(peaks) >= 2:
            rr_ms = np.diff(peaks) / fs * 1000.0
            peaks = peaks[np.concatenate([[True], (rr_ms >= 300) & (rr_ms <= 2000)])]
        return peaks
    except Exception:
        pass
    peaks, _ = find_peaks(ecg, distance=int(fs * 0.4), height=float(np.std(ecg)))
    return peaks


def _mean_hr(r_peaks: np.ndarray, fs: int) -> float:
    if len(r_peaks) < 2:
        return 0.0
    rr_samples = np.diff(r_peaks)
    rr_samples = rr_samples[rr_samples > 0]
    if len(rr_samples) == 0:
        return 0.0
    rr_ms = rr_samples / fs * 1000.0
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    if len(rr_ms) == 0:
        return 0.0
    return float(60000.0 / np.mean(rr_ms))


def _subseg_hrs(ecg_seg: np.ndarray, fs: int, n: int = 6) -> List[float]:
    chunk_len = len(ecg_seg) // n
    hrs = []
    for i in range(n):
        chunk = ecg_seg[i * chunk_len:(i + 1) * chunk_len]
        try:
            hrs.append(_mean_hr(_detect_r_peaks(_bandpass(chunk, fs), fs), fs))
        except Exception:
            hrs.append(0.0)
    return hrs


def _assess_signal_quality(
    raw_seg: np.ndarray,
    bp_seg:  np.ndarray,
    fs:      int,
    sub_hrs: List[float],
) -> str:
    if raw_seg is None or len(raw_seg) == 0:
        return "poor"
    finite = raw_seg[np.isfinite(raw_seg)]
    if len(finite) < 0.9 * len(raw_seg):
        return "poor"

    sd = float(np.std(finite))
    if sd < 1e-6:
        return "poor"

    lo, hi = float(finite.min()), float(finite.max())
    if hi > lo:
        if np.mean((finite <= lo + 1e-9) | (finite >= hi - 1e-9)) > 0.10:
            return "poor"

    window = max(1, int(fs * 2))
    if len(bp_seg) >= window:
        roll_std = pd.Series(bp_seg).rolling(window).std().dropna().values
        if len(roll_std) > 0 and float(roll_std.min()) < 1e-4 * (sd + 1e-9):
            return "poor"

    valid_hrs = [h for h in sub_hrs if h > 0]
    if len(valid_hrs) < len(sub_hrs) // 2:
        return "poor"
    if any(h < 20 or h > 220 for h in valid_hrs):
        return "poor"

    return "acceptable"


# ══════════════════════════════════════════════════════════════════════════════
#  TIME ALIGNMENT HELPERS  (unchanged from mongo_infer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_ts(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        return pd.to_datetime(ts, utc=True).to_pydatetime()
    except Exception:
        return None


def _build_time_index(
    doc_timestamps: List, doc_lengths: List[int]
) -> Optional[Tuple[List[Optional[datetime]], List[int]]]:
    starts, cum, running, n_valid = [], [], 0, 0
    for ts, length in zip(doc_timestamps, doc_lengths):
        norm = _normalize_ts(ts)
        starts.append(norm)
        cum.append(running)
        running += length
        if norm is not None:
            n_valid += 1
    return (starts, cum) if n_valid >= 1 else None


def _sample_index_for_time(
    target_ts:    datetime,
    starts:       List[Optional[datetime]],
    cum:          List[int],
    doc_lengths:  List[int],
    fs:           int,
) -> Optional[int]:
    best = None
    for i, ts in enumerate(starts):
        if ts is not None and ts <= target_ts:
            best = i
        elif ts is not None and ts > target_ts:
            break
    if best is None:
        return None
    offset_s       = (target_ts - starts[best]).total_seconds()
    offset_samples = int(round(offset_s * fs))
    next_cum = cum[best + 1] if (best + 1) < len(cum) else (cum[best] + doc_lengths[best])
    max_offset = max(0, next_cum - cum[best] - 1)
    offset_samples = max(0, min(offset_samples, max_offset))
    return cum[best] + offset_samples


def _timestamp_for_sample(
    sample_idx:  int,
    starts:      List[Optional[datetime]],
    cum:         List[int],
    doc_lengths: List[int],
    fs:          int,
) -> Optional[datetime]:
    n = len(starts)
    if n == 0:
        return None
    cum_arr = np.asarray(cum)
    doc_idx = max(0, int(np.searchsorted(cum_arr, sample_idx, side="right")) - 1)

    if starts[doc_idx] is not None:
        offset_s = (sample_idx - cum[doc_idx]) / fs
        return starts[doc_idx] + timedelta(seconds=offset_s)

    for radius in range(1, n):
        for j in (doc_idx - radius, doc_idx + radius):
            if 0 <= j < n and starts[j] is not None:
                offset_s = (sample_idx - cum[j]) / fs
                return starts[j] + timedelta(seconds=offset_s)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SPO2 FEATURE COMPUTATION  (unchanged from mongo_infer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _approx_entropy(sig: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    n  = len(sig)
    sd = float(np.std(sig))
    if n < 2 * m + 2 or sd < 1e-9:
        return 0.0
    r = r_factor * sd

    def _phi(mv: int) -> float:
        templates = np.array([sig[i:i + mv] for i in range(n - mv + 1)])
        count = np.sum(
            np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2) <= r,
            axis=1,
        )
        return float(np.mean(np.log(count / (n - mv + 1) + 1e-10)))

    return abs(_phi(m) - _phi(m + 1))


def _compute_global_spo2_baseline(spo2_signal: np.ndarray) -> float:
    finite = spo2_signal[np.isfinite(spo2_signal)]
    if len(finite) < FS_SPO2 * 60:
        return 97.0
    return float(np.percentile(np.clip(finite, 50.0, 100.0), 90))


_SPO2_DEFAULTS = {
    "spo2_mean": 97.0, "spo2_min": 97.0, "spo2_delta_index": 0.0,
    "odi": 0.0, "t90": 0.0, "spo2_approx_entropy": 0.0,
    "has_spo2": 0,
}


def _compute_spo2_features(
    spo2_seg:          np.ndarray,
    baseline_override: Optional[float] = None,
) -> Dict:
    if spo2_seg is None or len(spo2_seg) < 10:
        return dict(_SPO2_DEFAULTS)

    seg = np.clip(spo2_seg.astype(float), 50.0, 100.0)
    seg = pd.Series(seg).ffill().bfill().values
    if not np.isfinite(seg).all() or np.std(seg) < 1e-9:
        return dict(_SPO2_DEFAULTS)

    spo2_mean        = float(np.mean(seg))
    spo2_min         = float(np.min(seg))
    spo2_delta_index = float(np.mean(np.abs(np.diff(seg))))
    t90              = float(np.mean(seg < 90.0))

    baseline = baseline_override if baseline_override is not None else spo2_mean
    below    = seg < (baseline - 3.0)
    n_events = 0
    in_event = False
    for v in below:
        if v and not in_event:
            n_events += 1
            in_event  = True
        elif not v:
            in_event  = False
    odi = float(n_events * 120.0)

    apen = _approx_entropy(seg)

    return {
        "spo2_mean":           round(spo2_mean,        3),
        "spo2_min":            round(spo2_min,         3),
        "spo2_delta_index":    round(spo2_delta_index, 6),
        "odi":                 round(odi,              2),
        "t90":                 round(t90,              6),
        "spo2_approx_entropy": round(apen,             6),
        "has_spo2":            1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CSV BUILDER  (unchanged from mongo_infer.py — operates on numpy signals)
# ══════════════════════════════════════════════════════════════════════════════

def build_segment_csv(
    ecg_signal:         np.ndarray,
    spo2_signal:        np.ndarray,
    admission_id:       str,
    facility_id:        str,
    packet_timestamps:  List,
    output_path:        str,
    ecg_chunk_lengths:  Optional[List[int]] = None,
    spo2_timestamps:    Optional[List]      = None,
    spo2_chunk_lengths: Optional[List[int]] = None,
) -> int:
    n_segs = len(ecg_signal) // SAMPLES_SEG
    if n_segs < MIN_SEGMENTS:
        logger.warning(
            "[CSV] Only %d complete segments (need ≥ %d, %.1f min — need ≥ %.0f min).",
            n_segs, MIN_SEGMENTS,
            len(ecg_signal) / FS_ECG / 60,
            MIN_SEGMENTS * SEGMENT_LEN_S / 60)
        if n_segs == 0:
            return 0

    has_spo2_global = len(spo2_signal) >= SAMPLES_SPO2_SEG
    global_spo2_baseline: Optional[float] = None
    if has_spo2_global:
        global_spo2_baseline = _compute_global_spo2_baseline(spo2_signal)
        logger.info("[CSV] SpO2 available (%d samples). "
                    "Patient baseline (90th pct): %.1f%%",
                    len(spo2_signal), global_spo2_baseline)
    else:
        logger.warning("[CSV] No SpO2 — has_spo2=0 for all segments")

    ecg_chunk_lengths = ecg_chunk_lengths or [SAMPLES_PER_ECG_DOC] * len(packet_timestamps)
    ecg_time_idx  = _build_time_index(packet_timestamps, ecg_chunk_lengths)
    spo2_time_idx = (
        _build_time_index(spo2_timestamps, spo2_chunk_lengths)
        if (has_spo2_global and spo2_timestamps and spo2_chunk_lengths)
        else None
    )
    ts_align = has_spo2_global and ecg_time_idx is not None and spo2_time_idx is not None
    if has_spo2_global and not ts_align:
        logger.warning(
            "[CSV] Insufficient timestamp coverage — falling back to "
            "index-based SpO2 alignment (assumes both streams started together).")

    logger.info("[CSV] Building %d × 30s segments (SpO2 align: %s) ...",
                n_segs, "timestamp" if ts_align else "index")

    rows        = []
    n_spo2_segs = 0
    n_unresolved_ts = 0

    for seg_i in range(n_segs):
        ecg_start = seg_i * SAMPLES_SEG
        seg       = ecg_signal[ecg_start: ecg_start + SAMPLES_SEG]

        try:
            seg_bp = _bandpass(seg, FS_ECG)
        except Exception:
            seg_bp = seg.copy()

        sub_hrs = _subseg_hrs(seg_bp, FS_ECG)
        mean_hr = (float(np.mean([h for h in sub_hrs if h > 0]))
                   if any(h > 0 for h in sub_hrs) else 0.0)
        sig_quality = _assess_signal_quality(seg, seg_bp, FS_ECG, sub_hrs)

        seg_start_ts: Optional[datetime] = None
        if ecg_time_idx is not None:
            starts_e, cum_e = ecg_time_idx
            seg_start_ts = _timestamp_for_sample(
                ecg_start, starts_e, cum_e, ecg_chunk_lengths, FS_ECG)

        if seg_start_ts is not None:
            ts_str = str(seg_start_ts)
        else:
            ts_str = ""
            n_unresolved_ts += 1

        if ts_align and seg_start_ts is not None:
            starts_s, cum_s = spo2_time_idx
            spo2_start = _sample_index_for_time(
                seg_start_ts, starts_s, cum_s, spo2_chunk_lengths, FS_SPO2)
        else:
            spo2_start = seg_i * SAMPLES_SPO2_SEG if has_spo2_global else None

        if spo2_start is not None and has_spo2_global:
            spo2_end = spo2_start + SAMPLES_SPO2_SEG
            if spo2_end <= len(spo2_signal):
                spo2_feats = _compute_spo2_features(
                    spo2_signal[spo2_start:spo2_end],
                    baseline_override=global_spo2_baseline)
            elif spo2_start < len(spo2_signal):
                spo2_feats = _compute_spo2_features(
                    spo2_signal[spo2_start:],
                    baseline_override=global_spo2_baseline)
            else:
                spo2_feats = dict(_SPO2_DEFAULTS)
        else:
            spo2_feats = dict(_SPO2_DEFAULTS)

        if spo2_feats["has_spo2"] == 1:
            n_spo2_segs += 1

        row: Dict = {
            "admissionId":                     admission_id,
            "facilityId":                      facility_id,
            "timestamp":                       ts_str,
            "segment_idx":                     seg_i,
            "analysis.summary.signal_quality": sig_quality,
            "analysis.heart_rate_bpm":         round(mean_hr, 1),
            "analysis.background_rhythm":      "",
        }
        for i, hr in enumerate(sub_hrs):
            row[f"analysis.segments[{i}].morphology.hr_bpm"] = round(hr, 2)
            row[f"analysis.segments[{i}].rhythm_label"]      = ""
            row[f"analysis.segments[{i}].ectopy_label"]      = ""
        row.update(spo2_feats)
        row.update(dict(zip(ECG_COLS, np.round(seg.astype(float), 6).tolist())))
        rows.append(row)

    if n_unresolved_ts:
        logger.warning(
            "[CSV] %d / %d segments had no resolvable timestamp anywhere in "
            "the admission's row metadata — written with blank 'timestamp'.",
            n_unresolved_ts, n_segs)

    out_df = pd.DataFrame(rows)
    fixed_cols = (
        ["admissionId", "facilityId", "timestamp", "segment_idx",
         "analysis.summary.signal_quality",
         "analysis.heart_rate_bpm",
         "analysis.background_rhythm"]
        + HR_COLS + RHYTHM_COLS + ECTOPY_COLS
        + SPO2_FEATURE_COLS + ["has_spo2"]
        + ECG_COLS
    )
    for c in fixed_cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[fixed_cols]

    out_dir_abs = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir_abs, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("[CSV] Wrote %d segments → %s  (SpO2 in %d / %d segments)",
                len(out_df), output_path, n_spo2_segs, n_segs)
    return len(out_df)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE RUNNER  (unchanged from mongo_infer.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_on_csv(
    csv_path:     str,
    model_path:   str,
    scaler_path:  str,
    threshold:    float,
    out_dir:      str,
    admission_id: str,
) -> Optional[Dict]:
    project_root = str(Path(__file__).resolve().parent.parent)
    pipeline_dir = str(Path(__file__).resolve().parent.parent / "pipeline")
    for p in (project_root, pipeline_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from pipeline.infer import run_inference
    except ImportError:
        try:
            from infer import run_inference
        except ImportError as e:
            logger.error("[INFER] Cannot import infer.py: %s", e)
            return None

    logger.info("[INFER] Running inference for %s ...", admission_id)
    try:
        run_inference(
            csv_path=csv_path, model_path=model_path, scaler_path=scaler_path,
            threshold=threshold, out_dir=out_dir, admission_id=admission_id,
        )
    except Exception as e:
        logger.error("[INFER] Failed for %s: %s", admission_id, e)
        return None

    summary_csv = os.path.join(out_dir, "infer_summary.csv")
    if not os.path.exists(summary_csv):
        logger.warning("[INFER] No summary CSV for %s", admission_id)
        return None

    df = pd.read_csv(summary_csv)
    matches = df[df["admission_id"] == admission_id]
    if matches.empty:
        logger.warning("[INFER] No row for %s in summary", admission_id)
        return None
    return matches.iloc[0].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS ONE ADMISSION
# ══════════════════════════════════════════════════════════════════════════════

def process_admission(
    client,
    admission_id:      str,
    facility_id:       str,
    model_path:        str,
    scaler_path:       str,
    threshold:         float,
    output_dir:        str,
    dry_run:           bool = False,
    no_sleep_filter:   bool = False,
    bilstm_model_path: Optional[str] = None,
    bilstm_scaler_path: Optional[str] = None,
    threshold_bilstm:  Optional[float] = None,
) -> Optional[Dict]:
    logger.info("=" * 60)
    logger.info("  Processing: %s", admission_id)
    logger.info("=" * 60)

    adm_out_dir = os.path.join(output_dir, admission_id)
    os.makedirs(adm_out_dir, exist_ok=True)

    # ── ECG ──────────────────────────────────────────────────────────────────
    ecg_signal, packet_timestamps, ecg_chunk_lengths = extract_ecg_signal(client, admission_id)
    if len(ecg_signal) == 0:
        return None

    nan_mask = ~np.isfinite(ecg_signal)
    if nan_mask.any():
        ecg_signal[nan_mask] = float(np.nanmean(ecg_signal))
        logger.warning("[PROCESS] Filled %d NaN samples in ECG", int(nan_mask.sum()))

    # ── SpO2 ─────────────────────────────────────────────────────────────────
    spo2_signal, spo2_timestamps, spo2_chunk_lengths = extract_spo2_signal(client, admission_id)

    # ── Sleep filtering ───────────────────────────────────────────────────────
    if HAS_SLEEP_FILTER and not no_sleep_filter:
        sleep_df = detect_sleep_segments(
            ecg_signal        = ecg_signal,
            packet_timestamps = packet_timestamps,
            ecg_chunk_lengths = ecg_chunk_lengths,
        )
        n_sleep = int(sleep_df["is_sleep"].sum()) if len(sleep_df) else 0
        if n_sleep >= MIN_SEGMENTS:
            sleep_idxs = sleep_df.loc[sleep_df["is_sleep"] == 1, "segment_idx"].tolist()
            ecg_signal = np.concatenate([ecg_signal[i * SAMPLES_SEG:(i + 1) * SAMPLES_SEG]
                                          for i in sleep_idxs])
            ts_lookup = sleep_df.set_index("segment_idx")["timestamp_utc"]
            packet_timestamps = [ts_lookup.get(i) for i in sleep_idxs]
            ecg_chunk_lengths = [SAMPLES_SEG] * len(sleep_idxs)
            sleep_csv = os.path.join(adm_out_dir, f"{admission_id}_sleep_windows.csv")
            sleep_df.to_csv(sleep_csv, index=False)
            logger.info("[SLEEP] Kept %d / %d segments after sleep filter → %s",
                        len(sleep_idxs), len(sleep_df), sleep_csv)
        else:
            logger.warning(
                "[SLEEP] Only %d sleep segments detected (need ≥ %d) — "
                "running inference on full recording instead",
                n_sleep, MIN_SEGMENTS)
    elif not HAS_SLEEP_FILTER:
        logger.info("[SLEEP] Sleep filter unavailable — using full recording")
    else:
        logger.info("[SLEEP] Sleep filter disabled via --no-sleep-filter")

    # ── Build CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(adm_out_dir, f"{admission_id}_segments.csv")
    n_segs = build_segment_csv(
        ecg_signal         = ecg_signal,
        spo2_signal        = spo2_signal,
        admission_id       = admission_id,
        facility_id        = facility_id,
        packet_timestamps  = packet_timestamps,
        output_path        = csv_path,
        ecg_chunk_lengths  = ecg_chunk_lengths,
        spo2_timestamps    = spo2_timestamps,
        spo2_chunk_lengths = spo2_chunk_lengths,
    )
    if n_segs == 0:
        return None

    if dry_run:
        logger.info("[DRY RUN] CSV at %s — skipping inference", csv_path)
        return {"admission_id": admission_id, "status": "dry_run", "n_segments": n_segs}

    # ── Model 1 inference (XGBoost) ───────────────────────────────────────────
    summary_xgb = run_inference_on_csv(
        csv_path=csv_path, model_path=model_path, scaler_path=scaler_path,
        threshold=threshold, out_dir=adm_out_dir, admission_id=admission_id,
    )
    if not summary_xgb:
        logger.error("[INFER] XGBoost model failed for %s", admission_id)

    # ── Model 2 inference (BiLSTM) ────────────────────────────────────────────
    summary_bilstm = None
    run_bilstm = (
        bilstm_model_path and bilstm_scaler_path
        and os.path.exists(bilstm_model_path)
        and os.path.exists(bilstm_scaler_path)
    )
    if run_bilstm:
        logger.info("[INFER] Running BiLSTM ...")
        bilstm_out_dir = os.path.join(adm_out_dir, "bilstm")
        os.makedirs(bilstm_out_dir, exist_ok=True)
        bilstm_thresh = threshold_bilstm if threshold_bilstm is not None else threshold
        summary_bilstm = run_inference_on_csv(
            csv_path=csv_path, model_path=bilstm_model_path,
            scaler_path=bilstm_scaler_path, threshold=bilstm_thresh,
            out_dir=bilstm_out_dir, admission_id=admission_id,
        )
        if summary_bilstm:
            logger.info("[INFER] BiLSTM  AHI=%.1f  Severity=%s",
                        summary_bilstm.get("ahi_proxy", 0),
                        summary_bilstm.get("severity", "?"))
        else:
            logger.warning("[INFER] BiLSTM inference failed")
    else:
        logger.info("[INFER] BiLSTM paths not provided — skipping")

    if not summary_xgb and not summary_bilstm:
        logger.error("[INFER] Both models failed for %s", admission_id)
        return None

    summary = summary_xgb.copy() if summary_xgb else summary_bilstm.copy()

    if summary_xgb:
        summary["ahi_proxy"] = summary_xgb.get("ahi_proxy")
        summary["severity"]  = summary_xgb.get("severity")
        summary["apnea_pct"] = summary_xgb.get("apnea_pct")
    else:
        summary["ahi_proxy"] = None
        summary["severity"]  = None
        summary["apnea_pct"] = None

    if summary_bilstm:
        summary["ahi_bilstm"]      = summary_bilstm.get("ahi_proxy")
        summary["severity_bilstm"] = summary_bilstm.get("severity")
    else:
        summary["ahi_bilstm"]      = None
        summary["severity_bilstm"] = None

    # ── Physiological validation ───────────────────────────────────────────────
    try:
        from apnea_validator import validate_admission
        infer_csv = os.path.join(adm_out_dir, f"infer_results_{admission_id}.csv")
        if os.path.exists(infer_csv):
            logger.info("[VALIDATE] Running physiological validation ...")
            val_result = validate_admission(
                infer_csv    = infer_csv,
                admission_id = admission_id,
                verbose      = False,
            )
            summary["validated_ahi"]             = val_result.validated_ahi
            summary["validation_confirmed"]      = val_result.confirmed
            summary["validation_probable"]       = val_result.probable
            summary["validation_uncertain"]      = val_result.uncertain
            summary["validation_unconfirmed"]    = val_result.unconfirmed
            summary["validation_mean_score"]     = val_result.mean_score
            summary["physiologically_supported"] = val_result.validated_ahi >= 5.0
        else:
            logger.warning("[VALIDATE] infer_results CSV not found — skipping validation")
    except Exception as e:
        logger.warning("[VALIDATE] Skipped due to error: %s", e)

    logger.info("[RESULT] %s  XGB_AHI=%s  BiLSTM_AHI=%s  ValidatedAHI=%s",
                admission_id,
                f"{summary.get('ahi_proxy'):.1f}" if summary.get("ahi_proxy") is not None else "N/A",
                f"{summary.get('ahi_bilstm'):.1f}" if summary.get("ahi_bilstm") is not None else "N/A",
                f"{summary.get('validated_ahi'):.1f}" if summary.get("validated_ahi") is not None else "N/A",
                )

    # ── Per-segment apnea event timestamps (for apnea_events table) ───────────
    # Extracted here, attached under a private key so callers can push them to
    # Supabase without re-deriving file paths.
    events: List[Dict] = []
    if summary_xgb:
        xgb_csv = os.path.join(adm_out_dir, f"infer_results_{admission_id}.csv")
        events.extend(extract_apnea_events(xgb_csv, admission_id, "xgboost"))
    if summary_bilstm:
        bilstm_csv = os.path.join(adm_out_dir, "bilstm", f"infer_results_{admission_id}.csv")
        events.extend(extract_apnea_events(bilstm_csv, admission_id, "bilstm"))
    summary["_apnea_events"] = events

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE WRITE-BACK  (unchanged from mongo_infer.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_existing_result(supabase_client, admission_id: str) -> Optional[Dict]:
    try:
        resp = (
            supabase_client.table(RESULTS_TABLE)
            .select("admission_id,processed_at")
            .eq("admission_id", admission_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning("[SUPABASE] Idempotency check failed for %s: %s — proceeding",
                       admission_id, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  PER-EVENT APNEA TIMESTAMPS  (apnea_events table)
# ══════════════════════════════════════════════════════════════════════════════
#
# infer.py writes infer_results_<admission_id>.csv with one row per 30s
# segment, including 'timestamp', 'apnea_pred', and 'apnea_prob'. The
# aggregate write-back (write_results_to_supabase) only persists summary
# stats (AHI, severity, counts) — it never persisted *when* each apnea
# event actually occurred. These functions extract the flagged rows and
# push them into a dedicated apnea_events table.
#
# Required schema (run once):
#
#   CREATE TABLE apnea_events (
#       id BIGSERIAL PRIMARY KEY,
#       admission_id TEXT NOT NULL REFERENCES apnea_results(admission_id),
#       model_source TEXT NOT NULL,        -- 'xgboost' or 'bilstm'
#       segment_idx INTEGER NOT NULL,
#       event_timestamp TIMESTAMPTZ,
#       apnea_prob REAL,
#       signal_quality TEXT,
#       UNIQUE(admission_id, model_source, segment_idx)
#   );
#   CREATE INDEX idx_apnea_events_lookup
#       ON apnea_events(admission_id, event_timestamp);

EVENTS_TABLE = os.environ.get("EVENTS_TABLE", "apnea_events")
EVENTS_BATCH_SIZE = 200


def extract_apnea_events(
    infer_results_csv: str, admission_id: str, model_source: str,
) -> List[Dict]:
    """
    Read an infer_results_<admission_id>.csv (written by infer.py) and
    return one record per segment where apnea was predicted (apnea_pred == 1.0),
    ready for upsert into apnea_events.

    Returns [] if the CSV is missing or has no apnea predictions — this is
    a normal, non-error outcome (e.g. "No apnea detected above threshold").
    """
    if not os.path.exists(infer_results_csv):
        logger.warning("[EVENTS] %s not found — no events to extract for %s/%s",
                       infer_results_csv, admission_id, model_source)
        return []

    df = pd.read_csv(infer_results_csv)
    if "apnea_pred" not in df.columns:
        logger.warning("[EVENTS] %s has no apnea_pred column", infer_results_csv)
        return []

    flagged = df[df["apnea_pred"] == 1.0]
    events = []
    for _, row in flagged.iterrows():
        ts_raw = row.get("timestamp", "")
        event_ts = None
        if pd.notna(ts_raw) and str(ts_raw).strip():
            parsed = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if pd.notna(parsed):
                event_ts = parsed.isoformat()
        events.append({
            "admission_id":    admission_id,
            "model_source":    model_source,
            "segment_idx":     int(row.get("segment_idx", -1)),
            "event_timestamp": event_ts,
            "apnea_prob":      float(row.get("apnea_prob")) if pd.notna(row.get("apnea_prob")) else None,
            "signal_quality":  str(row.get("signal_quality", "")),
        })

    logger.info("[EVENTS] %s/%s: %d apnea events extracted from %s",
                admission_id, model_source, len(events), infer_results_csv)
    return events


def write_apnea_events_to_supabase(
    supabase_client, events: List[Dict],
) -> None:
    """Batch-upsert per-segment apnea events, on_conflict (admission_id, model_source, segment_idx)."""
    if not events:
        return
    total = len(events)
    uploaded = 0
    for i in range(0, total, EVENTS_BATCH_SIZE):
        batch = events[i:i + EVENTS_BATCH_SIZE]
        try:
            (
                supabase_client.table(EVENTS_TABLE)
                .upsert(batch, on_conflict="admission_id,model_source,segment_idx")
                .execute()
            )
            uploaded += len(batch)
        except Exception as e:
            logger.error("[EVENTS] Batch upsert failed (%d-%d / %d): %s",
                        i, i + len(batch), total, e)
            # Retry row-by-row so one bad row doesn't drop the whole batch
            for ev in batch:
                try:
                    (
                        supabase_client.table(EVENTS_TABLE)
                        .upsert(ev, on_conflict="admission_id,model_source,segment_idx")
                        .execute()
                    )
                    uploaded += 1
                except Exception as err:
                    logger.error("[EVENTS] Skipped event admission=%s segment=%s: %s",
                                ev.get("admission_id"), ev.get("segment_idx"), err)
    logger.info("[EVENTS] Upserted %d / %d events → %s", uploaded, total, EVENTS_TABLE)


def write_results_to_supabase(
    supabase_client,
    admission_id: str,
    facility_id:  str,
    summary:      Dict,
) -> None:
    ahi = summary.get("ahi_proxy")
    apnea_label = None
    has_apnea = None
    if ahi is not None:
        apnea_label = ("No Apnea" if ahi < 5 else
                       "Mild"     if ahi < 15 else
                       "Moderate" if ahi < 30 else "Severe")
        has_apnea = ahi >= 5.0

    record = {
        "admission_id":    admission_id,
        "facility_id":     facility_id,
        "processed_at":    datetime.now(timezone.utc).isoformat(),
        "ahi_proxy":       ahi,
        "severity":        summary.get("severity"),
        "apnea_label":     apnea_label,
        "has_apnea":       has_apnea,
        "apnea_pct":       summary.get("apnea_pct"),
        "total_segments":  summary.get("total_segments"),
        "scored_segments": summary.get("scored_segments"),
        "n_apnea":         summary.get("n_apnea"),
        "duration_min":    summary.get("duration_min"),
        "model_threshold": summary.get("threshold"),
        "ahi_bilstm":      summary.get("ahi_bilstm"),
        "severity_bilstm": summary.get("severity_bilstm"),
        "validated_ahi":             summary.get("validated_ahi"),
        "validation_confirmed":      summary.get("validation_confirmed"),
        "validation_probable":       summary.get("validation_probable"),
        "validation_uncertain":      summary.get("validation_uncertain"),
        "validation_unconfirmed":    summary.get("validation_unconfirmed"),
        "validation_mean_score":     summary.get("validation_mean_score"),
        "physiologically_supported": summary.get("physiologically_supported"),
    }
    try:
        (supabase_client.table(RESULTS_TABLE)
         .upsert(record, on_conflict="admission_id")
         .execute())
        logger.info("[SUPABASE] Upserted %s (XGB_AHI=%s, BiLSTM_AHI=%s)",
                    admission_id,
                    f"{ahi:.1f}" if ahi is not None else "N/A",
                    f"{summary.get('ahi_bilstm'):.1f}" if summary.get("ahi_bilstm") is not None else "N/A")
    except Exception as e:
        logger.error("[SUPABASE] Failed for %s: %s", admission_id, e)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supabase → ECG+SpO2 → Apnea inference → Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-bilstm",  default=os.environ.get("BILSTM_MODEL_PATH",  "apnea_model.keras"),
               help="Path to BiLSTM .keras for consensus (optional)")
    p.add_argument("--scaler-bilstm", default=os.environ.get("BILSTM_SCALER_PATH", "apnea_scaler.pkl"),
               help="Path to BiLSTM scaler for consensus (optional)")
    p.add_argument("--admission",       default=None)
    p.add_argument("--since",           default=None,
                   help="Last N hours, e.g. 24h")
    p.add_argument("--from",            dest="date_from", default=None,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--to",              dest="date_to",   default=None,
                   help="End date YYYY-MM-DD")
    p.add_argument("--model",           default=os.environ.get("MODEL_PATH",  "apnea_model_xgb_seq.pkl"),
               help="Path to the XGBoost (seq) model — must be the .pkl from pipeline.py, not the BiLSTM .keras")
    p.add_argument("--scaler",          default=os.environ.get("SCALER_PATH", "apnea_scaler_tree.pkl"),
               help="Path to the XGBoost tree scaler — apnea_scaler_tree.pkl, not apnea_scaler.pkl")
    p.add_argument("--threshold-bilstm", type=float, default=None,
               help="Optional separate threshold for the BiLSTM model. Defaults to --threshold.")
    p.add_argument("--threshold",       type=float,
                   default=float(os.environ.get("THRESHOLD", "0.45")))
    p.add_argument("--out-dir",         default=os.environ.get("OUTPUT_DIR",  "infer_output"))
    p.add_argument("--dry-run",         action="store_true",
                   help="Extract CSVs but skip inference")
    p.add_argument("--write-supabase",  action="store_true",
                   help="Write results to Supabase apnea_results table")
    p.add_argument("--reprocess",       action="store_true",
                   help="Re-run even if a Supabase result already exists")
    p.add_argument("--no-sleep-filter", action="store_true",
                   help="Skip ECG sleep detection and run inference on the full recording")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.dry_run:
        for path, label in [(args.model, "model"), (args.scaler, "scaler")]:
            if not os.path.exists(path):
                logger.error("[SETUP] %s not found: %s", label, path)
                logger.error("[SETUP] Train first: python pipeline/pipeline.py --fresh --save-model")
                sys.exit(1)

    import json as _json
    _threshold_json = Path(args.model).parent / "apnea_thresholds.json"
    env_threshold_set = bool(os.environ.get("THRESHOLD"))
    cli_threshold_set = "--threshold" in sys.argv
    if _threshold_json.exists() and not env_threshold_set and not cli_threshold_set:
        try:
            with open(_threshold_json) as _f:
                _saved = _json.load(_f)
            args.threshold = float(_saved.get("global", args.threshold))
            logger.info("[SETUP] Loaded cross-validated threshold from %s: %.3f",
                        _threshold_json.name, args.threshold)
        except Exception as _e:
            logger.warning("[SETUP] Could not load threshold from %s: %s — using %.3f",
                           _threshold_json.name, _e, args.threshold)
    else:
        logger.info("[SETUP] Using threshold: %.3f  (source: %s)",
                    args.threshold,
                    "THRESHOLD env" if env_threshold_set else
                    "--threshold CLI" if cli_threshold_set else "CLI default")

    client = get_supabase_client()

    # ── Guard against the exact bug that caused XGB/BiLSTM results to swap ────
    # Refuse to run if the "XGBoost" and "BiLSTM" slots point at the same file,
    # or if either points at a file with the wrong extension for its role.
    if os.path.abspath(args.model) == os.path.abspath(args.model_bilstm):
        logger.error(
            "[SETUP] --model and --model-bilstm resolve to the same file (%s). "
            "This previously caused the BiLSTM model to silently run twice "
            "while apnea_model_xgb_seq.pkl was never loaded. Refusing to run — "
            "set MODEL_PATH=apnea_model_xgb_seq.pkl and "
            "BILSTM_MODEL_PATH=apnea_model.keras explicitly.", args.model)
        sys.exit(1)
    if os.path.abspath(args.scaler) == os.path.abspath(args.scaler_bilstm):
        logger.error(
            "[SETUP] --scaler and --scaler-bilstm resolve to the same file (%s). "
            "Set SCALER_PATH=apnea_scaler_tree.pkl and "
            "BILSTM_SCALER_PATH=apnea_scaler.pkl explicitly.", args.scaler)
        sys.exit(1)
    if not args.model.endswith(".pkl"):
        logger.warning(
            "[SETUP] --model (XGBoost slot) is '%s', which doesn't end in .pkl. "
            "infer.py routes by extension — a non-.pkl path here will be loaded "
            "as a Keras model instead of XGBoost.", args.model)
    if not args.model_bilstm.endswith(".keras"):
        logger.warning(
            "[SETUP] --model-bilstm is '%s', which doesn't end in .keras. "
            "infer.py routes by extension — a .pkl path here will be loaded "
            "as an XGBoost/tree model instead of the BiLSTM.", args.model_bilstm)

    threshold_bilstm = (args.threshold_bilstm if args.threshold_bilstm is not None
                        else args.threshold)

    # ── Resolve admissions ────────────────────────────────────────────────────
    if args.admission:
        facility_id = _get_facility_id(client, args.admission)
        admissions = [{"admissionId": args.admission, "facilityId": facility_id}]
    else:
        since_hours = date_from = date_to = None
        if args.since:
            since_hours = float(args.since.replace("h", "").replace("H", ""))
        if args.date_from:
            date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
        if args.date_to:
            date_to = datetime.strptime(args.date_to, "%Y-%m-%d")
        if not any([since_hours, date_from, date_to]):
            logger.error("[SETUP] Specify --admission, --since, or --from/--to")
            sys.exit(1)
        admissions = find_completed_admissions(
            client, since_hours=since_hours,
            date_from=date_from, date_to=date_to)

    if not admissions:
        logger.info("[MAIN] No admissions to process.")
        return

    logger.info("[MAIN] %d admissions to process", len(admissions))
    successes = failures = 0

    for adm in admissions:
        admission_id = adm["admissionId"]
        facility_id  = adm.get("facilityId", "")

        # Idempotency skip
        if args.write_supabase and not args.dry_run and not args.reprocess:
            existing = get_existing_result(client, admission_id)
            if existing:
                logger.info("[MAIN] %s already processed at %s — skip "
                            "(--reprocess to override)",
                            admission_id, existing.get("processed_at"))
                continue

        try:
            summary = process_admission(
                client=client, admission_id=admission_id, facility_id=facility_id,
                model_path=args.model, scaler_path=args.scaler,
                threshold=args.threshold, output_dir=args.out_dir,
                dry_run=args.dry_run,
                no_sleep_filter=args.no_sleep_filter,
                bilstm_model_path=args.model_bilstm,
                bilstm_scaler_path=args.scaler_bilstm,
                threshold_bilstm=threshold_bilstm,
            )
            if summary:
                successes += 1
                events = summary.pop("_apnea_events", [])
                if args.write_supabase and not args.dry_run:
                    write_results_to_supabase(
                        client, admission_id, facility_id, summary)
                    write_apnea_events_to_supabase(client, events)
                else:
                    logger.info(
                        "[MAIN] %s processed but NOT written to Supabase "
                        "(--write-supabase not set%s) — results only on disk at %s",
                        admission_id,
                        " / --dry-run active" if args.dry_run else "",
                        os.path.abspath(args.out_dir))
            else:
                failures += 1
        except Exception as e:
            logger.error("[MAIN] Unhandled error for %s: %s",
                         admission_id, e, exc_info=True)
            failures += 1

    logger.info("=" * 60)
    logger.info("[DONE]  %d succeeded  |  %d failed  |  output → %s",
                successes, failures, os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()