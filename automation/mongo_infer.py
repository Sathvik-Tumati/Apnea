"""
mongo_infer.py
==============
Pulls ECG + SpO2 data for a given admissionId from MongoDB,
assembles 30-second segments, runs the apnea inference pipeline,
and writes results to Supabase.

Usage
-----
  python automation/mongo_infer.py --admission ADM1819906487
  python automation/mongo_infer.py --since 24h
  python automation/mongo_infer.py --from 2026-06-01 --to 2026-06-15
  python automation/mongo_infer.py --since 24h --dry-run
  python automation/mongo_infer.py --since 24h --write-supabase
  python automation/mongo_infer.py --since 24h --write-supabase --reprocess

Environment variables (set in .env at project root)
----------------------------------------------------
MONGO_URI      mongodb+srv://user:pass@cluster/
MONGO_DB       your_database_name
SUPABASE_URL   https://your-project.supabase.co
SUPABASE_KEY   your-service-role-or-anon-key
MODEL_PATH     /path/to/apnea_model.keras      (default: apnea_model.keras)
SCALER_PATH    /path/to/apnea_scaler.pkl       (default: apnea_scaler.pkl)
THRESHOLD      0.45                             (default: 0.45 — matches infer.py)
OUTPUT_DIR     /path/to/output/                (default: infer_output/)
"""

# ── .env loading — resolved path only, no bare CWD-dependent call ────────────
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
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample_poly, welch

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Sleep filter  ─────────────────────────────────────────────
try:
    from ecg_sleep_filter import detect_sleep_segments, filter_to_sleep
    HAS_SLEEP_FILTER = True
except ImportError:
    # Fallback when running from outside the automation/ directory
    try:
        from automation.ecg_sleep_filter import detect_sleep_segments, filter_to_sleep
        HAS_SLEEP_FILTER = True
    except ImportError:
        HAS_SLEEP_FILTER = False
        logger.warning("[SLEEP] ecg_sleep_filter.py not importable — sleep filtering disabled")

# ── MongoDB config ─────────────────────────────────────────────────────────────
MONGO_CONFIG = {
    "uri": os.environ.get("MONGO_URI"),
    "db":  os.environ.get("MONGO_DB"),
}

# ── SSH Tunnel config ──────────────────────────────────────────────────────────
JUMP_HOST  = "3.109.139.226"
JUMP_USER  = "ubuntu"
JUMP_PORT  = 22
PEM_FILE   = os.path.join(os.path.dirname(__file__), "ls-nexus-qa-app 1.pem")
MONGO_HOST = "10.17.131.14"
MONGO_PORT = 27017
LOCAL_PORT = 27018

# ── Supabase config ────────────────────────────────────────────────────────────
SUPABASE_CONFIG = {
    "url": os.environ.get("SUPABASE_URL"),
    "key": os.environ.get("SUPABASE_KEY"),
}

# ── Pipeline constants ────────────────────────────────────────────────────────
FS_ECG               = 125
FS_SPO2              = 1      # device-computed SpO2 from spo2_unfiltered_data is 1 Hz
SEGMENT_LEN_S        = 30
SAMPLES_SEG          = FS_ECG  * SEGMENT_LEN_S   # 3750 ECG samples per segment
SAMPLES_SPO2_SEG     = FS_SPO2 * SEGMENT_LEN_S   # 30 SpO2 samples per segment
SAMPLES_PER_ECG_DOC  = 125
MIN_SEGMENTS         = 11
MIN_DURATION_MINUTES = 30
COMPLETION_GAP_HOURS = 2

ECG_COLS    = [f"ecgData[{i}]"                              for i in range(SAMPLES_SEG)]
HR_COLS     = [f"analysis.segments[{i}].morphology.hr_bpm" for i in range(6)]
RHYTHM_COLS = [f"analysis.segments[{i}].rhythm_label"      for i in range(6)]
ECTOPY_COLS = [f"analysis.segments[{i}].ectopy_label"      for i in range(6)]

# These names must match APNEA_FEATURE_COLS in infer.py exactly.
SPO2_FEATURE_COLS = [
    "spo2_mean", "spo2_min", "spo2_delta_index",
    "odi", "t90", "spo2_approx_entropy",
]
SPO2_RAW_COLS = [f"spo2Data[{i}]" for i in range(SAMPLES_SPO2_SEG)]


# ══════════════════════════════════════════════════════════════════════════════
#  SSH TUNNEL
# ══════════════════════════════════════════════════════════════════════════════

def _wait_for_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


class _SSHTunnel:
    """Context manager that opens an SSH tunnel if one isn't already running."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "_SSHTunnel":
        if self._port_open():
            logger.info("[SSH] Using existing tunnel on localhost:%d", LOCAL_PORT)
            return self

        if not os.path.exists(PEM_FILE):
            raise RuntimeError(f"[SSH] PEM file not found: {PEM_FILE}")

        cmd = [
            "ssh", "-i", PEM_FILE,
            "-L", f"{LOCAL_PORT}:{MONGO_HOST}:{MONGO_PORT}",
            "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=10",
            "-o", "ConnectTimeout=10",
            "-o", "LogLevel=ERROR",
            f"{JUMP_USER}@{JUMP_HOST}",
            "-p", str(JUMP_PORT),
        ]
        logger.info("[SSH] Opening tunnel → %s:%d via %s ...",
                    MONGO_HOST, MONGO_PORT, JUMP_HOST)
        self.proc = subprocess.Popen(cmd,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE,
                                     text=True)
        time.sleep(5)
        if self.proc.poll() is not None:
            _, stderr = self.proc.communicate()
            raise RuntimeError(
                f"[SSH] Tunnel exited immediately (code {self.proc.returncode}): {stderr}")

        if not _wait_for_port(LOCAL_PORT, timeout=45):
            self.proc.terminate()
            raise RuntimeError("SSH Tunnel started but port not forwarded within 45 s")

        logger.info("[SSH] Tunnel open → localhost:%d", LOCAL_PORT)
        return self

    def __exit__(self, *_) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            logger.info("[SSH] Tunnel closed")

    @staticmethod
    def _port_open() -> bool:
        try:
            with socket.create_connection(("127.0.0.1", LOCAL_PORT), timeout=2):
                return True
        except OSError:
            return False


# ══════════════════════════════════════════════════════════════════════════════
#  CONNECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    """Return a pymongo database handle via the SSH tunnel."""
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.error("pymongo not installed — pip install pymongo")
        sys.exit(1)
    client = MongoClient("127.0.0.1", LOCAL_PORT, serverSelectionTimeoutMS=10000)
    try:
        client.admin.command("ping")
        logger.info("[MONGO] Connected to %s", MONGO_CONFIG["db"])
    except Exception as e:
        logger.error("[MONGO] Connection failed: %s", e)
        sys.exit(1)
    return client[MONGO_CONFIG["db"]]


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


# ══════════════════════════════════════════════════════════════════════════════
#  ADMISSION DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_completed_admissions(
    db,
    since_hours: Optional[float] = None,
    date_from:   Optional[datetime] = None,
    date_to:     Optional[datetime] = None,
) -> List[Dict]:
    """
    Return admissions that have ended (no new ECG data for COMPLETION_GAP_HOURS)
    and have enough data to run inference.
    """
    # Build time filter — handle the case where both --since and --from are given
    # by taking the more restrictive (later) lower bound.
    since_cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)
                    if since_hours else None)
    from_cutoff  = (date_from.replace(tzinfo=timezone.utc) if date_from else None)

    ts_filter: Dict = {}
    if since_cutoff and from_cutoff:
        chosen = max(since_cutoff, from_cutoff)
        logger.warning(
            "[DISCOVER] Both --since and --from given — using later bound: %s", chosen)
        ts_filter["$gte"] = chosen
    elif since_cutoff:
        ts_filter["$gte"] = since_cutoff
    elif from_cutoff:
        ts_filter["$gte"] = from_cutoff
    if date_to:
        ts_filter["$lte"] = date_to.replace(tzinfo=timezone.utc)

    pipeline = []
    if ts_filter:
        pipeline.append({"$match": {"utcTimestamp": ts_filter}})
    pipeline += [
        {"$group": {
            "_id":        "$admissionId",
            "facilityId": {"$first": "$facilityId"},
            "first_ts":   {"$min":   "$utcTimestamp"},
            "last_ts":    {"$max":   "$utcTimestamp"},
            "n_docs":     {"$sum":   1},
        }},
        {"$project": {
            "_id": 0,
            "admissionId":  "$_id",
            "facilityId":   1,
            "first_ts":     1,
            "last_ts":      1,
            "n_docs":       1,
            "duration_min": {"$divide": [
                {"$subtract": ["$last_ts", "$first_ts"]}, 60000]},
        }},
    ]

    results = list(db.ecg_data_by_admission_id.aggregate(pipeline))
    now       = datetime.now(timezone.utc)
    completed = []
    for r in results:
        last_ts = r["last_ts"]
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        hours_since_last = (now - last_ts).total_seconds() / 3600
        has_enough_data  = r["duration_min"] >= MIN_DURATION_MINUTES
        recording_ended  = hours_since_last >= COMPLETION_GAP_HOURS
        if recording_ended and has_enough_data:
            r["hours_since_last"] = round(hours_since_last, 1)
            completed.append(r)
            logger.info("[DISCOVER] %s  %.0f min  %d docs  %.1fh ago  ✓ ELIGIBLE",
                        r["admissionId"], r["duration_min"],
                        r["n_docs"], hours_since_last)
        else:
            reason = []
            if not recording_ended:
                reason.append(f"still active ({hours_since_last:.1f}h since last)")
            if not has_enough_data:
                reason.append(f"too short ({r['duration_min']:.0f} min)")
            logger.info("[DISCOVER] %s  SKIP — %s",
                        r["admissionId"], " | ".join(reason))

    logger.info("[DISCOVER] %d / %d eligible", len(completed), len(results))
    return completed


# ══════════════════════════════════════════════════════════════════════════════
#  ECG EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_ecg_signal(
    db, admission_id: str
) -> Tuple[np.ndarray, List[datetime], List[int]]:
    """
    Pull all ECG documents for this admission ordered by packetNo.
    Returns (signal, doc_timestamps, doc_chunk_lengths).

    doc_timestamps[i] and doc_chunk_lengths[i] describe the i-th retained
    document chunk so callers can map a sample index back to wall-clock time.
    """
    logger.info("[ECG] Fetching for %s ...", admission_id)
    docs = list(
        db.ecg_data_by_admission_id
        .find({"admissionId": admission_id},
              {"packetNo": 1, "utcTimestamp": 1, "value": 1, "_id": 0})
        .sort("packetNo", 1)
    )
    if not docs:
        logger.error("[ECG] No documents for %s", admission_id)
        return np.array([]), [], []

    logger.info("[ECG] %d documents retrieved", len(docs))
    chunks, timestamps, chunk_lengths, skipped = [], [], [], 0

    for doc in docs:
        raw = doc.get("value")
        if isinstance(raw, list) and len(raw) > 0:
            chunk = raw[0] if isinstance(raw[0], list) else raw
            if len(chunk) > 0:
                arr = np.array(chunk, dtype=float)
                chunks.append(arr)
                timestamps.append(doc.get("utcTimestamp"))
                chunk_lengths.append(len(arr))
                continue
        skipped += 1

    if skipped:
        logger.warning("[ECG] Skipped %d docs with empty/missing value", skipped)
    if not chunks:
        return np.array([]), [], []

    # Validate assumed sample count per doc against reality
    lengths_arr = np.array(chunk_lengths)
    modal_len   = int(np.bincount(lengths_arr).argmax())
    if modal_len != SAMPLES_PER_ECG_DOC:
        logger.error(
            "[ECG] SAMPLE-RATE MISMATCH for %s: docs have %d samples, expected %d. "
            "Update SAMPLES_PER_ECG_DOC / FS_ECG before re-running.",
            admission_id, modal_len, SAMPLES_PER_ECG_DOC)
        return np.array([]), [], []

    n_odd = int(np.sum(lengths_arr != modal_len))
    if n_odd:
        logger.warning("[ECG] %d / %d docs have non-standard length (edge packets)",
                       n_odd, len(lengths_arr))

    signal = np.concatenate(chunks)
    logger.info("[ECG] Assembled %d samples (%.1f min at %d Hz)",
                len(signal), len(signal) / FS_ECG / 60, FS_ECG)
    return signal, timestamps, chunk_lengths


# ══════════════════════════════════════════════════════════════════════════════
#  SPO2 EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_spo2_signal(
    db, admission_id: str
) -> Tuple[np.ndarray, List[datetime], List[int]]:
    """
    Pull device-computed SpO2 readings from spo2_unfiltered_data (1 Hz).
    Each document holds one SpO2% value.

    Returns (spo2_signal, doc_timestamps, doc_chunk_lengths).
    doc_chunk_lengths is always a list of 1s (one sample per doc).

    Returns (empty, [], []) if no data is found.
    """
    docs = list(
        db.spo2_unfiltered_data
        .find({"admissionId": admission_id},
              {"utcTimestamp": 1, "spo2.spo2": 1, "_id": 0})
        .sort("utcTimestamp", 1)
    )
    if not docs:
        logger.warning("[SPO2] No spo2_unfiltered_data for %s", admission_id)
        return np.array([]), [], []

    values, timestamps, chunk_lengths = [], [], []
    for doc in docs:
        val = doc.get("spo2", {}).get("spo2")
        if val is not None:
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if 30.0 <= fval <= 105.0:      # physiological guard
                values.append(fval)
                timestamps.append(doc.get("utcTimestamp"))
                chunk_lengths.append(1)    # one sample per document

    if not values:
        logger.warning("[SPO2] All docs had out-of-range SpO2 for %s", admission_id)
        return np.array([]), [], []

    signal = np.array(values, dtype=float)
    logger.info("[SPO2] %d readings (1 Hz, %.1f min)  range %.0f–%.0f%%",
                len(signal), len(signal) / FS_SPO2 / 60,
                signal.min(), signal.max())
    return signal, timestamps, chunk_lengths


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL HELPERS
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
    # Guard: filter zero-length intervals (duplicate peak indices from neurokit2
    # on artifact-heavy ECG). Without this, mean RR ≈ 0 → HR = 60B BPM.
    rr_samples = rr_samples[rr_samples > 0]
    if len(rr_samples) == 0:
        return 0.0
    rr_ms = rr_samples / fs * 1000.0
    # Physiological filter: 300–2000 ms → 30–200 BPM
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    if len(rr_ms) == 0:
        return 0.0
    return float(60000.0 / np.mean(rr_ms))


def _subseg_hrs(ecg_seg: np.ndarray, fs: int, n: int = 6) -> List[float]:
    """Compute HR for each of n equal sub-segments of a 30s window."""
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
    """
    Return 'acceptable' or 'poor' for a 30-second ECG segment.
    Catches: flatlines, saturation/clipping, local flatline runs, no heartbeat.
    Errs on the side of accepting — a false 'poor' wastes a scorable segment.
    """
    if raw_seg is None or len(raw_seg) == 0:
        return "poor"
    finite = raw_seg[np.isfinite(raw_seg)]
    if len(finite) < 0.9 * len(raw_seg):
        return "poor"

    sd = float(np.std(finite))
    if sd < 1e-6:
        return "poor"           # true flatline

    lo, hi = float(finite.min()), float(finite.max())
    if hi > lo:
        if np.mean((finite <= lo + 1e-9) | (finite >= hi - 1e-9)) > 0.10:
            return "poor"       # signal pinned at rail > 10% of the time

    window = max(1, int(fs * 2))
    if len(bp_seg) >= window:
        roll_std = pd.Series(bp_seg).rolling(window).std().dropna().values
        if len(roll_std) > 0 and float(roll_std.min()) < 1e-4 * (sd + 1e-9):
            return "poor"       # local flatline run ≥ 2 s

    valid_hrs = [h for h in sub_hrs if h > 0]
    # Need at least half of sub-segments to have a plausible heartbeat
    if len(valid_hrs) < len(sub_hrs) // 2:
        return "poor"
    if any(h < 20 or h > 220 for h in valid_hrs):
        return "poor"

    return "acceptable"


# ══════════════════════════════════════════════════════════════════════════════
#  TIME ALIGNMENT HELPERS
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
    """
    Build (starts, cumulative_sample_starts) from per-document metadata.
    Returns None when fewer than 2 documents have usable timestamps.
    """
    starts, cum, running, n_valid = [], [], 0, 0
    for ts, length in zip(doc_timestamps, doc_lengths):
        norm = _normalize_ts(ts)
        starts.append(norm)
        cum.append(running)
        running += length
        if norm is not None:
            n_valid += 1
    # Even 1 valid timestamp is useful as an anchor for outward-search fallback
    # in _timestamp_for_sample(). Old threshold of 2 wasted single-timestamp admissions.
    return (starts, cum) if n_valid >= 1 else None


def _sample_index_for_time(
    target_ts:    datetime,
    starts:       List[Optional[datetime]],
    cum:          List[int],
    doc_lengths:  List[int],
    fs:           int,
) -> Optional[int]:
    """
    Map a wall-clock timestamp to an absolute sample index in the concatenated
    signal by finding the document whose window contains target_ts.
    Returns None if no document is at or before target_ts.
    """
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
    # Clamp to the actual span of this document (not doc_lengths which for
    # 1-Hz SpO2 is always 1, making the clamp effectively always 0 or 1).
    # Use the number of samples between this doc's cumulative start and the
    # next doc's start (or end-of-recording) as the real upper bound.
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
    """
    Resolve a wall-clock timestamp for an absolute sample index.

    Primary: use the document whose window contains sample_idx and offset
    within it. Fallback: if that document has no timestamp (gap in
    metadata), search outward — nearest earlier doc first, then nearest
    later doc — for any document with a valid timestamp and offset by the
    sample-count difference.

    Returns None only if *no* document anywhere has a valid timestamp
    (i.e. starts is all-None) — never returns a value derived from a
    missing timestamp, which previously surfaced downstream as a blank
    CSV cell that some consumers parse as epoch-zero, producing wildly
    wrong "recording duration" values.
    """
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
#  SPO2 FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _approx_entropy(sig: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Approximate entropy (ApEn) of a 1D signal.
    Standard parameters: m=2, r = r_factor × std(sig).
    Returns 0.0 for signals that are too short or have zero variance.

    Note: at FS_SPO2=1 Hz a 30-sample segment is passed directly —
    no downsampling is needed or applied (the O(n²) cost is trivial at n=30).
    """
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
    """
    Patient-level SpO2 baseline = 90th percentile of the full recording.

    Using the per-segment mean as the baseline underestimates ODI for
    chronically desaturated patients. The 90th percentile is robust to
    desaturation events pulling the mean down. Needs ≥ 60 samples (1 min).
    """
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
    """
    Compute the 6 SpO2 features for a 30-second window of 1 Hz SpO2 data
    (30 samples).

    Features
    --------
    spo2_mean          mean SpO2 in the segment
    spo2_min           minimum SpO2
    spo2_delta_index   mean |sample-to-sample change| — variability proxy
    odi                oxygen desaturation events ≥ 3% below the patient-level
                       baseline, scaled to events/hour (×120 for a 30s window)
    t90                fraction of segment where SpO2 < 90%
    spo2_approx_entropy ApEn regularity measure

    Returns neutral defaults + has_spo2=0 for empty/invalid input.
    """
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

    # ODI: count contiguous dip events ≥ 3% below patient-level baseline.
    # Each segment is 30 s → scale to /hr by ×120.
    # At 1 Hz resolution ODI per segment can only be 0, 120, 240 ... /hr.
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
#  CSV BUILDER  (format compatible with infer.py)
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
    """
    Slice ECG (and aligned SpO2 when available) into 30-second segments and
    write a CSV that infer.py can consume.

    SpO2 alignment
    ──────────────
    ECG segment i covers ECG samples [i×3750 : (i+1)×3750].
    Its real start timestamp is resolved from packet_timestamps /
    ecg_chunk_lengths and used to look up the matching point in the 1 Hz
    SpO2 stream via spo2_timestamps / spo2_chunk_lengths.

    Falls back to index-based alignment (assumes both streams started at
    the same moment) when timestamp coverage is insufficient.

    Returns number of segments written.
    """
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

    # Build time indexes for timestamp-based SpO2 alignment
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

        # ── Segment start timestamp ───────────────────────────────────────
        # Always resolved via the cumulative-sample time index, which
        # searches outward for the nearest document with a valid timestamp
        # if the document covering this exact sample has none. This never
        # silently emits "" for a segment unless *no* document in the whole
        # admission has a usable timestamp (see _timestamp_for_sample).
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

        # ── SpO2 features ─────────────────────────────────────────────────
        if ts_align and seg_start_ts is not None:
            starts_s, cum_s = spo2_time_idx
            spo2_start = _sample_index_for_time(
                seg_start_ts, starts_s, cum_s, spo2_chunk_lengths, FS_SPO2)
        else:
            spo2_start = seg_i * SAMPLES_SPO2_SEG if has_spo2_global else None

        if spo2_start is not None and has_spo2_global:
            spo2_end = spo2_start + SAMPLES_SPO2_SEG
            if spo2_end <= len(spo2_signal):
                spo2_seg = spo2_signal[spo2_start:spo2_end]
                spo2_feats = _compute_spo2_features(
                    spo2_seg,
                    baseline_override=global_spo2_baseline)
            elif spo2_start < len(spo2_signal):
                # Tail: partial segment — use whatever remains if ≥ 10 samples
                spo2_seg = spo2_signal[spo2_start:]
                spo2_feats = _compute_spo2_features(
                    spo2_seg,
                    baseline_override=global_spo2_baseline)
                spo2_seg = np.pad(spo2_seg, (0, SAMPLES_SPO2_SEG - len(spo2_seg)), constant_values=np.nan)
            else:
                spo2_feats = dict(_SPO2_DEFAULTS)
                spo2_seg = np.full(SAMPLES_SPO2_SEG, np.nan)
        else:
            spo2_feats = dict(_SPO2_DEFAULTS)
            spo2_seg = np.full(SAMPLES_SPO2_SEG, np.nan)

        if spo2_feats["has_spo2"] == 1:
            n_spo2_segs += 1

        # ── Build row ─────────────────────────────────────────────────────
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
        row.update(dict(zip(SPO2_RAW_COLS, np.round(spo2_seg.astype(float), 2).tolist())))
        row.update(dict(zip(ECG_COLS, np.round(seg.astype(float), 6).tolist())))
        rows.append(row)

    if n_unresolved_ts:
        logger.warning(
            "[CSV] %d / %d segments had no resolvable timestamp anywhere in "
            "the admission's document metadata — written with blank "
            "'timestamp'. Downstream timestamp-derived duration calculations "
            "should treat these admissions' duration as unreliable.",
            n_unresolved_ts, n_segs)

    out_df = pd.DataFrame(rows)
    fixed_cols = (
        ["admissionId", "facilityId", "timestamp", "segment_idx",
         "analysis.summary.signal_quality",
         "analysis.heart_rate_bpm",
         "analysis.background_rhythm"]
        + HR_COLS + RHYTHM_COLS + ECTOPY_COLS
        + SPO2_FEATURE_COLS + ["has_spo2"]
        + SPO2_RAW_COLS
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
#  INFERENCE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_on_csv(
    csv_path:     str,
    model_path:   str,
    scaler_path:  str,
    threshold:    float,
    out_dir:      str,
    admission_id: str,
) -> Optional[Dict]:
    """Call infer.py run_inference() and return the summary dict."""
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
    db,
    admission_id:      str,
    facility_id:       str,
    model_path:        str,
    scaler_path:       str,
    threshold:         float,
    output_dir:        str,
    dry_run:           bool = False,
    no_sleep_filter:   bool = False,
) -> Optional[Dict]:
    logger.info("=" * 60)
    logger.info("  Processing: %s", admission_id)
    logger.info("=" * 60)

    adm_out_dir = os.path.join(output_dir, admission_id)
    os.makedirs(adm_out_dir, exist_ok=True)

    # ── ECG ──────────────────────────────────────────────────────────────────
    ecg_signal, packet_timestamps, ecg_chunk_lengths = extract_ecg_signal(db, admission_id)
    if len(ecg_signal) == 0:
        return None

    nan_mask = ~np.isfinite(ecg_signal)
    if nan_mask.any():
        ecg_signal[nan_mask] = float(np.nanmean(ecg_signal))
        logger.warning("[PROCESS] Filled %d NaN samples in ECG", int(nan_mask.sum()))

    # ── SpO2 ─────────────────────────────────────────────────────────────────
    spo2_signal, spo2_timestamps, spo2_chunk_lengths = extract_spo2_signal(db, admission_id)

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
        return None

    summary = summary_xgb.copy()

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

    logger.info("[RESULT] %s  XGB_AHI=%s  ValidatedAHI=%s",
                admission_id,
                f"{summary.get('ahi_proxy'):.1f}" if summary.get("ahi_proxy") is not None else "N/A",
                f"{summary.get('validated_ahi'):.1f}" if summary.get("validated_ahi") is not None else "N/A",
                )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE WRITE-BACK
# ══════════════════════════════════════════════════════════════════════════════

def get_existing_result(supabase_client, admission_id: str) -> Optional[Dict]:
    """Return the existing Supabase row for this admission, or None."""
    try:
        resp = (
            supabase_client.table("apnea_results")
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
        # Validation fields
        "validated_ahi":             summary.get("validated_ahi"),
        "validation_confirmed":      summary.get("validation_confirmed"),
        "validation_probable":       summary.get("validation_probable"),
        "validation_uncertain":      summary.get("validation_uncertain"),
        "validation_unconfirmed":    summary.get("validation_unconfirmed"),
        "validation_mean_score":     summary.get("validation_mean_score"),
        "physiologically_supported": summary.get("physiologically_supported"),
    }
    try:
        (supabase_client.table("apnea_results")
         .upsert(record, on_conflict="admission_id")
         .execute())
        logger.info("[SUPABASE] Upserted %s (XGB_AHI=%s)",
                    admission_id,
                    f"{ahi:.1f}" if ahi is not None else "N/A")
    except Exception as e:
        logger.error("[SUPABASE] Failed for %s: %s", admission_id, e)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MongoDB → ECG+SpO2 → Apnea inference → Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--admission",       default=None)
    p.add_argument("--since",           default=None,
                   help="Last N hours, e.g. 24h")
    p.add_argument("--from",            dest="date_from", default=None,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--to",              dest="date_to",   default=None,
                   help="End date YYYY-MM-DD")
    p.add_argument("--model",           default=os.environ.get("MODEL_PATH",  "apnea_model.keras"))
    p.add_argument("--scaler",          default=os.environ.get("SCALER_PATH", "apnea_scaler.pkl"))
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

    # ── Load threshold from training artefact if env/CLI did not override ──────
    # apnea_thresholds.json is written by train.py with the cross-validated
    # optimal threshold. The .env THRESHOLD and CLI --threshold override it
    # (useful for tuning sensitivity/specificity post-deployment).
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

    with _SSHTunnel():
        db              = get_db()
        supabase_client = get_supabase_client() if args.write_supabase else None

        # ── Resolve admissions ────────────────────────────────────────────────
        if args.admission:
            doc = db.ecg_data_by_admission_id.find_one(
                {"admissionId": args.admission}, {"facilityId": 1, "_id": 0})
            if not doc:
                logger.error("[SETUP] %s not found in MongoDB", args.admission)
                sys.exit(1)
            admissions = [{"admissionId": args.admission,
                           "facilityId":  doc.get("facilityId", "")}]
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
                db, since_hours=since_hours,
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
                existing = get_existing_result(supabase_client, admission_id)
                if existing:
                    logger.info("[MAIN] %s already processed at %s — skip "
                                "(--reprocess to override)",
                                admission_id, existing.get("processed_at"))
                    continue

            try:
                summary = process_admission(
                    db=db, admission_id=admission_id, facility_id=facility_id,
                    model_path=args.model, scaler_path=args.scaler,
                    threshold=args.threshold, output_dir=args.out_dir,
                    dry_run=args.dry_run,
                    no_sleep_filter=args.no_sleep_filter,
                )
                if summary:
                    successes += 1
                    if args.write_supabase and not args.dry_run and supabase_client:
                        write_results_to_supabase(
                            supabase_client, admission_id, facility_id, summary)
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