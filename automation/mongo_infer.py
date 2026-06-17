"""
mongo_infer.py
==============
Pulls ECG + SpO2 data for a given admissionId from MongoDB,
assembles 30-second segments, runs the apnea inference pipeline,
and writes results to a local output directory (and optionally Supabase).

Usage
-----
# Single admission
python automation/mongo_infer.py --admission ADM1819906487

# All admissions recorded in the last 24 hours
python automation/mongo_infer.py --since 24h

# All admissions recorded between two dates
python automation/mongo_infer.py --from 2026-06-01 --to 2026-06-15

# Dry run — extract and save CSVs but skip inference
python automation/mongo_infer.py --admission ADM1819906487 --dry-run

# Write results to Supabase after inference
python automation/mongo_infer.py --since 24h --write-supabase

Environment variables (or set directly in MONGO_CONFIG / SUPABASE_CONFIG below)
--------------------------------------------------------------------------------
MONGO_URI       = mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB        = your_database_name
SUPABASE_URL    = https://your-project.supabase.co
SUPABASE_KEY    = your-service-role-or-anon-key
MODEL_PATH      = /path/to/apnea_model.keras      (default: apnea_model.keras)
SCALER_PATH     = /path/to/apnea_scaler.pkl       (default: apnea_scaler.pkl)
THRESHOLD       = 0.45                             (default: 0.45 — matches infer.py CLI default)
OUTPUT_DIR      = /path/to/output/                (default: infer_output/)
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Load .env from project root automatically ─────────────────────────────────
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass

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

# ── MongoDB config ─────────────────────────────────────────────────────────────
MONGO_CONFIG = {
    "uri": os.environ.get("MONGO_URI"),
    "db":  os.environ.get("MONGO_DB"),
}

# ── Supabase config ────────────────────────────────────────────────────────────
SUPABASE_CONFIG = {
    "url": os.environ.get("SUPABASE_URL"),
    "key": os.environ.get("SUPABASE_KEY"),
}

# ── Pipeline config ───────────────────────────────────────────────────────────
FS_ECG              = 125
FS_SPO2             = 100       # pleth ~100 Hz
SEGMENT_LEN_S       = 30
SAMPLES_SEG         = FS_ECG * SEGMENT_LEN_S        # 3750 ECG samples per segment
SAMPLES_SPO2_SEG    = FS_SPO2 * SEGMENT_LEN_S       # 3000 SpO2 samples per segment
SAMPLES_PER_ECG_DOC = 125
DOCS_PER_SEG        = SAMPLES_SEG // SAMPLES_PER_ECG_DOC
MIN_SEGMENTS        = 11
MIN_DURATION_MINUTES = 30
COMPLETION_GAP_HOURS = 2

ECG_COLS    = [f"ecgData[{i}]" for i in range(SAMPLES_SEG)]
HR_COLS     = [f"analysis.segments[{i}].morphology.hr_bpm" for i in range(6)]
RHYTHM_COLS = [f"analysis.segments[{i}].rhythm_label" for i in range(6)]
ECTOPY_COLS = [f"analysis.segments[{i}].ectopy_label" for i in range(6)]

# SpO2 feature columns written into the segment CSV so infer.py can read them
SPO2_FEATURE_COLS = [
    "spo2_mean", "spo2_min", "spo2_delta_index",
    "odi", "t90", "spo2_approx_entropy",
]


# ══════════════════════════════════════════════════════════════════════════════
#  MONGODB CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.error("pymongo not installed — pip install pymongo")
        sys.exit(1)
    client = MongoClient(MONGO_CONFIG["uri"], serverSelectionTimeoutMS=10000)
    try:
        client.admin.command("ping")
        logger.info("[MONGO] Connected to %s", MONGO_CONFIG["db"])
    except Exception as e:
        logger.error("[MONGO] Connection failed: %s", e)
        sys.exit(1)
    return client[MONGO_CONFIG["db"]]


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_supabase_client():
    try:
        from supabase import create_client, Client
    except ImportError:
        logger.error("supabase-py not installed — pip install supabase")
        sys.exit(1)

    url = SUPABASE_CONFIG["url"]
    key = SUPABASE_CONFIG["key"]

    if not url:
        logger.error("[SUPABASE] SUPABASE_URL is not configured.")
        sys.exit(1)
    if not key:
        logger.error("[SUPABASE] SUPABASE_KEY is not configured.")
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
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[Dict]:
    pipeline = []
    ts_filter = {}
    since_cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)
                    if since_hours else None)
    from_cutoff = date_from.replace(tzinfo=timezone.utc) if date_from else None

    if since_cutoff and from_cutoff:
        # Both filters given — don't let one silently clobber the other.
        # Merge by taking the more restrictive (later) lower bound.
        chosen = max(since_cutoff, from_cutoff)
        logger.warning(
            "[DISCOVER] Both --since (%s) and --from (%s) were given — "
            "using the later of the two as the cutoff: %s",
            since_cutoff, from_cutoff, chosen,
        )
        ts_filter["$gte"] = chosen
    elif since_cutoff:
        ts_filter["$gte"] = since_cutoff
    elif from_cutoff:
        ts_filter["$gte"] = from_cutoff

    if date_to:
        ts_filter["$lte"] = date_to.replace(tzinfo=timezone.utc)
    if ts_filter:
        pipeline.append({"$match": {"utcTimestamp": ts_filter}})

    pipeline += [
        {"$group": {
            "_id": "$admissionId",
            "facilityId": {"$first": "$facilityId"},
            "first_ts":   {"$min": "$utcTimestamp"},
            "last_ts":    {"$max": "$utcTimestamp"},
            "n_docs":     {"$sum": 1},
        }},
        {"$project": {
            "_id": 0,
            "admissionId":  "$_id",
            "facilityId":   1,
            "first_ts":     1,
            "last_ts":      1,
            "n_docs":       1,
            "duration_min": {
                "$divide": [
                    {"$subtract": ["$last_ts", "$first_ts"]},
                    60000,
                ]
            },
        }},
    ]

    results = list(db.ecg_data_by_admission_id.aggregate(pipeline))
    now = datetime.now(timezone.utc)
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
            logger.info(
                "[DISCOVER] %s  duration=%.0f min  docs=%d  "
                "last_seen=%.1fh ago  ✓ ELIGIBLE",
                r["admissionId"], r["duration_min"],
                r["n_docs"], hours_since_last,
            )
        else:
            reason = []
            if not recording_ended:
                reason.append(f"still recording ({hours_since_last:.1f}h since last packet)")
            if not has_enough_data:
                reason.append(f"too short ({r['duration_min']:.0f} min)")
            logger.info("[DISCOVER] %s  SKIP — %s",
                        r["admissionId"], " | ".join(reason))

    logger.info("[DISCOVER] %d / %d admissions eligible",
                len(completed), len(results))
    return completed


# ══════════════════════════════════════════════════════════════════════════════
#  ECG EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_ecg_signal(db, admission_id: str) -> Tuple[np.ndarray, List[datetime], List[int]]:
    """
    Returns (signal, doc_timestamps, doc_chunk_lengths).
    doc_timestamps[i] / doc_chunk_lengths[i] describe the i-th retained chunk
    that was concatenated into `signal`, in order — this lets callers map an
    absolute sample index back to a real timestamp without assuming every
    document contributed exactly SAMPLES_PER_ECG_DOC samples.
    """
    logger.info("[ECG] Fetching documents for %s ...", admission_id)
    docs = list(
        db.ecg_data_by_admission_id
        .find(
            {"admissionId": admission_id},
            {"packetNo": 1, "utcTimestamp": 1, "value": 1, "_id": 0},
        )
        .sort("packetNo", 1)
    )

    if not docs:
        logger.error("[ECG] No documents found for %s", admission_id)
        return np.array([]), [], []

    logger.info("[ECG] %d documents retrieved", len(docs))
    signal_chunks, timestamps, chunk_lengths, skipped = [], [], [], 0

    for doc in docs:
        raw = doc.get("value")
        if isinstance(raw, list) and len(raw) > 0:
            chunk = raw[0] if isinstance(raw[0], list) else raw
            if len(chunk) > 0:
                signal_chunks.append(np.array(chunk, dtype=float))
                timestamps.append(doc.get("utcTimestamp"))
                chunk_lengths.append(len(chunk))
                continue
        skipped += 1

    if skipped:
        logger.warning("[ECG] Skipped %d documents with missing/empty value", skipped)
    if not signal_chunks:
        return np.array([]), [], []

    # ── Validate assumed per-doc sample count / FS_ECG against reality ──────
    # build_segment_csv and infer.py both assume each document holds
    # SAMPLES_PER_ECG_DOC samples at FS_ECG Hz. If the device is actually
    # sending at a different rate, every downstream segment boundary,
    # timestamp lookup, and the AHI denominator would be silently wrong.
    sample_lengths = np.array(chunk_lengths)
    modal_len = int(np.bincount(sample_lengths).argmax()) if len(sample_lengths) else 0
    if modal_len and modal_len != SAMPLES_PER_ECG_DOC:
        implied_fs = modal_len  # docs are nominally 1 second each
        logger.error(
            "[ECG] *** SAMPLE-RATE MISMATCH for %s *** "
            "Documents contain %d samples/doc, but SAMPLES_PER_ECG_DOC=%d "
            "(FS_ECG=%d) is assumed throughout the pipeline. This implies "
            "the actual ECG rate is closer to %d Hz, not %d Hz. Refusing to "
            "process this admission — update SAMPLES_PER_ECG_DOC/FS_ECG (or "
            "add resampling) before re-running, otherwise segment boundaries "
            "and the AHI denominator will be wrong by roughly %.1fx.",
            admission_id, modal_len, SAMPLES_PER_ECG_DOC, FS_ECG,
            implied_fs, FS_ECG, modal_len / SAMPLES_PER_ECG_DOC,
        )
        return np.array([]), [], []
    elif modal_len:
        n_mismatched = int(np.sum(sample_lengths != modal_len))
        if n_mismatched:
            logger.warning(
                "[ECG] %d / %d documents have a non-standard chunk length "
                "(expected %d) — likely partial/edge packets; proceeding.",
                n_mismatched, len(sample_lengths), SAMPLES_PER_ECG_DOC,
            )

    signal = np.concatenate(signal_chunks)
    logger.info("[ECG] Signal assembled: %d samples  (%.1f minutes at %d Hz)",
                len(signal), len(signal) / FS_ECG / 60, FS_ECG)
    return signal, timestamps, chunk_lengths


# ══════════════════════════════════════════════════════════════════════════════
#  SPO2 EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _validate_spo2_units(signal: np.ndarray, admission_id: str) -> bool:
    """
    Sanity-check whether `signal` actually looks like SpO2 percentages
    (0-100, slow-varying) rather than a raw AC/DC pleth waveform
    (large/arbitrary range, strong periodic oscillation at the cardiac rate).

    This does NOT prove the data is correct — it only catches the common
    failure mode where the collection stores raw photoplethysmography
    counts instead of a computed SpO2% channel. Uses a spectral check
    (power concentrated in the ~0.5-3 Hz pulsatile band is the signature of
    a true AC pleth waveform; a computed SpO2% channel should show almost
    none) rather than a simple value-range/noise heuristic, since ordinary
    measurement noise on a real SpO2% channel can look superficially
    "oscillatory" at the sample level.
    """
    if len(signal) < FS_SPO2 * 10:
        return True  # too little data to judge either way; let it through

    finite = signal[np.isfinite(signal)]
    if len(finite) == 0:
        return False

    frac_in_range = float(np.mean((finite >= 30) & (finite <= 105)))

    nperseg = min(len(finite), max(64, FS_SPO2 * 10))
    f, pxx = welch(finite - np.mean(finite), fs=FS_SPO2, nperseg=nperseg)
    total_power = float(np.sum(pxx)) or 1.0
    pulsatile_band = (f >= 0.5) & (f <= 3.0)  # 30-180 bpm cardiac rate
    pulsatile_frac = float(np.sum(pxx[pulsatile_band])) / total_power

    looks_like_spo2 = frac_in_range > 0.9 and pulsatile_frac < 0.15

    if not looks_like_spo2:
        logger.error(
            "[SPO2] *** UNIT CHECK FAILED for %s *** "
            "%.0f%% of samples in [30,105]%%, %.0f%% of signal power is in "
            "the 0.5-3 Hz cardiac/pulsatile band. This looks like a raw "
            "pleth/AC waveform, not a computed SpO2%% channel. Refusing to "
            "compute SpO2 features from this data; has_spo2 will be 0 for "
            "this admission. Confirm what spo2_pleth_by_admission_id "
            "actually stores before overriding this check.",
            admission_id, frac_in_range * 100, pulsatile_frac * 100,
        )
        return False

    logger.info("[SPO2] Unit check passed for %s (%.0f%% in-range, "
                "%.0f%% pulsatile-band power) — treating as SpO2%%.",
                admission_id, frac_in_range * 100, pulsatile_frac * 100)
    return True


def extract_spo2_signal(
    db, admission_id: str
) -> Tuple[np.ndarray, List[datetime], List[int]]:
    """
    Pull SpO2 data for this admission.
    Returns (signal, doc_timestamps, doc_chunk_lengths) at FS_SPO2 Hz, or
    empty containers if unavailable or if the data fails the SpO2-unit
    sanity check (see `_validate_spo2_units`) — in that case SpO2 features
    are zeroed for the whole admission rather than computed from data that
    may actually be a raw pleth waveform.
    """
    docs = list(
        db.spo2_pleth_by_admission_id
        .find(
            {"admissionId": admission_id},
            {"utcTimestamp": 1, "value": 1, "_id": 0},
        )
        .sort("utcTimestamp", 1)
    )

    if not docs:
        logger.warning("[SPO2] No pleth data for %s — SpO2 features will be zeroed",
                       admission_id)
        return np.array([]), [], []

    chunks, timestamps, chunk_lengths = [], [], []
    for doc in docs:
        raw = doc.get("value", [])
        if isinstance(raw, list) and len(raw) > 0:
            chunks.append(np.array(raw, dtype=float))
            timestamps.append(doc.get("utcTimestamp"))
            chunk_lengths.append(len(raw))

    if not chunks:
        return np.array([]), [], []

    signal = np.concatenate(chunks)
    logger.info("[SPO2] %d pleth samples retrieved for %s (%.1f min at %d Hz)",
                len(signal), admission_id, len(signal) / FS_SPO2 / 60, FS_SPO2)

    if not _validate_spo2_units(signal, admission_id):
        return np.array([]), [], []

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
    rr_ms = np.diff(r_peaks) / fs * 1000.0
    return float(60000.0 / (np.mean(rr_ms) + 1e-6))


def _subseg_hrs(ecg_seg: np.ndarray, fs: int, n: int = 6) -> List[float]:
    chunk_len = len(ecg_seg) // n
    hrs = []
    for i in range(n):
        chunk = ecg_seg[i * chunk_len:(i + 1) * chunk_len]
        try:
            bp    = _bandpass(chunk, fs)
            peaks = _detect_r_peaks(bp, fs)
            hrs.append(_mean_hr(peaks, fs))
        except Exception:
            hrs.append(0.0)
    return hrs


def _assess_signal_quality(
    raw_seg: np.ndarray,
    bp_seg: np.ndarray,
    fs: int,
    sub_hrs: List[float],
) -> str:
    """
    Lightweight ECG signal-quality check, run per 30-second segment.

    Returns "acceptable" or "poor". This intentionally errs on the side of
    flagging real problems (flatline, saturation/clipping, no detectable
    heartbeat) rather than trying to be a clinical-grade quality classifier —
    it exists to stop obviously-bad segments from reaching the model, which
    is what the downstream `qual_ok` gate in infer.py expects.
    """
    if raw_seg is None or len(raw_seg) == 0:
        return "poor"

    finite = raw_seg[np.isfinite(raw_seg)]
    if len(finite) < 0.9 * len(raw_seg):
        return "poor"  # too many NaN/inf samples

    # Flatline: near-zero variance for a stretch that should show cardiac activity
    sd = float(np.std(finite))
    if sd < 1e-6:
        return "poor"

    # Saturation / clipping: many samples pinned at the signal's own min or max
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi > lo:
        at_extreme = np.mean((finite <= lo + 1e-9) | (finite >= hi - 1e-9))
        if at_extreme > 0.10:
            return "poor"

    # Local flatline runs: any 2+ second stretch with essentially no change
    window = max(1, int(fs * 2))
    if len(bp_seg) >= window:
        roll_std = pd.Series(bp_seg).rolling(window).std().dropna().values
        if len(roll_std) > 0 and float(np.min(roll_std)) < 1e-4 * (sd + 1e-9):
            return "poor"

    # No physiologically plausible heartbeat detected in most sub-segments
    valid_hrs = [h for h in sub_hrs if h > 0]
    if len(valid_hrs) < len(sub_hrs) // 2:
        return "poor"
    if any(h < 20 or h > 220 for h in valid_hrs):
        return "poor"

    return "acceptable"


# ══════════════════════════════════════════════════════════════════════════════
#  SPO2 FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _approx_entropy(sig: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Approximate entropy of a 1D signal.
    m=2, r = r_factor * std(sig) — standard ApEn parameters.
    Returns 0.0 for signals too short or with zero variance.
    """
    n = len(sig)
    if n < 2 * m + 2:
        return 0.0
    sd = float(np.std(sig))
    if sd < 1e-9:
        return 0.0
    r = r_factor * sd

    def _phi(m_val):
        # Template vectors of length m_val
        templates = np.array([sig[i:i + m_val] for i in range(n - m_val + 1)])
        count = np.sum(
            np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2) <= r,
            axis=1,
        )
        return float(np.mean(np.log(count / (n - m_val + 1) + 1e-10)))

    return abs(_phi(m) - _phi(m + 1))


def _compute_global_spo2_baseline(spo2_signal: np.ndarray, fs: int = FS_SPO2) -> float:
    """
    Estimate a patient-level SpO2 baseline for ODI desaturation detection,
    rather than letting each 30 s segment use its own mean as the baseline.

    Using the segment's own mean understates ODI for chronically desaturated
    patients (dips get measured relative to an already-low segment average
    instead of the patient's normal level). The 90th percentile of the full
    recording is a robust proxy for "normal" saturation outside of
    desaturation events — transient dips pull the mean down but rarely touch
    the upper percentiles, so this stays stable even on noisy recordings.
    """
    finite = spo2_signal[np.isfinite(spo2_signal)]
    if len(finite) < fs * 60:  # need at least ~1 minute of data to trust this
        return 97.0
    clipped = np.clip(finite, 50.0, 100.0)
    return float(np.percentile(clipped, 90))


def _compute_spo2_features(
    spo2_seg: np.ndarray,
    fs: int = FS_SPO2,
    baseline_override: Optional[float] = None,
) -> Dict:
    """
    Compute the 6 SpO2 features that APNEA_FEATURE_COLS expects, plus has_spo2 flag.

    Features
    --------
    spo2_mean         : mean SpO2 over the segment
    spo2_min          : minimum SpO2
    spo2_delta_index  : mean absolute sample-to-sample change (variability proxy)
    odi               : oxygen desaturation index — desaturations ≥3% from a
                        patient-level baseline (90th-percentile of the full
                        recording, passed in as `baseline_override`) rather
                        than the segment's own mean — per hour, scaled to the
                        30 s window
    t90               : fraction of segment where SpO2 < 90%
    spo2_approx_entropy : ApEn regularity measure (low = regular, high = irregular)

    Returns default neutral values + has_spo2=0 if the segment is empty/invalid.
    """
    defaults = {
        "spo2_mean": 97.0, "spo2_min": 97.0, "spo2_delta_index": 0.0,
        "odi": 0.0, "t90": 0.0, "spo2_approx_entropy": 0.0,
        "has_spo2": 0,
    }

    if spo2_seg is None or len(spo2_seg) < 10:
        return defaults

    seg = spo2_seg.astype(float).copy()

    # Clip to physiological range — raw pleth can drift outside 0-100
    seg = np.clip(seg, 50.0, 100.0)

    # Forward-fill any NaNs, then drop if still bad
    seg = pd.Series(seg).ffill().bfill().values
    if not np.isfinite(seg).all() or np.std(seg) < 1e-9:
        return defaults

    spo2_mean = float(np.mean(seg))
    spo2_min  = float(np.min(seg))

    # Delta index: mean |Δ| per sample
    spo2_delta_index = float(np.mean(np.abs(np.diff(seg))))

    # T90: fraction of samples below 90%
    t90 = float(np.mean(seg < 90.0))

    # ODI: count ≥3% dips from a patient-level baseline (90th-percentile of
    # the whole recording, passed in via baseline_override) rather than this
    # segment's own mean. A dip = any point where seg < (baseline - 3). We
    # count contiguous events (not individual samples) to avoid one long dip
    # inflating the count. Falls back to the segment mean only if no
    # recording-level baseline was supplied (e.g. when called standalone).
    baseline = baseline_override if baseline_override is not None else spo2_mean
    below    = seg < (baseline - 3.0)
    n_events = 0
    in_event = False
    for v in below:
        if v and not in_event:
            n_events += 1
            in_event = True
        elif not v:
            in_event = False
    # Scale to events per hour (segment is 30 s → ×120)
    odi = float(n_events * 120.0)

    # Approximate entropy — downsample to ~4 Hz first. SpO2 desaturation
    # dynamics are slow (seconds-scale), so this loses no clinically
    # relevant information while shrinking the O(n^2) template-matching
    # array from ~3000^2 to ~120^2 (the dominant cost of _approx_entropy).
    apen_fs = 4
    decim   = max(1, fs // apen_fs)
    seg_ds  = seg[::decim] if decim > 1 else seg
    apen    = _approx_entropy(seg_ds)

    return {
        "spo2_mean":           round(spo2_mean, 3),
        "spo2_min":            round(spo2_min, 3),
        "spo2_delta_index":    round(spo2_delta_index, 6),
        "odi":                 round(odi, 2),
        "t90":                 round(t90, 6),
        "spo2_approx_entropy": round(apen, 6),
        "has_spo2":            1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CSV BUILDER  (format compatible with infer.py)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  TIME ALIGNMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_ts(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        parsed = pd.to_datetime(ts, utc=True)
        return parsed.to_pydatetime()
    except Exception:
        return None


def _build_time_index(
    doc_timestamps: List[datetime], doc_lengths: List[int]
) -> Optional[Tuple[List[datetime], List[int]]]:
    """
    Build a (start_ts_per_doc, cumulative_sample_start_per_doc) index from
    per-document timestamps/lengths. Returns None if fewer than 2 documents
    have usable timestamps — in that case callers should fall back to naive
    index-based alignment.
    """
    starts, cum, running = [], [], 0
    n_valid = 0
    for ts, length in zip(doc_timestamps, doc_lengths):
        norm = _normalize_ts(ts)
        starts.append(norm)
        cum.append(running)
        running += length
        if norm is not None:
            n_valid += 1
    if n_valid < 2:
        return None
    return starts, cum


def _sample_index_for_time(
    target_ts: datetime,
    starts: List[datetime],
    cum: List[int],
    lengths: List[int],
    fs: int,
) -> Optional[int]:
    """
    Map a wall-clock timestamp to an absolute sample index in the
    concatenated signal, by finding which document's time window contains
    `target_ts` and offsetting within it. Returns None if no document has a
    usable timestamp at or before target_ts.
    """
    best_idx = None
    for i, ts in enumerate(starts):
        if ts is not None and ts <= target_ts:
            best_idx = i
        elif ts is not None and ts > target_ts:
            break
    if best_idx is None:
        return None
    offset_s = (target_ts - starts[best_idx]).total_seconds()
    offset_samples = int(round(offset_s * fs))
    offset_samples = max(0, min(offset_samples, lengths[best_idx]))
    return cum[best_idx] + offset_samples


# ══════════════════════════════════════════════════════════════════════════════
#  CSV BUILDER  (format compatible with infer.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_segment_csv(
    ecg_signal: np.ndarray,
    spo2_signal: np.ndarray,
    admission_id: str,
    facility_id: str,
    packet_timestamps: List[datetime],
    output_path: str,
    ecg_chunk_lengths: Optional[List[int]] = None,
    spo2_timestamps: Optional[List[datetime]] = None,
    spo2_chunk_lengths: Optional[List[int]] = None,
) -> int:
    """
    Slice ECG (and aligned SpO2 when available) into 30-second segments
    and write a CSV that infer.py can consume.

    SpO2 alignment
    --------------
    Each ECG segment's real start timestamp is resolved from
    `packet_timestamps`/`ecg_chunk_lengths` and used to look up the matching
    point in the SpO2 stream via `spo2_timestamps`/`spo2_chunk_lengths` —
    i.e. segments are matched by actual wall-clock time, not by assuming
    both streams started at the same sample index. This is robust to the
    two streams starting at different moments and to ECG (sorted by
    packetNo) vs SpO2 (sorted by utcTimestamp) using different orderings
    upstream.

    If timestamps aren't usable for either stream (too few valid values),
    this falls back to the original index-based alignment with a warning,
    so the function still degrades gracefully rather than crashing.

    Returns number of segments written.
    """
    n_segs = len(ecg_signal) // SAMPLES_SEG
    if n_segs < MIN_SEGMENTS:
        logger.warning(
            "[CSV] Only %d complete segments (need ≥%d). "
            "Recording is %.1f min — need ≥%.0f min.",
            n_segs, MIN_SEGMENTS,
            len(ecg_signal) / FS_ECG / 60,
            MIN_SEGMENTS * SEGMENT_LEN_S / 60,
        )
        if n_segs == 0:
            return 0

    has_spo2_global = len(spo2_signal) >= SAMPLES_SPO2_SEG
    if has_spo2_global:
        logger.info("[CSV] SpO2 signal available (%d samples) — "
                    "will compute SpO2 features per segment", len(spo2_signal))
        global_spo2_baseline = _compute_global_spo2_baseline(spo2_signal)
        logger.info("[CSV] Patient-level SpO2 baseline (90th pct): %.1f%%",
                    global_spo2_baseline)
    else:
        logger.warning("[CSV] SpO2 signal too short or missing — "
                       "has_spo2=0 for all segments")
        global_spo2_baseline = None

    # Build time indexes for both streams so SpO2 can be matched to ECG by
    # actual wall-clock time rather than by parallel array index.
    ecg_chunk_lengths = ecg_chunk_lengths or [SAMPLES_PER_ECG_DOC] * len(packet_timestamps)
    ecg_time_index  = _build_time_index(packet_timestamps, ecg_chunk_lengths)
    spo2_time_index = (
        _build_time_index(spo2_timestamps, spo2_chunk_lengths)
        if (has_spo2_global and spo2_timestamps and spo2_chunk_lengths)
        else None
    )
    use_timestamp_alignment = has_spo2_global and ecg_time_index and spo2_time_index
    if has_spo2_global and not use_timestamp_alignment:
        logger.warning(
            "[CSV] Insufficient timestamp coverage on ECG and/or SpO2 stream — "
            "falling back to index-based SpO2 alignment (assumes both streams "
            "started at the same moment)."
        )

    logger.info("[CSV] Building %d × 30s segments (SpO2 alignment: %s) ...",
                n_segs, "timestamp-based" if use_timestamp_alignment else "index-based")
    rows = []
    n_spo2_segs = 0

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

        # ── Segment start timestamp (used for both the CSV row and SpO2 lookup) ──
        seg_start_ts = None
        if ecg_time_index is not None:
            starts, cum = ecg_time_index
            # Find the ECG doc containing ecg_start, offset within it
            doc_idx = max(0, np.searchsorted(cum, ecg_start, side="right") - 1)
            if starts[doc_idx] is not None:
                offset_s = (ecg_start - cum[doc_idx]) / FS_ECG
                seg_start_ts = starts[doc_idx] + timedelta(seconds=offset_s)
        if seg_start_ts is not None:
            ts_str = str(seg_start_ts)
        else:
            doc_idx_fallback = ecg_start // SAMPLES_PER_ECG_DOC
            if (doc_idx_fallback < len(packet_timestamps)
                    and packet_timestamps[doc_idx_fallback] is not None):
                ts_str = str(packet_timestamps[doc_idx_fallback])
            else:
                ts_str = ""

        # ── SpO2 features ──────────────────────────────────────────────────
        if use_timestamp_alignment and seg_start_ts is not None:
            starts_s, cum_s = spo2_time_index
            spo2_start = _sample_index_for_time(
                seg_start_ts, starts_s, cum_s, spo2_chunk_lengths, FS_SPO2)
        else:
            spo2_start = seg_i * SAMPLES_SPO2_SEG  # index-based fallback

        if spo2_start is None:
            spo2_feats = {
                "spo2_mean": 97.0, "spo2_min": 97.0, "spo2_delta_index": 0.0,
                "odi": 0.0, "t90": 0.0, "spo2_approx_entropy": 0.0,
                "has_spo2": 0,
            }
        else:
            spo2_end = spo2_start + SAMPLES_SPO2_SEG
            if has_spo2_global and spo2_end <= len(spo2_signal):
                spo2_seg     = spo2_signal[spo2_start:spo2_end]
                spo2_feats   = _compute_spo2_features(
                    spo2_seg, baseline_override=global_spo2_baseline)
            elif has_spo2_global and spo2_start < len(spo2_signal):
                # Partial tail segment — use whatever remains if ≥10 samples
                spo2_seg   = spo2_signal[spo2_start:]
                spo2_feats = _compute_spo2_features(
                    spo2_seg, baseline_override=global_spo2_baseline)
            else:
                spo2_feats = {
                    "spo2_mean": 97.0, "spo2_min": 97.0, "spo2_delta_index": 0.0,
                    "odi": 0.0, "t90": 0.0, "spo2_approx_entropy": 0.0,
                    "has_spo2": 0,
                }
        if spo2_feats["has_spo2"] == 1:
            n_spo2_segs += 1

        row: dict = {
            "admissionId":                     admission_id,
            "facilityId":                      facility_id,
            "timestamp":                       ts_str,
            "segment_idx":                     seg_i,
            "analysis.summary.signal_quality": sig_quality,
            "analysis.heart_rate_bpm":         round(mean_hr, 1),
            "analysis.background_rhythm":      "",
        }

        # Sub-segment HR columns
        for i, hr in enumerate(sub_hrs):
            row[f"analysis.segments[{i}].morphology.hr_bpm"] = round(hr, 2)
            row[f"analysis.segments[{i}].rhythm_label"]      = ""
            row[f"analysis.segments[{i}].ectopy_label"]      = ""

        # SpO2 feature columns
        row.update(spo2_feats)


        # Raw ECG samples
        for j, v in enumerate(seg):
            row[f"ecgData[{j}]"] = round(float(v), 6)

        rows.append(row)

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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("[CSV] Wrote %d segments → %s  (SpO2 available in %d / %d segments)",
                len(out_df), output_path, n_spo2_segs, n_segs)
    return len(out_df)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_on_csv(
    csv_path: str,
    model_path: str,
    scaler_path: str,
    threshold: float,
    out_dir: str,
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
            csv_path     = csv_path,
            model_path   = model_path,
            scaler_path  = scaler_path,
            threshold    = threshold,
            out_dir      = out_dir,
            admission_id = admission_id,
        )
    except Exception as e:
        logger.error("[INFER] Inference failed for %s: %s", admission_id, e)
        return None

    summary_csv = os.path.join(out_dir, "infer_summary.csv")
    if not os.path.exists(summary_csv):
        logger.warning("[INFER] No summary CSV found after inference for %s", admission_id)
        return None

    summary_df = pd.read_csv(summary_csv)
    matches    = summary_df[summary_df["admission_id"] == admission_id]
    if matches.empty:
        logger.warning("[INFER] No summary row for %s in %s", admission_id, summary_csv)
        return None

    return matches.iloc[0].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS ONE ADMISSION
# ══════════════════════════════════════════════════════════════════════════════

def process_admission(
    db,
    admission_id: str,
    facility_id: str,
    model_path: str,
    scaler_path: str,
    threshold: float,
    output_dir: str,
    dry_run: bool = False,
) -> Optional[Dict]:
    logger.info("=" * 60)
    logger.info("  Processing: %s", admission_id)
    logger.info("=" * 60)

    adm_out_dir = os.path.join(output_dir, admission_id)
    os.makedirs(adm_out_dir, exist_ok=True)

    ecg_signal, packet_timestamps, ecg_chunk_lengths = extract_ecg_signal(db, admission_id)
    if len(ecg_signal) == 0:
        logger.error("[PROCESS] No ECG signal for %s — skipping", admission_id)
        return None

    nan_mask = ~np.isfinite(ecg_signal)
    if nan_mask.any():
        ecg_signal[nan_mask] = float(np.nanmean(ecg_signal))
        logger.warning("[PROCESS] Filled %d NaN samples in ECG", int(nan_mask.sum()))

    spo2_signal, spo2_timestamps, spo2_chunk_lengths = extract_spo2_signal(db, admission_id)

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
        logger.error("[PROCESS] No segments built for %s", admission_id)
        return None

    if dry_run:
        logger.info("[DRY RUN] CSV written to %s — skipping inference", csv_path)
        return {"admission_id": admission_id, "status": "dry_run", "n_segments": n_segs}

    summary = run_inference_on_csv(
        csv_path     = csv_path,
        model_path   = model_path,
        scaler_path  = scaler_path,
        threshold    = threshold,
        out_dir      = adm_out_dir,
        admission_id = admission_id,
    )

    if summary:
        logger.info(
            "[RESULT] %s  AHI=%.1f  Severity=%s  Apnea%%=%s",
            admission_id,
            summary.get("ahi_proxy", 0),
            summary.get("severity", "unknown"),
            summary.get("apnea_pct", "?"),
        )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE RESULT WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_results_to_supabase(
    supabase_client, admission_id: str, facility_id: str, summary: Dict
) -> None:
    """
    Upsert inference results into the Supabase `apnea_results` table.

    Expected table schema (run once in the Supabase SQL editor):
    ──────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS apnea_results (
        admission_id      TEXT PRIMARY KEY,
        facility_id       TEXT,
        processed_at      TIMESTAMPTZ,
        ahi_proxy         FLOAT,
        severity          TEXT,
        apnea_pct         FLOAT,
        total_segments    INT,
        scored_segments   INT,
        n_apnea           INT,
        duration_min      FLOAT,
        model_threshold   FLOAT
    );
    ──────────────────────────────────────────────────────────────
    """
    record = {
        "admission_id":    admission_id,
        "facility_id":     facility_id,
        "processed_at":    datetime.now(timezone.utc).isoformat(),
        "ahi_proxy":       summary.get("ahi_proxy"),
        "severity":        summary.get("severity"),
        "apnea_pct":       summary.get("apnea_pct"),
        "total_segments":  summary.get("total_segments"),
        "scored_segments": summary.get("scored_segments"),
        "n_apnea":         summary.get("n_apnea"),
        "duration_min":    summary.get("duration_min"),
        "model_threshold": summary.get("threshold"),
    }
    try:
        (
            supabase_client
            .table("apnea_results")
            .upsert(record, on_conflict="admission_id")
            .execute()
        )
        logger.info("[SUPABASE] Results upserted for %s", admission_id)
    except Exception as e:
        logger.error("[SUPABASE] Failed to write results for %s: %s", admission_id, e)


def get_existing_result(supabase_client, admission_id: str) -> Optional[Dict]:
    """
    Check whether this admission already has a recorded result in Supabase.
    Returns the existing row (at least `admission_id`/`processed_at`) if
    found, else None. Used to skip redundant re-processing of admissions
    that show up again in overlapping --since/--from windows.
    """
    try:
        resp = (
            supabase_client
            .table("apnea_results")
            .select("admission_id,processed_at")
            .eq("admission_id", admission_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning(
            "[SUPABASE] Idempotency check failed for %s: %s — proceeding "
            "with processing anyway.", admission_id, e,
        )
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MongoDB → ECG+SpO2 extraction → Apnea inference → Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python automation/mongo_infer.py --admission ADM1819906487
  python automation/mongo_infer.py --since 24h
  python automation/mongo_infer.py --from 2026-06-01 --to 2026-06-15
  python automation/mongo_infer.py --since 24h --dry-run
  python automation/mongo_infer.py --since 24h --write-supabase
        """,
    )
    p.add_argument("--admission",      default=None)
    p.add_argument("--since",          default=None,
                   help="Process admissions from last N hours (e.g. 24h, 8h)")
    p.add_argument("--from",           dest="date_from", default=None,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--to",             dest="date_to",   default=None,
                   help="End date YYYY-MM-DD")
    p.add_argument("--model",          default=os.environ.get("MODEL_PATH",  "apnea_model.keras"))
    p.add_argument("--scaler",         default=os.environ.get("SCALER_PATH", "apnea_scaler.pkl"))
    p.add_argument("--threshold",      type=float,
                   default=float(os.environ.get("THRESHOLD", "0.45")))
    p.add_argument("--out-dir",        default=os.environ.get("OUTPUT_DIR",  "infer_output"))
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--write-supabase", action="store_true",
                   help="Write results to Supabase apnea_results table")
    p.add_argument("--reprocess",      action="store_true",
                   help="Process even if a Supabase result already exists "
                        "for this admission (default: skip already-processed "
                        "admissions when --write-supabase is set)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.dry_run:
        for path, name in [(args.model, "model"), (args.scaler, "scaler")]:
            if not os.path.exists(path):
                logger.error("[SETUP] %s not found at %s", name, path)
                logger.error("[SETUP] Train first: python pipeline/pipeline.py --fresh --save-model")
                sys.exit(1)

    db = get_db()

    supabase_client = None
    if args.write_supabase:
        supabase_client = get_supabase_client()

    if args.admission:
        doc = db.ecg_data_by_admission_id.find_one(
            {"admissionId": args.admission},
            {"facilityId": 1, "_id": 0},
        )
        if not doc:
            logger.error("[SETUP] admissionId %s not found in MongoDB", args.admission)
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
            db, since_hours=since_hours, date_from=date_from, date_to=date_to)

    if not admissions:
        logger.info("[MAIN] No admissions to process.")
        return

    logger.info("[MAIN] Processing %d admissions ...", len(admissions))
    successes = failures = 0
    for adm in admissions:
        admission_id = adm["admissionId"]
        facility_id  = adm.get("facilityId", "")

        if args.write_supabase and not args.dry_run and not args.reprocess and supabase_client:
            existing = get_existing_result(supabase_client, admission_id)
            if existing:
                logger.info(
                    "[MAIN] %s already processed at %s — skipping "
                    "(use --reprocess to override)",
                    admission_id, existing.get("processed_at"),
                )
                continue

        try:
            summary = process_admission(
                db           = db,
                admission_id = admission_id,
                facility_id  = facility_id,
                model_path   = args.model,
                scaler_path  = args.scaler,
                threshold    = args.threshold,
                output_dir   = args.out_dir,
                dry_run      = args.dry_run,
            )
            if summary:
                successes += 1
                if args.write_supabase and not args.dry_run and supabase_client:
                    write_results_to_supabase(supabase_client, admission_id, facility_id, summary)
            else:
                failures += 1
        except Exception as e:
            logger.error("[MAIN] Unhandled error for %s: %s", admission_id, e,
                         exc_info=True)
            failures += 1

    logger.info("=" * 60)
    logger.info("[DONE]  %d succeeded  |  %d failed  |  output → %s",
                successes, failures, os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()