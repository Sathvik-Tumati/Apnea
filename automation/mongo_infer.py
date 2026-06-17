"""
mongo_infer.py
==============
Pulls ECG + SpO2 data for a given admissionId from MongoDB,
assembles 30-second segments, runs the apnea inference pipeline,
and writes results to a local output directory (and optionally MongoDB).

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

Environment variables (or set directly in MONGO_CONFIG below)
-------------------------------------------------------------
MONGO_URI      = mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB       = your_database_name
MODEL_PATH     = /path/to/apnea_model.keras      (default: apnea_model.keras)
SCALER_PATH    = /path/to/apnea_scaler.pkl       (default: apnea_scaler.pkl)
THRESHOLD      = 0.60                             (default: 0.60)
OUTPUT_DIR     = /path/to/output/                (default: infer_output/)
"""

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
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── MongoDB config — override via env vars or edit here ───────────────────────
MONGO_CONFIG = {
    "uri":    os.environ.get("MONGO_URI",   "mongodb+srv://YOUR_USER:YOUR_PASS@YOUR_CLUSTER/"),
    "db":     os.environ.get("MONGO_DB",    "YOUR_DATABASE_NAME"),
}

# ── Pipeline config ───────────────────────────────────────────────────────────
FS_ECG        = 125
FS_SPO2       = 100       # pleth docs have ~150 samples at ~1.5s each ≈ 100Hz
SEGMENT_LEN_S = 30
SAMPLES_SEG   = FS_ECG * SEGMENT_LEN_S          # 3750 ECG samples per segment
SAMPLES_PER_ECG_DOC = 125                        # value[0] has 125 samples
DOCS_PER_SEG  = SAMPLES_SEG // SAMPLES_PER_ECG_DOC   # 30 docs per segment
MIN_SEGMENTS  = 11                               # LSTM needs ≥11 for first prediction
MIN_DURATION_MINUTES = 30                        # ignore recordings shorter than this
COMPLETION_GAP_HOURS = 2                         # recording considered done if no
                                                 # new data for this many hours

ECG_COLS  = [f"ecgData[{i}]" for i in range(SAMPLES_SEG)]
HR_COLS   = [f"analysis.segments[{i}].morphology.hr_bpm" for i in range(6)]
RHYTHM_COLS = [f"analysis.segments[{i}].rhythm_label" for i in range(6)]
ECTOPY_COLS = [f"analysis.segments[{i}].ectopy_label" for i in range(6)]


# ══════════════════════════════════════════════════════════════════════════════
#  MONGODB CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    """Return a pymongo database handle."""
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
#  ADMISSION DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_completed_admissions(
    db,
    since_hours: Optional[float] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[Dict]:
    """
    Find admissions where recording has ended (no new ECG data for
    COMPLETION_GAP_HOURS) and that have enough data to run inference.

    Returns list of dicts: {admissionId, facilityId, first_ts, last_ts,
                             n_docs, duration_min}
    """
    pipeline = []

    # Date filter on utcTimestamp
    ts_filter = {}
    if since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        ts_filter["$gte"] = cutoff
    if date_from:
        ts_filter["$gte"] = date_from.replace(tzinfo=timezone.utc)
    if date_to:
        ts_filter["$lte"] = date_to.replace(tzinfo=timezone.utc)
    if ts_filter:
        pipeline.append({"$match": {"utcTimestamp": ts_filter}})

    # Group by admissionId to get time range and document count
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
            "admissionId":   "$_id",
            "facilityId":    1,
            "first_ts":      1,
            "last_ts":       1,
            "n_docs":        1,
            "duration_min": {
                "$divide": [
                    {"$subtract": ["$last_ts", "$first_ts"]},
                    60000,   # ms → minutes
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
            logger.info(
                "[DISCOVER] %s  SKIP — %s",
                r["admissionId"], " | ".join(reason),
            )

    logger.info("[DISCOVER] %d / %d admissions eligible for inference",
                len(completed), len(results))
    return completed


# ══════════════════════════════════════════════════════════════════════════════
#  ECG EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_ecg_signal(db, admission_id: str) -> Tuple[np.ndarray, List[datetime]]:
    """
    Pull all ECG documents for this admission, ordered by packetNo,
    flatten value[0] arrays into a single continuous signal.

    Returns (signal_array, packet_timestamps)
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
        return np.array([]), []

    logger.info("[ECG] %d documents retrieved", len(docs))

    signal_chunks = []
    timestamps = []
    skipped = 0

    for doc in docs:
        raw = doc.get("value")
        # value is a nested array: value[0] contains the samples
        if isinstance(raw, list) and len(raw) > 0:
            chunk = raw[0] if isinstance(raw[0], list) else raw
            if len(chunk) > 0:
                signal_chunks.append(np.array(chunk, dtype=float))
                timestamps.append(doc.get("utcTimestamp"))
                continue
        skipped += 1

    if skipped:
        logger.warning("[ECG] Skipped %d documents with missing/empty value", skipped)

    if not signal_chunks:
        return np.array([]), []

    signal = np.concatenate(signal_chunks)
    n_samples = len(signal)
    duration_s = n_samples / FS_ECG
    logger.info(
        "[ECG] Signal assembled: %d samples  (%.1f minutes at %d Hz)",
        n_samples, duration_s / 60, FS_ECG,
    )
    return signal, timestamps


# ══════════════════════════════════════════════════════════════════════════════
#  SPO2 EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_spo2_signal(db, admission_id: str) -> np.ndarray:
    """
    Pull SpO2 pleth waveform for this admission.
    Returns a 1D array at native sampling rate (resampled to 4Hz for features).
    Returns zeros array if not available.
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
        logger.warning("[SPO2] No pleth data for %s — SpO2 features will be zeroed", admission_id)
        return np.array([])

    chunks = []
    for doc in docs:
        raw = doc.get("value", [])
        if isinstance(raw, list) and len(raw) > 0:
            chunks.append(np.array(raw, dtype=float))

    if not chunks:
        return np.array([])

    signal = np.concatenate(chunks)
    logger.info("[SPO2] %d pleth samples retrieved for %s", len(signal), admission_id)
    return signal


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
        peaks = info["ECG_R_Peaks"]
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
    """HR for each of n equal sub-segments (used by infer.py HR gate)."""
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
) -> int:
    """
    Slice ECG into 30-second segments and write a CSV that infer.py can consume.
    Returns number of segments written.
    """
    n_segs = len(ecg_signal) // SAMPLES_SEG
    if n_segs < MIN_SEGMENTS:
        logger.warning(
            "[CSV] Only %d complete segments (need ≥%d for inference). "
            "Recording is %.1f minutes — need ≥%.0f minutes.",
            n_segs, MIN_SEGMENTS,
            len(ecg_signal) / FS_ECG / 60,
            MIN_SEGMENTS * SEGMENT_LEN_S / 60,
        )
        if n_segs == 0:
            return 0

    logger.info("[CSV] Building %d × 30s segments ...", n_segs)
    rows = []

    for seg_i in range(n_segs):
        start = seg_i * SAMPLES_SEG
        seg   = ecg_signal[start: start + SAMPLES_SEG]

        # Bandpass and compute sub-segment HRs for the HR validation gate
        try:
            seg_bp = _bandpass(seg, FS_ECG)
        except Exception:
            seg_bp = seg.copy()

        sub_hrs  = _subseg_hrs(seg_bp, FS_ECG)
        mean_hr  = float(np.mean([h for h in sub_hrs if h > 0])) if any(h > 0 for h in sub_hrs) else 0.0

        # Estimate timestamp for this segment
        # Each ECG doc = 125 samples; packet_timestamps aligns to doc boundaries
        doc_idx = (start // SAMPLES_PER_ECG_DOC)
        if doc_idx < len(packet_timestamps) and packet_timestamps[doc_idx] is not None:
            ts_str = str(packet_timestamps[doc_idx])
        else:
            ts_str = ""

        row: dict = {
            "admissionId":                     admission_id,
            "facilityId":                      facility_id,
            "timestamp":                       ts_str,
            "segment_idx":                     seg_i,
            "analysis.summary.signal_quality": "acceptable",
            "analysis.heart_rate_bpm":         round(mean_hr, 1),
            "analysis.background_rhythm":      "",
        }

        # Sub-segment HR columns
        for i, hr in enumerate(sub_hrs):
            row[f"analysis.segments[{i}].morphology.hr_bpm"] = round(hr, 2)
            row[f"analysis.segments[{i}].rhythm_label"]      = ""
            row[f"analysis.segments[{i}].ectopy_label"]      = ""

        # Raw ECG samples
        for j, v in enumerate(seg):
            row[f"ecgData[{j}]"] = round(float(v), 6)

        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Ensure column order matches infer.py expectations
    fixed_cols = [
        "admissionId", "facilityId", "timestamp", "segment_idx",
        "analysis.summary.signal_quality",
        "analysis.heart_rate_bpm",
        "analysis.background_rhythm",
    ] + HR_COLS + RHYTHM_COLS + ECTOPY_COLS + ECG_COLS

    for c in fixed_cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[fixed_cols]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("[CSV] Wrote %d segments → %s", len(out_df), output_path)
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
    """
    Call the existing infer.py run_inference() function on the prepared CSV.
    Returns the summary dict for this admission or None on failure.
    """
    # Add project root to path so pipeline imports work
    project_root = str(Path(__file__).resolve().parent.parent)
    pipeline_dir = str(Path(__file__).resolve().parent.parent / "pipeline")
    for p in (project_root, pipeline_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from pipeline.infer import run_inference
    except ImportError:
        try:
            from infer import run_inference   # fallback if running from pipeline/
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

    # Read back the summary CSV that infer.py wrote
    summary_csv = os.path.join(out_dir, "infer_summary.csv")
    if not os.path.exists(summary_csv):
        logger.warning("[INFER] No summary CSV found after inference for %s", admission_id)
        return None

    summary_df = pd.read_csv(summary_csv)
    matches = summary_df[summary_df["admission_id"] == admission_id]
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
    """
    Full pipeline for one admission:
      1. Extract ECG + SpO2 from MongoDB
      2. Build segment CSV
      3. Run inference
      4. Return summary dict

    Returns summary dict or None on failure.
    """
    logger.info("=" * 60)
    logger.info("  Processing: %s", admission_id)
    logger.info("=" * 60)

    adm_out_dir = os.path.join(output_dir, admission_id)
    os.makedirs(adm_out_dir, exist_ok=True)

    # ── Step 1: Extract ECG ───────────────────────────────────────────────────
    ecg_signal, packet_timestamps = extract_ecg_signal(db, admission_id)
    if len(ecg_signal) == 0:
        logger.error("[PROCESS] No ECG signal for %s — skipping", admission_id)
        return None

    # Fill any NaNs from packet gaps
    nan_mask = ~np.isfinite(ecg_signal)
    if nan_mask.any():
        ecg_signal[nan_mask] = float(np.nanmean(ecg_signal))
        logger.warning("[PROCESS] Filled %d NaN samples in ECG", int(nan_mask.sum()))

    # ── Step 2: Extract SpO2 (best-effort) ───────────────────────────────────
    spo2_signal = extract_spo2_signal(db, admission_id)

    # ── Step 3: Build CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(adm_out_dir, f"{admission_id}_segments.csv")
    n_segs = build_segment_csv(
        ecg_signal       = ecg_signal,
        spo2_signal      = spo2_signal,
        admission_id     = admission_id,
        facility_id      = facility_id,
        packet_timestamps = packet_timestamps,
        output_path      = csv_path,
    )

    if n_segs == 0:
        logger.error("[PROCESS] No segments built for %s", admission_id)
        return None

    if dry_run:
        logger.info("[DRY RUN] CSV written to %s — skipping inference", csv_path)
        return {"admission_id": admission_id, "status": "dry_run", "n_segments": n_segs}

    # ── Step 4: Run inference ─────────────────────────────────────────────────
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
#  RESULT WRITER (stub — fill in once you know the target collection/format)
# ══════════════════════════════════════════════════════════════════════════════

def write_results_to_mongodb(db, admission_id: str, summary: Dict) -> None:
    """
    Write inference results back to MongoDB.
    Fill in the target collection name and document structure here
    once you decide where results should live.
    """
    # TODO: replace 'apnea_results' with your chosen collection name
    result_doc = {
        "admissionId":   admission_id,
        "processedAt":   datetime.now(timezone.utc),
        "ahi_proxy":     summary.get("ahi_proxy"),
        "severity":      summary.get("severity"),
        "apnea_pct":     summary.get("apnea_pct"),
        "total_segments": summary.get("total_segments"),
        "scored_segments": summary.get("scored_segments"),
        "n_apnea":       summary.get("n_apnea"),
        "duration_min":  summary.get("duration_min"),
        "model_threshold": summary.get("threshold"),
    }

    try:
        db["apnea_results"].update_one(
            {"admissionId": admission_id},
            {"$set": result_doc},
            upsert=True,
        )
        logger.info("[MONGO] Results written for %s", admission_id)
    except Exception as e:
        logger.error("[MONGO] Failed to write results for %s: %s", admission_id, e)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MongoDB → ECG extraction → Apnea inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Single admission
  python automation/mongo_infer.py --admission ADM1819906487

  # All admissions from last 24 hours
  python automation/mongo_infer.py --since 24h

  # Date range
  python automation/mongo_infer.py --from 2026-06-01 --to 2026-06-15

  # Dry run (extract CSVs only, no inference)
  python automation/mongo_infer.py --since 24h --dry-run

  # Write results back to MongoDB
  python automation/mongo_infer.py --since 24h --write-mongo
        """,
    )
    p.add_argument("--admission",  default=None,
                   help="Single admissionId to process")
    p.add_argument("--since",      default=None,
                   help="Process admissions from last N hours (e.g. 24h, 8h)")
    p.add_argument("--from",       dest="date_from", default=None,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--to",         dest="date_to",   default=None,
                   help="End date YYYY-MM-DD")
    p.add_argument("--model",      default=os.environ.get("MODEL_PATH",  "apnea_model.keras"))
    p.add_argument("--scaler",     default=os.environ.get("SCALER_PATH", "apnea_scaler.pkl"))
    p.add_argument("--threshold",  type=float, default=float(os.environ.get("THRESHOLD", "0.60")))
    p.add_argument("--out-dir",    default=os.environ.get("OUTPUT_DIR",  "infer_output"))
    p.add_argument("--dry-run",    action="store_true",
                   help="Extract CSVs but skip inference")
    p.add_argument("--write-mongo", action="store_true",
                   help="Write results back to MongoDB apnea_results collection")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate model/scaler exist unless dry-run
    if not args.dry_run:
        for path, name in [(args.model, "model"), (args.scaler, "scaler")]:
            if not os.path.exists(path):
                logger.error("[SETUP] %s not found at %s", name, path)
                logger.error("[SETUP] Train first: python pipeline/pipeline.py --fresh --save-model")
                sys.exit(1)

    db = get_db()

    # ── Resolve which admissions to process ──────────────────────────────────
    if args.admission:
        # Single admission — look up its facilityId
        doc = db.ecg_data_by_admission_id.find_one(
            {"admissionId": args.admission},
            {"facilityId": 1, "_id": 0},
        )
        if not doc:
            logger.error("[SETUP] admissionId %s not found in MongoDB", args.admission)
            sys.exit(1)
        admissions = [{
            "admissionId": args.admission,
            "facilityId":  doc.get("facilityId", ""),
        }]

    else:
        # Discover completed admissions by time range
        since_hours = None
        date_from   = None
        date_to     = None

        if args.since:
            val = args.since.replace("h", "").replace("H", "")
            since_hours = float(val)
        if args.date_from:
            date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
        if args.date_to:
            date_to = datetime.strptime(args.date_to, "%Y-%m-%d")

        if not any([since_hours, date_from, date_to]):
            logger.error("[SETUP] Specify --admission, --since, or --from/--to")
            sys.exit(1)

        admissions = find_completed_admissions(
            db, since_hours=since_hours,
            date_from=date_from, date_to=date_to,
        )

    if not admissions:
        logger.info("[MAIN] No admissions to process.")
        return

    logger.info("[MAIN] Processing %d admissions ...", len(admissions))

    # ── Process each admission ────────────────────────────────────────────────
    successes, failures = 0, 0
    for adm in admissions:
        admission_id = adm["admissionId"]
        facility_id  = adm.get("facilityId", "")

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
                if args.write_mongo and not args.dry_run:
                    write_results_to_mongodb(db, admission_id, summary)
            else:
                failures += 1

        except Exception as e:
            logger.error("[MAIN] Unhandled error for %s: %s", admission_id, e, exc_info=True)
            failures += 1

    logger.info("=" * 60)
    logger.info(
        "[DONE]  %d succeeded  |  %d failed  |  output → %s",
        successes, failures, os.path.abspath(args.out_dir),
    )


if __name__ == "__main__":
    main()
