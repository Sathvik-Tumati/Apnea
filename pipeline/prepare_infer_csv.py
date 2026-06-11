"""
prepare_infer_csv.py
====================
Converts raw wearable ECG packet CSV into the flat segment format
expected by infer.py.

Input format (your MongoDB export)
-----------------------------------
Columns: utcTimestamp, admissionId, value, facilityId, packetNo
  - One row per ECG sample (float, millivolts)
  - Every 100 rows = one packet; metadata columns only filled on first row
  - Timestamps are unreliable (network jitter) — packetNo used for ordering
  - Sampling rate: 125 Hz (100 samples / 0.8s per packet)

Output format (infer.py compatible)
-------------------------------------
Columns: admissionId, timestamp, analysis.summary.signal_quality,
         analysis.heart_rate_bpm, analysis.segments[N].morphology.hr_bpm (×6),
         analysis.segments[N].rhythm_label (×6),
         analysis.segments[N].ectopy_label (×6),
         ecgData[0] … ecgData[3749]
  - One row per 30-second segment (3750 samples at 125 Hz)
  - Pre-computed HR columns filled from scipy R-peak detection
  - signal_quality always 'acceptable' (no upstream QA available)

Usage
-----
# Single file
python prepare_infer_csv.py --input recording.csv --output ready_for_infer.csv

# Then run inference
python infer.py --csv ready_for_infer.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
FS            = 125          # confirmed sampling rate
SAMPLES_PKT   = 100          # samples per packet
SAMPLES_SEG   = 3750         # samples per 30-second segment (125 Hz × 30 s)
SEGMENT_LEN_S = 30
N_SUBSEG      = 6            # number of 5-second sub-segments for HR columns
SUBSEG_LEN    = SAMPLES_SEG // N_SUBSEG   # 625 samples per 5-second sub-segment

ECG_DATA_COLS = [f"ecgData[{i}]" for i in range(SAMPLES_SEG)]
HR_SUBSEG_COLS = [f"analysis.segments[{i}].morphology.hr_bpm" for i in range(N_SUBSEG)]
RHYTHM_COLS    = [f"analysis.segments[{i}].rhythm_label"      for i in range(N_SUBSEG)]
ECTOPY_COLS    = [f"analysis.segments[{i}].ectopy_label"       for i in range(N_SUBSEG)]


# ── Signal helpers ────────────────────────────────────────────────────────────

def _bandpass(sig: np.ndarray, fs: int = FS,
              lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    nyq = fs / 2.0
    hi  = min(hi, nyq - 0.1)
    lo  = min(lo, hi - 0.1)
    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _detect_r_peaks(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """neurokit2 if available, otherwise robust scipy fallback."""
    try:
        import neurokit2 as nk
        _, info = nk.ecg_process(ecg, sampling_rate=fs)
        return info["ECG_R_Peaks"]
    except Exception:
        pass
    threshold = float(np.mean(ecg) + 0.3 * np.std(ecg))
    peaks, _  = find_peaks(ecg, distance=int(fs * 0.4), height=threshold)
    if len(peaks) < 2:
        peaks, _ = find_peaks(ecg, distance=int(fs * 0.4), height=float(np.mean(ecg)))
    return peaks


def _hr_from_peaks(r_peaks: np.ndarray, fs: int = FS) -> float:
    """Mean HR in BPM from R-peak indices."""
    if len(r_peaks) < 2:
        return 0.0
    rr_ms = np.diff(r_peaks) / fs * 1000.0
    return float(60000.0 / (np.mean(rr_ms) + 1e-6))


def _subseg_hr(seg: np.ndarray, fs: int = FS) -> List[float]:
    """
    Compute HR for each of 6 non-overlapping 5-second sub-segments.
    This fills the analysis.segments[N].morphology.hr_bpm columns that
    infer.py uses for the HR validation gate.
    """
    hrs = []
    for i in range(N_SUBSEG):
        chunk = seg[i * SUBSEG_LEN:(i + 1) * SUBSEG_LEN]
        try:
            bp    = _bandpass(chunk, fs)
            peaks = _detect_r_peaks(bp, fs)
            hrs.append(_hr_from_peaks(peaks, fs))
        except Exception:
            hrs.append(0.0)
    return hrs


# ── Core conversion ───────────────────────────────────────────────────────────

def convert_packet_csv(
    input_path: str,
    output_path: str,
    min_segments: int = 1,
) -> int:
    """
    Read a raw packet CSV and write a flat segment CSV for infer.py.

    Returns the number of segments written.
    """
    logger.info("[PREP] Reading %s ...", input_path)
    df = pd.read_csv(input_path, low_memory=False)

    # ── Validate columns ──────────────────────────────────────────────────────
    required = {"value", "packetNo"}
    missing  = required - set(df.columns)
    if missing:
        logger.error("[PREP] Missing required columns: %s", missing)
        logger.error("[PREP] Found columns: %s", df.columns.tolist())
        return 0

    logger.info("[PREP] %d rows loaded", len(df))

    # ── Extract metadata from packet header rows ──────────────────────────────
    meta_rows   = df[df["packetNo"].notna()].copy()
    admission_id = str(meta_rows["admissionId"].dropna().iloc[0]) \
                   if "admissionId" in df.columns else "UNKNOWN"
    facility_id  = str(meta_rows["facilityId"].dropna().iloc[0]) \
                   if "facilityId" in df.columns else ""

    logger.info("[PREP] admissionId=%s  facilityId=%s", admission_id, facility_id)

    # ── Reconstruct ordered signal from packets ───────────────────────────────
    # Sort packets by packetNo (timestamps are unreliable due to network jitter).
    # Each packet header row is at index i*SAMPLES_PKT; the following
    # SAMPLES_PKT-1 rows are its samples.
    #
    # Strategy: sort the header rows by packetNo, then rebuild the full
    # sample array in that order.

    # Find the row index of each packet's first sample
    pkt_header_idx = meta_rows.sort_values("packetNo").index.tolist()

    samples_ordered: List[float] = []
    timestamps_ordered: List[str] = []

    for idx in pkt_header_idx:
        # Collect SAMPLES_PKT rows starting from this header
        pkt_rows = df.iloc[idx:idx + SAMPLES_PKT]
        vals     = pkt_rows["value"].values.astype(float)

        # Fill any NaNs within the packet by linear interpolation
        if np.isnan(vals).any():
            s = pd.Series(vals)
            vals = s.interpolate(method="linear").bfill().ffill().values

        samples_ordered.extend(vals.tolist())

        # Grab timestamp from the header row
        ts = df.iloc[idx].get("utcTimestamp", "")
        timestamps_ordered.append(str(ts) if pd.notna(ts) else "")

    signal = np.array(samples_ordered, dtype=float)
    n_total = len(signal)
    logger.info("[PREP] Reconstructed signal: %d samples  (%.1f s at %d Hz)",
                n_total, n_total / FS, FS)

    # ── Slice into 30-second segments ─────────────────────────────────────────
    n_segs = n_total // SAMPLES_SEG
    if n_segs < min_segments:
        logger.warning(
            "[PREP] Only %d complete 30s segments available (need %d). "
            "Recording is %.1f seconds — need at least %.0f seconds.",
            n_segs, min_segments, n_total / FS, min_segments * SEGMENT_LEN_S)
        if n_segs == 0:
            logger.error("[PREP] No complete segments — cannot continue.")
            return 0

    logger.info("[PREP] Slicing into %d × 30s segments ...", n_segs)

    out_rows = []
    for seg_i in range(n_segs):
        seg = signal[seg_i * SAMPLES_SEG:(seg_i + 1) * SAMPLES_SEG]

        # Bandpass then compute sub-segment HR values for the HR gate
        try:
            seg_bp = _bandpass(seg)
        except Exception:
            seg_bp = seg.copy()

        sub_hrs  = _subseg_hr(seg_bp)
        mean_hr  = float(np.mean([h for h in sub_hrs if h > 0])) if any(h > 0 for h in sub_hrs) else 0.0

        # Timestamp for this segment: use the packet timestamp closest to its start
        pkt_start = seg_i * SAMPLES_SEG // SAMPLES_PKT
        ts_str    = timestamps_ordered[pkt_start] if pkt_start < len(timestamps_ordered) else ""

        # Build output row
        row: dict = {
            "admissionId":                          admission_id,
            "facilityId":                           facility_id,
            "timestamp":                            ts_str,
            "segment_idx":                          seg_i,
            "analysis.summary.signal_quality":      "acceptable",
            "analysis.heart_rate_bpm":              round(mean_hr, 1),
            "analysis.background_rhythm":           "",
        }

        # Sub-segment HR columns (what the HR validation gate compares against)
        for i, hr in enumerate(sub_hrs):
            row[f"analysis.segments[{i}].morphology.hr_bpm"] = round(hr, 2)
            row[f"analysis.segments[{i}].rhythm_label"]       = ""
            row[f"analysis.segments[{i}].ectopy_label"]       = ""

        # Raw ECG samples — the main payload
        for j, v in enumerate(seg):
            row[f"ecgData[{j}]"] = round(float(v), 6)

        out_rows.append(row)

        if (seg_i + 1) % 10 == 0 or seg_i == n_segs - 1:
            logger.info("[PREP] Processed %d / %d segments  mean_hr=%.1f",
                        seg_i + 1, n_segs, mean_hr)

    out_df = pd.DataFrame(out_rows)

    # Ensure column order matches what infer.py expects
    fixed_cols = [
        "admissionId", "facilityId", "timestamp", "segment_idx",
        "analysis.summary.signal_quality",
        "analysis.heart_rate_bpm",
        "analysis.background_rhythm",
    ] + HR_SUBSEG_COLS + RHYTHM_COLS + ECTOPY_COLS + ECG_DATA_COLS

    for c in fixed_cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[fixed_cols]

    out_df.to_csv(output_path, index=False)
    logger.info("[PREP] Wrote %d segments → %s", len(out_df), output_path)

    if n_segs < 11:
        logger.warning(
            "[PREP] Only %d segments — LSTM needs ≥11 for any predictions "
            "(10-segment lookback window + 1 target). "
            "This recording is %.1f minutes; need ≥5.5 minutes.",
            n_segs, n_total / FS / 60)

    return len(out_df)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw wearable packet CSV → infer.py segment CSV")
    p.add_argument("--input",  "-i", required=True,
                   help="Path to raw packet CSV (MongoDB export)")
    p.add_argument("--output", "-o", default=None,
                   help="Output path (default: <input>_prepared.csv)")
    p.add_argument("--min-segments", type=int, default=1,
                   help="Abort if fewer than this many 30s segments (default: 1)")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    out    = args.output or str(Path(args.input).with_suffix("")) + "_prepared.csv"
    n_segs = convert_packet_csv(args.input, out, args.min_segments)
    if n_segs == 0:
        sys.exit(1)
    logger.info("[PREP] Done. Run inference with:")
    logger.info("       python infer.py --csv %s", out)


if __name__ == "__main__":
    main()