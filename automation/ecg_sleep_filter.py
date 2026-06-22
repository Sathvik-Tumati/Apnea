"""
ecg_sleep_filter.py
===================
Detect sleep windows from continuous ECG data pulled from MongoDB.
Designed for recordings that span the full day — filters to sleep-only
segments before apnea inference.

No EEG, no accelerometer, no skin temperature required.
Works from ECG alone using:
  1. Time-of-day gate    (IST 9pm–9am)  — coarse, free
  2. HR-based scoring    (low HR = sleep candidate)
  3. SDNN-based scoring  (high SDNN = parasympathetic dominance = sleep)
  4. Minimum duration    (≥20 min contiguous to qualify as sleep)

Called from mongo_infer.py before build_segment_csv().

Usage (standalone, for debugging)
----------------------------------
  python automation/ecg_sleep_filter.py \
      --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
      --out infer_output/ADM1819906487/sleep_windows.csv \
      --plot

  python automation/ecg_sleep_filter.py \
      --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
      --stats   # just print sleep window summary, no files written
"""

import argparse
import logging
import os
from datetime import timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, medfilt

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

FS_ECG        = 125
SEGMENT_LEN_S = 30
SAMPLES_SEG   = FS_ECG * SEGMENT_LEN_S  

# Time-of-day gate: keep segments between 9pm and 9am IST
SLEEP_HOUR_START = 21  
SLEEP_HOUR_END   =  9   

# Sleep scoring thresholds
HR_PERCENTILE_THRESHOLD  = 45  
SDNN_PERCENTILE_THRESHOLD = 55  
MIN_SLEEP_EPOCHS          = 40 
MAX_WAKE_GAP_EPOCHS       =  6  


# ══════════════════════════════════════════════════════════════════════════════
#  ECG FEATURE EXTRACTION PER SEGMENT
# ══════════════════════════════════════════════════════════════════════════════

def _bandpass(sig: np.ndarray, fs: int = FS_ECG,
              lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    nyq = fs / 2.0
    hi  = min(hi, nyq - 0.1)
    lo  = min(lo, hi - 0.1)
    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _detect_r_peaks(ecg: np.ndarray, fs: int = FS_ECG) -> np.ndarray:
    """Lightweight R-peak detector — neurokit2 if available, scipy fallback."""
    if not np.isfinite(ecg).all():
        ecg = np.where(np.isfinite(ecg), ecg, 0.0)
    try:
        import neurokit2 as nk
        _, info = nk.ecg_process(ecg, sampling_rate=fs)
        peaks = np.array(info["ECG_R_Peaks"], dtype=int)
        if len(peaks) >= 2:
            rr_ms = np.diff(peaks) / fs * 1000.0
            peaks = peaks[np.concatenate([[True], (rr_ms >= 300) & (rr_ms <= 2000)])]
        return peaks
    except Exception:
        pass
    peaks, _ = find_peaks(ecg, distance=int(fs * 0.4), height=float(np.std(ecg)))
    return peaks


def _hrv_features(ecg_seg: np.ndarray, fs: int = FS_ECG) -> Dict[str, float]:
    """
    Compute mean HR and SDNN from a 30-second ECG segment.
    Returns zeros on failure (e.g. too few R-peaks).
    """
    try:
        bp     = _bandpass(ecg_seg, fs)
        peaks  = _detect_r_peaks(bp, fs)
        if len(peaks) < 3:
            return {"mean_hr": 0.0, "sdnn": 0.0, "n_peaks": len(peaks)}
        rr_ms  = np.diff(peaks) / fs * 1000.0
        mean_hr = float(60000.0 / (np.mean(rr_ms) + 1e-6))
        sdnn    = float(np.std(rr_ms))
        # Sanity gate: physiologically implausible values → treat as bad segment
        if not (20 < mean_hr < 220):
            return {"mean_hr": 0.0, "sdnn": 0.0, "n_peaks": len(peaks)}
        return {"mean_hr": mean_hr, "sdnn": sdnn, "n_peaks": len(peaks)}
    except Exception:
        return {"mean_hr": 0.0, "sdnn": 0.0, "n_peaks": 0}


# ══════════════════════════════════════════════════════════════════════════════
#  TIME-OF-DAY GATE
# ══════════════════════════════════════════════════════════════════════════════

def _in_sleep_hours(ts_utc, start_hour: int = SLEEP_HOUR_START,
                    end_hour: int = SLEEP_HOUR_END) -> bool:
    """
    Return True if timestamp (UTC) falls within the sleep window in IST.
    Sleep window wraps midnight: start_hour=21, end_hour=9 means 9pm–9am.
    """
    if ts_utc is None:
        return True   # no timestamp → don't gate out (conservative)
    try:
        ts_ist = pd.Timestamp(ts_utc, tz="UTC").tz_convert("Asia/Kolkata")
        h = ts_ist.hour
        # Window wraps midnight
        if start_hour > end_hour:
            return h >= start_hour or h < end_hour
        else:
            return start_hour <= h < end_hour
    except Exception:
        return True   # parse failure → don't gate out


# ══════════════════════════════════════════════════════════════════════════════
#  SLEEP/WAKE RUN-LENGTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fill_short_gaps(arr: np.ndarray, gap: int) -> np.ndarray:
    """Bridge runs of 0 ≤ gap epochs long that are flanked by sleep on both sides."""
    arr = arr.copy()
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            j = i
            while j < len(arr) and arr[j] == 0:
                j += 1
            if (j - i) <= gap and i > 0 and j < len(arr) \
                    and arr[i - 1] == 1 and arr[j] == 1:
                arr[i:j] = 1
            i = max(i + 1, j)
        else:
            i += 1
    return arr


def _remove_short_bouts(arr: np.ndarray, min_run: int) -> np.ndarray:
    """Remove contiguous sleep runs shorter than min_run epochs."""
    arr = arr.copy()
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            j = i
            while j < len(arr) and arr[j] == 1:
                j += 1
            if (j - i) < min_run:
                arr[i:j] = 0
            i = max(i + 1, j)
        else:
            i += 1
    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SLEEP DETECTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_sleep_segments(
    ecg_signal: np.ndarray,
    packet_timestamps: List,
    ecg_chunk_lengths: Optional[List[int]] = None,
    hr_pct:   int = HR_PERCENTILE_THRESHOLD,
    sdnn_pct: int = SDNN_PERCENTILE_THRESHOLD,
    min_epochs: int = MIN_SLEEP_EPOCHS,
    gap_epochs: int = MAX_WAKE_GAP_EPOCHS,
    use_time_gate: bool = True,
) -> pd.DataFrame:
    """
    Detect sleep segments from a continuous ECG signal.
    """
    n_segs = len(ecg_signal) // SAMPLES_SEG
    if n_segs == 0:
        logger.error("[SLEEP] Signal too short for any 30s segments")
        return pd.DataFrame()

    logger.info("[SLEEP] Extracting HRV features from %d segments ...", n_segs)

    # ── Build timestamp lookup ────────────────────────────────────────────────
    if ecg_chunk_lengths:
        _cum_samples = np.cumsum([0] + list(ecg_chunk_lengths[:-1]))
    else:
        _cum_samples = np.array([], dtype=np.int64)

    def _seg_timestamp(seg_i: int):
        if not packet_timestamps or len(_cum_samples) == 0:
            return None
        sample_idx = seg_i * SAMPLES_SEG
        doc_idx = max(0, int(np.searchsorted(_cum_samples, sample_idx, side="right")) - 1)
        if doc_idx < len(packet_timestamps):
            return packet_timestamps[doc_idx]
        return None

    # ── Per-segment HRV features ──────────────────────────────────────────────
    rows = []
    for seg_i in range(n_segs):
        start = seg_i * SAMPLES_SEG
        seg   = ecg_signal[start: start + SAMPLES_SEG]
        feats = _hrv_features(seg)
        ts    = _seg_timestamp(seg_i)

        # Convert UTC → IST for display and time-gating
        ts_ist = None
        hour_ist = -1
        try:
            if ts is not None:
                ts_ist = (pd.Timestamp(ts).tz_localize("UTC") if ts.tzinfo is None
                          else pd.Timestamp(ts).tz_convert("Asia/Kolkata"))
                hour_ist = ts_ist.hour
        except Exception:
            pass

        rows.append({
            "segment_idx":   seg_i,
            "timestamp_utc": ts,
            "timestamp_ist": str(ts_ist) if ts_ist else "",
            "hour_ist":      hour_ist,
            **feats,
        })

        if (seg_i + 1) % 200 == 0:
            logger.info("[SLEEP] HRV features: %d / %d", seg_i + 1, n_segs)

    df = pd.DataFrame(rows)

    # ── Time-of-day gate ──────────────────────────────────────────────────────
    # BUG FIX: Compute in_sleep_hours FIRST, before using it
    if use_time_gate:
        df["in_sleep_hours"] = df["timestamp_utc"].apply(_in_sleep_hours)
        n_gated = int((~df["in_sleep_hours"]).sum())
        logger.info(
            "[SLEEP] Time gate (IST %d:00–%d:00): %d / %d segments in window, "
            "%d excluded",
            SLEEP_HOUR_START, SLEEP_HOUR_END,
            int(df["in_sleep_hours"].sum()), n_segs, n_gated,
        )
    else:
        df["in_sleep_hours"] = True
        logger.info("[SLEEP] Time gate disabled — all %d segments considered", n_segs)

    # ── HR and SDNN scoring ───────────────────────────────────────────────────
    # Only compute thresholds from nighttime segments to avoid daytime
    # HR/SDNN pulling the percentiles in the wrong direction.
    night_mask = df["in_sleep_hours"] & (df["mean_hr"] > 0)
    night_df   = df[night_mask]

    if len(night_df) < 20:
        # Not enough nighttime data — fall back to all valid segments
        logger.warning(
            "[SLEEP] Only %d nighttime segments with valid HR — "
            "computing thresholds from all segments", len(night_df),
        )
        valid_mask = df["mean_hr"] > 0
        night_df   = df[valid_mask]

    if len(night_df) == 0:
        logger.error("[SLEEP] No valid HR segments found")
        df["hr_sleep"]    = False
        df["sdnn_sleep"]  = False
        df["sleep_score"] = 0.0
        df["is_sleep"]    = 0
        df["sleep_window_id"] = -1
        return df

    hr_thresh   = float(np.percentile(night_df["mean_hr"], hr_pct))
    sdnn_thresh = float(np.percentile(night_df["sdnn"],    sdnn_pct))

    logger.info(
        "[SLEEP] Thresholds — HR < %.1f bpm (p%d),  SDNN > %.1f ms (p%d)",
        hr_thresh, hr_pct, sdnn_thresh, sdnn_pct,
    )

    # A segment must have valid HR to score; if HR=0 (bad segment) → not sleep
    df["hr_sleep"]   = (df["mean_hr"] > 0) & (df["mean_hr"]  < hr_thresh)
    df["sdnn_sleep"] = (df["mean_hr"] > 0) & (df["sdnn"]     > sdnn_thresh)

    # Sleep score: 0, 0.5, or 1.0
    #   both criteria + in time window  → 1.0
    #   one criterion  + in time window → 0.5
    #   time gate only                  → 0.25 (unlikely to be sleep but possible)
    #   nothing                         → 0.0
    score = np.zeros(len(df))
    score += 0.5  * df["hr_sleep"].values.astype(float)
    score += 0.5  * df["sdnn_sleep"].values.astype(float)
    score *= df["in_sleep_hours"].values.astype(float)   # zero out daytime

    # 5-epoch median filter to smooth out brief artifacts
    score_smooth = medfilt(score, kernel_size=5) if len(score) >= 5 else score
    df["sleep_score"] = score_smooth

    # ── Threshold and enforce minimum duration ────────────────────────────────
    # Require score ≥ 0.5 (at least one HRV criterion met, within time window)
    is_sleep = (score_smooth >= 0.50).astype(int)
    
    # BUG FIX: Apply hard clamp HERE, after is_sleep exists and before run-length processing
    if use_time_gate:
        is_sleep[~df["in_sleep_hours"].values] = 0

    is_sleep = _fill_short_gaps(is_sleep, gap=gap_epochs)
    is_sleep = _remove_short_bouts(is_sleep, min_run=min_epochs)
    df["is_sleep"] = is_sleep

    # ── Assign sleep window IDs ───────────────────────────────────────────────
    window_id = np.full(len(df), -1, dtype=int)
    wid, in_sleep_flag = 0, False
    for i, s in enumerate(is_sleep):
        if s and not in_sleep_flag:
            in_sleep_flag = True
        if in_sleep_flag:
            if s:
                window_id[i] = wid
            else:
                wid += 1
                in_sleep_flag = False
    df["sleep_window_id"] = window_id

    # ── Summary ───────────────────────────────────────────────────────────────
    n_sleep   = int(is_sleep.sum())
    n_windows = int(window_id.max()) + 1 if window_id.max() >= 0 else 0
    logger.info(
        "[SLEEP] Result: %d / %d segments classified as sleep (%.0f%%)  "
        "= %.1f hours  across %d window(s)",
        n_sleep, n_segs, 100 * n_sleep / max(n_segs, 1),
        n_sleep * SEGMENT_LEN_S / 3600, n_windows,
    )
    for wid_i in range(n_windows):
        mask = window_id == wid_i
        wdf  = df[mask]
        t0   = wdf["timestamp_ist"].iloc[0]  if len(wdf) else ""
        t1   = wdf["timestamp_ist"].iloc[-1] if len(wdf) else ""
        logger.info(
            "[SLEEP]   Window %d: %d segments (%.0f min)  %s → %s",
            wid_i, int(mask.sum()), int(mask.sum()) * SEGMENT_LEN_S / 60, t0, t1,
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: filter a segment CSV to sleep-only rows
# ══════════════════════════════════════════════════════════════════════════════

def filter_to_sleep(
    segment_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given a segment CSV and the sleep detection DataFrame,
    return only the rows where is_sleep == 1.

    segment_df must have a 'segment_idx' column.
    """
    sleep_idxs = set(sleep_df.loc[sleep_df["is_sleep"] == 1, "segment_idx"].tolist())
    filtered   = segment_df[segment_df["segment_idx"].isin(sleep_idxs)].reset_index(drop=True)
    logger.info(
        "[SLEEP] filter_to_sleep: %d → %d segments (%.0f%% kept)",
        len(segment_df), len(filtered),
        100 * len(filtered) / max(len(segment_df), 1),
    )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_sleep_detection(sleep_df: pd.DataFrame, out_path: str) -> None:
    """Save a 3-panel plot: HR, SDNN, sleep score with shaded sleep windows."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[SLEEP] matplotlib not installed — skipping plot")
        return

    t = sleep_df["segment_idx"].values * SEGMENT_LEN_S / 3600  # hours

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("ECG-based Sleep Detection", fontsize=13, fontweight="bold")

    def _shade(ax):
        is_sl = sleep_df["is_sleep"].values
        in_s  = False
        for i, s in enumerate(is_sl):
            if s and not in_s:
                x0 = t[i]; in_s = True
            if not s and in_s:
                ax.axvspan(x0, t[i], alpha=0.18, color="steelblue")
                in_s = False
        if in_s:
            ax.axvspan(x0, t[-1], alpha=0.18, color="steelblue")

    # Panel 1: HR
    ax = axes[0]
    valid = sleep_df["mean_hr"] > 0
    ax.plot(t[valid], sleep_df.loc[valid, "mean_hr"], color="crimson",
            linewidth=0.7, alpha=0.85)
    ax.set_ylabel("Mean HR (bpm)", fontsize=9)
    ax.grid(True, alpha=0.3)
    _shade(ax)

    # Panel 2: SDNN
    ax = axes[1]
    ax.plot(t[valid], sleep_df.loc[valid, "sdnn"], color="darkorange",
            linewidth=0.7, alpha=0.85)
    ax.set_ylabel("SDNN (ms)", fontsize=9)
    ax.grid(True, alpha=0.3)
    _shade(ax)

    # Panel 3: Time gate
    ax = axes[2]
    ax.fill_between(t, sleep_df["in_sleep_hours"].astype(int),
                    step="pre", alpha=0.5, color="seagreen")
    ax.set_ylabel("In sleep\nhours (IST)", fontsize=9)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, alpha=0.3)

    # Panel 4: Sleep score + final classification
    ax = axes[3]
    ax.plot(t, sleep_df["sleep_score"], color="steelblue", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="threshold=0.5")
    ax.fill_between(t, sleep_df["is_sleep"].astype(float) * 0.5,
                    alpha=0.3, color="steelblue", label="sleep")
    ax.set_ylabel("Sleep score", fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _shade(ax)

    axes[-1].set_xlabel("Time (hours from recording start)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    logger.info("[SLEEP] Plot saved → %s", out_path)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI (standalone debugging)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="ECG-only sleep detection for MongoDB wearable data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv",     required=True,
                   help="Segment CSV produced by mongo_infer.py --dry-run")
    p.add_argument("--out",     default=None,
                   help="Output sleep windows CSV (default: <csv>_sleep.csv)")
    p.add_argument("--plot",    action="store_true",
                   help="Save a diagnostic PNG alongside the output CSV")
    p.add_argument("--stats",   action="store_true",
                   help="Print summary only, write no files")
    p.add_argument("--no-time-gate", action="store_true",
                   help="Disable IST 9pm–9am filter (useful for short test recordings)")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    args = _parse_args()

    logger.info("[CLI] Loading segment CSV: %s", args.csv)
    seg_df = pd.read_csv(args.csv, low_memory=False)
    n_segs = len(seg_df)
    logger.info("[CLI] %d segments loaded", n_segs)

    # Reconstruct ECG signal from the CSV columns
    ecg_cols = [f"ecgData[{i}]" for i in range(SAMPLES_SEG)]
    missing  = [c for c in ecg_cols if c not in seg_df.columns]
    if missing:
        logger.error("[CLI] %d ecgData[] columns missing from CSV", len(missing))
        return

    logger.info("[CLI] Reconstructing ECG signal from CSV (%d segments × %d samples) ...",
                n_segs, SAMPLES_SEG)
    ecg_signal = seg_df[ecg_cols].values.astype(float).ravel()

    # Timestamps from the CSV
    timestamps = (
        pd.to_datetime(seg_df["timestamp"], utc=True, errors="coerce").tolist()
        if "timestamp" in seg_df.columns else []
    )

    sleep_df = detect_sleep_segments(
        ecg_signal         = ecg_signal,
        packet_timestamps  = timestamps,
        ecg_chunk_lengths  = [SAMPLES_SEG] * n_segs,   # 1 seg = 1 "doc" in CLI mode
        use_time_gate      = not args.no_time_gate,
    )

    if args.stats:
        n_sleep = int(sleep_df["is_sleep"].sum())
        print(f"\nSleep segments : {n_sleep} / {n_segs}  "
              f"({100*n_sleep/max(n_segs,1):.0f}%)  "
              f"= {n_sleep*SEGMENT_LEN_S/3600:.1f}h")
        n_wins = int(sleep_df["sleep_window_id"].max()) + 1 \
                 if sleep_df["sleep_window_id"].max() >= 0 else 0
        print(f"Sleep windows  : {n_wins}")
        return

    out_path = args.out or args.csv.replace(".csv", "_sleep_windows.csv")
    sleep_df.to_csv(out_path, index=False)
    logger.info("[CLI] Sleep windows saved → %s", out_path)

    if args.plot:
        plot_path = out_path.replace(".csv", "_plot.png")
        plot_sleep_detection(sleep_df, plot_path)


if __name__ == "__main__":
    main()