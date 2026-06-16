"""
sleep_filter.py
===============
Detect sleep windows from wearable signals (no EEG required) and filter
converted EDF segments to sleep-only before pipeline inference.

Sleep detection uses a multi-signal heuristic classifier:
  - Activity level (ACTIVITY channel or ACCEL magnitude)
  - Posture (lying down: POSTURE ∈ {1,2,3,4} — not upright)
  - Heart rate (lower during sleep)
  - Skin temperature (rises ~0.5°C after sleep onset)
  - Sustained inactivity duration (must be >10 min to count as sleep)

Pipeline
--------
  Step 1:  sleep_filter.py --detect        → detects sleep windows, saves sleep_windows.csv
  Step 2:  sleep_filter.py --filter        → filters converted CSVs to sleep segments only
  Step 3:  edf_test_loader.py --data ...   → run inference on sleep-only segments

Usage
-----
  # Detect sleep windows from raw EDF (fast — only reads lightweight channels)
  python sleep_filter.py --detect \\
      --input 'AEUNI_...edf' \\
      --out-dir ./converted/

  # Filter existing converted CSVs using detected sleep windows
  python sleep_filter.py --filter \\
      --csv-dir ./converted/ \\
      --windows ./converted/AEUNI_sleep_windows.csv \\
      --out-dir ./converted/sleep_only/

  # Do both in one call
  python sleep_filter.py --detect --filter \\
      --input 'AEUNI_...edf' \\
      --csv-dir ./converted/ \\
      --out-dir ./converted/

  # Visualise the sleep detection result (saves a PNG)
  python sleep_filter.py --detect --plot \\
      --input 'AEUNI_...edf' \\
      --out-dir ./converted/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import medfilt
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pyedflib
    HAS_PYEDF = True
except ImportError:
    HAS_PYEDF = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

if not HAS_PYEDF and not HAS_MNE:
    print("ERROR: pip install pyedflib")
    sys.exit(1)


SEG_LEN = 30   # seconds, must match pipeline.py


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT EDF CHANNEL READER
# ══════════════════════════════════════════════════════════════════════════════

def read_channel(edf_path: str, channel: str,
                 max_duration_s: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Read a single channel from EDF/BDF. Returns (signal, fs)."""
    if HAS_PYEDF:
        f = pyedflib.EdfReader(edf_path)
        labels = f.getSignalLabels()
        if channel not in labels:
            f.close()
            raise KeyError(f"Channel '{channel}' not found. Available: {labels}")
        idx = labels.index(channel)
        fs  = float(f.getSampleFrequency(idx))
        n   = int(f.getNSamples()[idx])
        if max_duration_s:
            n = min(n, int(max_duration_s * fs))
        sig = f.readSignal(idx, start=0, n=n, digital=False).astype(np.float32)
        f.close()
        return sig, fs
    else:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        if channel not in raw.ch_names:
            raise KeyError(f"Channel '{channel}' not found.")
        fs = float(raw.info["sfreq"])
        data, _ = raw[channel]
        sig = data[0].astype(np.float32)
        if max_duration_s:
            sig = sig[:int(max_duration_s * fs)]
        return sig, fs


def get_channel_list(edf_path: str) -> List[str]:
    if HAS_PYEDF:
        f = pyedflib.EdfReader(edf_path)
        labels = f.getSignalLabels()
        f.close()
        return labels
    else:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        return raw.ch_names


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (1 value per 30-second epoch)
# ══════════════════════════════════════════════════════════════════════════════

def _epoch_signal(sig: np.ndarray, fs: float, seg_len: int = SEG_LEN) -> np.ndarray:
    """Downsample signal to one value per epoch by taking mean."""
    spe = int(fs * seg_len)
    n   = len(sig) // spe
    return np.array([np.mean(sig[i*spe:(i+1)*spe]) for i in range(n)], dtype=float)


def _epoch_std(sig: np.ndarray, fs: float, seg_len: int = SEG_LEN) -> np.ndarray:
    """Std per epoch (activity measure)."""
    spe = int(fs * seg_len)
    n   = len(sig) // spe
    return np.array([np.std(sig[i*spe:(i+1)*spe]) for i in range(n)], dtype=float)


def extract_sleep_features(
    edf_path: str,
    max_duration_s: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract per-30s-epoch features from lightweight wearable channels.
    Only reads channels that exist in the file.
    """
    channels = get_channel_list(edf_path)
    print(f"  Available channels: {channels}")

    features: Dict[str, np.ndarray] = {}
    n_epochs = None

    # ── Activity from ACTIVITY channel ───────────────────────────────────────
    if "ACTIVITY" in channels:
        sig, fs = read_channel(edf_path, "ACTIVITY", max_duration_s)
        sig = np.clip(sig, -2, 10)   # filter sentinel values (-2 = invalid)
        activity = _epoch_signal(sig, fs)
        n_epochs  = len(activity)
        features["activity_mean"] = activity
        features["activity_std"]  = _epoch_std(sig, fs)
        print(f"  ACTIVITY: {len(activity)} epochs")

    # ── Activity from ACCEL magnitude (more reliable) ─────────────────────────
    accel_channels = [c for c in ["ACCEL_X", "ACCEL_Y", "ACCEL_Z"] if c in channels]
    if len(accel_channels) == 3:
        ax, fs_a = read_channel(edf_path, "ACCEL_X", max_duration_s)
        ay, _    = read_channel(edf_path, "ACCEL_Y", max_duration_s)
        az, _    = read_channel(edf_path, "ACCEL_Z", max_duration_s)
        n        = min(len(ax), len(ay), len(az))
        mag      = np.sqrt(ax[:n]**2 + ay[:n]**2 + az[:n]**2)
        # Remove gravity component (high-pass: subtract slow trend)
        if HAS_SCIPY:
            w = int(fs_a * 2)
            trend = uniform_filter1d(mag.astype(float), size=w)
            dynamic = np.abs(mag - trend)
        else:
            dynamic = np.abs(np.diff(mag, prepend=mag[0]))
        accel_activity = _epoch_signal(dynamic, fs_a)
        accel_std      = _epoch_std(dynamic, fs_a)
        if n_epochs is None:
            n_epochs = len(accel_activity)
        else:
            accel_activity = accel_activity[:n_epochs]
            accel_std      = accel_std[:n_epochs]
        features["accel_activity"] = accel_activity
        features["accel_std"]      = accel_std
        print(f"  ACCEL: {len(accel_activity)} epochs")

    # ── Posture ───────────────────────────────────────────────────────────────
    if "POSTURE" in channels:
        sig, fs = read_channel(edf_path, "POSTURE", max_duration_s)
        # Values: -2=invalid, 0=upright, 1=supine, 2=left, 3=right, 4=prone
        # Lying = posture in {1, 2, 3, 4}
        lying    = ((sig >= 1) & (sig <= 4)).astype(float)
        posture  = _epoch_signal(lying, fs)
        if n_epochs is None:
            n_epochs = len(posture)
        features["posture_lying_frac"] = posture[:n_epochs]
        print(f"  POSTURE: {len(posture)} epochs")

    # ── Heart rate ────────────────────────────────────────────────────────────
    if "HR" in channels:
        sig, fs = read_channel(edf_path, "HR", max_duration_s)
        # Filter invalid HR values (device uses -3 or 0 as sentinel)
        sig = np.where((sig > 20) & (sig < 220), sig, np.nan)
        # Forward-fill NaNs
        sig = pd.Series(sig).ffill().bfill().values.astype(float)
        hr_mean = _epoch_signal(sig, fs)
        if n_epochs is None:
            n_epochs = len(hr_mean)
        features["hr_mean"] = hr_mean[:n_epochs]
        print(f"  HR: {len(hr_mean)} epochs")

    # ── Skin temperature ──────────────────────────────────────────────────────
    if "SKINTEMP" in channels:
        sig, fs = read_channel(edf_path, "SKINTEMP", max_duration_s)
        # Filter invalid values (0 or negative often means sensor off)
        sig = np.where(sig > 20, sig, np.nan)
        sig = pd.Series(sig).ffill().bfill().values.astype(float)
        skin_temp = _epoch_signal(sig, fs)
        if n_epochs is None:
            n_epochs = len(skin_temp)
        features["skin_temp"] = skin_temp[:n_epochs]
        print(f"  SKINTEMP: {len(skin_temp)} epochs")

    if n_epochs is None or n_epochs == 0:
        raise RuntimeError("No usable channels found for sleep detection")

    # Align all arrays to same length
    for k in features:
        features[k] = features[k][:n_epochs]

    df = pd.DataFrame(features)
    df.insert(0, "segment_idx", np.arange(n_epochs))
    df.insert(1, "onset_s",     np.arange(n_epochs) * SEG_LEN)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SLEEP / WAKE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_sleep(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-signal sleep/wake scorer.
    Returns feat_df with added columns: sleep_score, is_sleep, sleep_window_id
    
    Scoring logic (each criterion adds to a 0-4 score):
      +1  low accel activity (< p25 of recording)
      +1  lying posture (≥ 70% of epoch)
      +1  HR below recording median
      +1  skin temp above recording 40th percentile (distal vasodilation)
    
    Sleep epoch = score ≥ 2 (majority of criteria met).
    Sleep window = contiguous run of sleep epochs ≥ 20 minutes.
    """
    df = feat_df.copy()
    n  = len(df)
    score = np.zeros(n, dtype=float)

    # ── Criterion 1: low movement ─────────────────────────────────────────────
    if "accel_activity" in df.columns:
        thresh = np.nanpercentile(df["accel_activity"], 25)
        score += (df["accel_activity"] < thresh).astype(float)
    elif "activity_mean" in df.columns:
        # ACTIVITY channel: 0=rest, 1=sedentary, 2=light, 3=moderate, 4=active
        score += (df["activity_mean"] < 1.0).astype(float)

    # ── Criterion 2: lying posture ────────────────────────────────────────────
    if "posture_lying_frac" in df.columns:
        score += (df["posture_lying_frac"] >= 0.70).astype(float)
    else:
        # No posture channel: add 0.5 as neutral (don't penalise or reward)
        score += 0.5

    # ── Criterion 3: low heart rate ───────────────────────────────────────────
    if "hr_mean" in df.columns:
        hr_median = np.nanmedian(df["hr_mean"])
        score += (df["hr_mean"] < hr_median).astype(float)

    # ── Criterion 4: elevated skin temperature ────────────────────────────────
    if "skin_temp" in df.columns:
        temp_p40 = np.nanpercentile(df["skin_temp"], 40)
        score += (df["skin_temp"] > temp_p40).astype(float)

    # Normalise score to max achievable
    max_score = sum([
        1.0 if "accel_activity" in df.columns or "activity_mean" in df.columns else 0,
        1.0 if "posture_lying_frac" in df.columns else 0.5,
        1.0 if "hr_mean" in df.columns else 0,
        1.0 if "skin_temp" in df.columns else 0,
    ])
    score_norm = score / max(max_score, 1.0)   # 0–1

    # ── Smoothing: 5-epoch (~2.5 min) median filter to remove brief wake bursts
    if HAS_SCIPY and len(score_norm) >= 5:
        score_smooth = medfilt(score_norm, kernel_size=5)
    else:
        score_smooth = score_norm

    df["sleep_score"] = score_smooth

    # ── Threshold: sleep if score ≥ 0.5 (≥ half the criteria met) ────────────
    df["is_sleep_raw"] = (score_smooth >= 0.50).astype(int)

    # ── Minimum duration gate: must have ≥ 20 consecutive sleep epochs (~10 min)
    MIN_SLEEP_EPOCHS = 20   # 20 × 30s = 10 minutes

    is_sleep = df["is_sleep_raw"].values.copy()

    # Fill short wake gaps inside sleep (≤ 3 epochs = 1.5 min)
    is_sleep = _fill_short_gaps(is_sleep, gap_fill=3)

    # Remove short sleep bouts
    is_sleep = _remove_short_runs(is_sleep, min_run=MIN_SLEEP_EPOCHS)

    df["is_sleep"] = is_sleep

    # ── Assign contiguous sleep window IDs ────────────────────────────────────
    window_id = np.full(n, -1, dtype=int)
    wid = 0
    in_sleep = False
    for i in range(n):
        if is_sleep[i] and not in_sleep:
            in_sleep = True
        if in_sleep:
            if is_sleep[i]:
                window_id[i] = wid
            else:
                wid += 1
                in_sleep = False
    df["sleep_window_id"] = window_id

    n_sleep  = int(is_sleep.sum())
    n_windows = int(window_id.max()) + 1 if window_id.max() >= 0 else 0
    print(f"\n  Sleep detection summary:")
    print(f"    Total epochs   : {n}")
    print(f"    Sleep epochs   : {n_sleep} ({100*n_sleep/max(n,1):.0f}%)  "
          f"= {n_sleep*SEG_LEN/3600:.1f}h")
    print(f"    Sleep windows  : {n_windows}")
    for wid_i in range(n_windows):
        mask  = window_id == wid_i
        start = int(df.loc[mask, "onset_s"].min())
        end   = int(df.loc[mask, "onset_s"].max()) + SEG_LEN
        dur   = end - start
        n_ap  = int(df.loc[mask, "true_label"].sum()) if "true_label" in df.columns else "?"
        print(f"    Window {wid_i}: {start//3600:.1f}h–{end//3600:.1f}h  "
              f"({dur//60}min)  segments={int(mask.sum())}  apnea={n_ap}")

    return df


def _fill_short_gaps(arr: np.ndarray, gap_fill: int) -> np.ndarray:
    """Fill runs of 0 shorter than gap_fill with 1 (bridge short wake epochs)."""
    arr = arr.copy()
    i = 0
    while i < len(arr):
        if arr[i] == 0:
            j = i
            while j < len(arr) and arr[j] == 0:
                j += 1
            if j - i <= gap_fill:
                # Only fill if flanked by sleep on both sides
                if i > 0 and j < len(arr) and arr[i-1] == 1 and arr[j] == 1:
                    arr[i:j] = 1
            i = j
        else:
            i += 1
    return arr


def _remove_short_runs(arr: np.ndarray, min_run: int) -> np.ndarray:
    """Remove runs of 1 shorter than min_run."""
    arr = arr.copy()
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            j = i
            while j < len(arr) and arr[j] == 1:
                j += 1
            if j - i < min_run:
                arr[i:j] = 0
            i = j
        else:
            i += 1
    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT  (optional)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sleep_detection(scored_df: pd.DataFrame, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    t_h = scored_df["onset_s"].values / 3600.0
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Sleep Detection — Wearable Signals", fontsize=13, fontweight="bold")

    def _shade_sleep(ax):
        sleep = scored_df["is_sleep"].values
        in_s  = False
        for i, s in enumerate(sleep):
            if s and not in_s:
                x0 = t_h[i]; in_s = True
            if not s and in_s:
                ax.axvspan(x0, t_h[i], alpha=0.15, color="steelblue")
                in_s = False
        if in_s:
            ax.axvspan(x0, t_h[-1], alpha=0.15, color="steelblue")

    panel_specs = [
        ("accel_activity", "ACCEL activity\n(dynamic)",  "darkorange",  True),
        ("posture_lying_frac", "Lying posture\n(fraction)", "seagreen", False),
        ("hr_mean",        "Heart rate\n(bpm)",           "crimson",    False),
        ("skin_temp",      "Skin temp\n(°C)",              "purple",     False),
        ("sleep_score",    "Sleep score\n(0–1)",           "steelblue",  False),
    ]

    for ax, (col, label, color, log_scale) in zip(axes, panel_specs):
        if col not in scored_df.columns:
            ax.text(0.5, 0.5, f"{col} not available",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_ylabel(label, fontsize=9)
            _shade_sleep(ax)
            continue
        ax.plot(t_h, scored_df[col], color=color, linewidth=0.7, alpha=0.85)
        if log_scale and scored_df[col].max() > 0:
            ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        _shade_sleep(ax)

    # Sleep/wake band at bottom of last panel
    axes[-1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    axes[-1].set_xlabel("Time (hours from recording start)", fontsize=10)
    sleep_patch = mpatches.Patch(color="steelblue", alpha=0.3, label="Detected sleep")
    fig.legend(handles=[sleep_patch], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER CONVERTED CSVs TO SLEEP-ONLY
# ══════════════════════════════════════════════════════════════════════════════

def filter_csv_to_sleep(
    csv_dir: str,
    windows_csv: str,
    out_dir: str,
    stem_filter: Optional[str] = None,
):
    """
    Read sleep windows CSV, filter all *_ecg.csv / *_resp.csv to sleep epochs only.
    """
    os.makedirs(out_dir, exist_ok=True)
    windows_df = pd.read_csv(windows_csv)
    sleep_segs = set(windows_df[windows_df["is_sleep"] == 1]["segment_idx"].tolist())
    print(f"\n  Sleep segments to keep: {len(sleep_segs)}")

    csv_dir_p = Path(csv_dir)
    for channel in ["ecg", "resp", "summary"]:
        pattern = f"*_{channel}.csv" if not stem_filter else f"{stem_filter}_{channel}.csv"
        for csv_path in sorted(csv_dir_p.glob(pattern)):
            df = pd.read_csv(csv_path)
            before = len(df)
            df_sleep = df[df["segment_idx"].isin(sleep_segs)].reset_index(drop=True)
            after = len(df_sleep)

            out_name = csv_path.stem + "_sleep.csv"
            out_path = os.path.join(out_dir, out_name)
            df_sleep.to_csv(out_path, index=False)
            print(f"  {csv_path.name}: {before} → {after} segments → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Detect sleep windows from wearable EDF and filter segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--detect",  action="store_true", help="Run sleep detection from EDF")
    ap.add_argument("--filter",  action="store_true", help="Filter converted CSVs to sleep-only")
    ap.add_argument("--plot",    action="store_true", help="Save sleep detection plot")
    ap.add_argument("--input",   "-i", default=None,  help="EDF/BDF file (required for --detect)")
    ap.add_argument("--csv-dir", default="./converted", help="Directory of converted CSVs")
    ap.add_argument("--windows", default=None,
                    help="Sleep windows CSV (auto-named if --detect also given)")
    ap.add_argument("--out-dir", "-o", default="./converted",
                    help="Output directory for windows CSV / filtered CSVs")
    ap.add_argument("--max-duration-s", type=int, default=None,
                    help="Limit EDF reading to N seconds (set same as edf_to_pipeline.py)")
    args = ap.parse_args()

    if not args.detect and not args.filter:
        ap.print_help()
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)
    windows_csv = args.windows

    # ── DETECT ────────────────────────────────────────────────────────────────
    if args.detect:
        if not args.input:
            print("ERROR: --input required for --detect")
            sys.exit(1)

        stem = Path(args.input).stem
        windows_csv = os.path.join(args.out_dir, f"{stem}_sleep_windows.csv")

        print(f"\nExtracting sleep features from: {os.path.basename(args.input)}")
        feat_df = extract_sleep_features(args.input, args.max_duration_s)

        print("\nScoring sleep/wake ...")
        scored_df = score_sleep(feat_df)

        scored_df.to_csv(windows_csv, index=False)
        print(f"\n  Sleep windows saved → {windows_csv}")

        if args.plot:
            plot_path = os.path.join(args.out_dir, f"{stem}_sleep_plot.png")
            plot_sleep_detection(scored_df, plot_path)

    # ── FILTER ────────────────────────────────────────────────────────────────
    if args.filter:
        if not windows_csv:
            print("ERROR: --windows <sleep_windows.csv> required for --filter")
            sys.exit(1)
        if not os.path.exists(windows_csv):
            print(f"ERROR: {windows_csv} not found")
            sys.exit(1)

        stem_filter = Path(args.input).stem if args.input else None
        sleep_out   = os.path.join(args.out_dir, "sleep_only")
        filter_csv_to_sleep(args.csv_dir, windows_csv, sleep_out, stem_filter)
        print(f"\n✓ Sleep-filtered CSVs → {sleep_out}/")
        print("  Next: python edf_test_loader.py --data", sleep_out)


if __name__ == "__main__":
    main()