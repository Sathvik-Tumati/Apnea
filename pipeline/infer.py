"""
infer.py
========
Unified apnea inference — handles both 125 Hz and 250 Hz ECG CSV files
automatically. No separate scripts needed.

Detection
---------
Reads the first 3 rows of the CSV, counts ecgData[] columns:
  3750 columns → 125 Hz  (30 s × 125 = 3750 samples per row)
  7500 columns → 250 Hz  (30 s × 250 = 7500 samples per row)
A spectral sanity check runs alongside the column count to catch
mislabelled files. Override with --force-hz if needed.

250 Hz path
-----------
Dual-path downsampling to 125 Hz:
  Path A : Cubic spline interpolation (smooth, shape-preserving)
  Path B : Polyphase FIR decimation   (anti-aliasing, noise-robust)
  Fused  : Element-wise mean of A and B
All downstream processing (R-peaks, HRV, EDR, features) runs at 125 Hz.

SpO2 features
-------------
When mongo_infer.py has computed real SpO2 features and set has_spo2=1 in
the CSV, _extract_features() reads them directly from the row instead of
substituting neutral defaults.  The model therefore receives real SpO2
signal whenever it is available, and the modality flag routes accordingly.

AHI denominator
---------------
Duration is computed from actual first/last timestamps in the DataFrame
when they are parseable, falling back to n_segments × 30s only when
timestamps are missing or malformed.  This avoids inflating AHI when
there are packet gaps in the recording.

Feature representation (flatten vs aggregate)
-----------------------------------------------
pipeline.py's models flatten each (T=10, F=30) feature sequence into 300
raw positional columns. pipeline/train_improved.py can ALSO train models
on an "aggregate" representation (per-feature mean/std/min/max/last/delta
across the window, ~180 columns) which has tested better in controlled
comparisons. A model trained on one representation is meaningless if fed
the other — same weights, different-meaning input columns.

To avoid that failure mode, train_improved.py writes a companion
`<model_stem>_meta.json` file recording which representation the model
expects. This script looks for that file next to --model and builds
features accordingly automatically. If no metadata file is found (e.g.
a plain pipeline.py-trained model), it assumes "flatten" — pipeline.py's
only representation — which preserves backward compatibility.

Workflow
--------
1. Train on MIMIC and save model + scaler:
       python pipeline/pipeline.py --fresh --save-model
   ...or, for the tuned/aggregate-features path:
       python pipeline/train_improved.py --feature-mode aggregate --save-model

2. Run inference:
       python infer.py --csv /path/to/ecg_analysis.arrhythmia_results.csv

Output (in infer_output/)
--------------------------
  infer_results_<admissionId>.csv   per-segment features + predictions
  infer_summary.csv                 one row per patient — AHI proxy + severity
  infer_summary.txt                 human-readable report

Usage
-----
python infer.py --csv /path/to/file.csv
python infer.py --csv /path/to/file.csv --force-hz 250
python infer.py --csv /path/to/file.csv --admission ADM914251465 --threshold 0.40
python infer.py --csv /path/to/file.csv --feature-mode aggregate   # override auto-detect
"""

import argparse
import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, find_peaks, resample_poly, welch
from scipy.signal.windows import tukey

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── neurokit2 ─────────────────────────────────────────────────────────────────
try:
    import neurokit2 as nk
    HAS_NK = True
    logger.info("neurokit2 available — using for R-peak detection")
except Exception:
    HAS_NK = False
    logger.warning("neurokit2 not available — using scipy R-peak fallback")

# ── Constants ─────────────────────────────────────────────────────────────────
FS_ECG        = 125
FS_RESP       = 4
SEGMENT_LEN_S = 30
SAMPLES_125   = 3750
SAMPLES_250   = 7500
TIMESTEPS     = 10
HR_TOLERANCE  = 15.0

KNOWN_SAMPLE_COUNTS = {3750: 125, 7500: 250}

ECG_COLS_125   = [f"ecgData[{i}]" for i in range(SAMPLES_125)]
ECG_COLS_250   = [f"ecgData[{i}]" for i in range(SAMPLES_250)]
HR_SUBSEG_COLS = [f"analysis.segments[{i}].morphology.hr_bpm" for i in range(6)]
SIG_QUAL_COL   = "analysis.summary.signal_quality"
RHYTHM_COLS    = [f"analysis.segments[{i}].rhythm_label"  for i in range(6)]
ECTOPY_COLS    = [f"analysis.segments[{i}].ectopy_label"  for i in range(6)]

# SpO2 columns that mongo_infer.py may have written into the CSV
SPO2_CSV_COLS = [
    "spo2_mean", "spo2_min", "spo2_delta_index",
    "odi", "t90", "spo2_approx_entropy", "has_spo2",
]

# Timezone for human-readable log output
from datetime import timezone as _tz, timedelta as _td
_IST = _tz(_td(hours=5, minutes=30))

def _fmt_ts_ist(ts_raw) -> str:
    """Convert a raw UTC timestamp (str, Timestamp, datetime, or None) to
    an IST string suitable for log output.  Always safe — returns the raw
    value stringified on any parse failure, never raises."""
    if ts_raw is None or (isinstance(ts_raw, str) and not ts_raw.strip()):
        return ""
    try:
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return str(ts_raw)

APNEA_FEATURE_COLS = [
    "rr_mean", "rr_std", "rmssd", "pnn50", "mean_hr", "hr_range", "lf_hf_ratio",
    "resp_rate_bpm", "resp_rate_variability", "flatline_duration_s",
    "resp_amplitude_mean", "resp_amplitude_std",
    "spo2_mean", "spo2_min", "spo2_delta_index", "odi", "t90", "spo2_approx_entropy",
    "map_mean", "map_std", "map_variability", "sbp_max", "dbp_min", "pulse_pressure",
    "resp_spo2_lag_s", "ptt_ms", "ecg_resp_coherence",
    "has_spo2", "has_abp", "has_resp_gt",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL FEATURE-REPRESENTATION METADATA
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_FEATURE_MODE = "flatten"  # pipeline.py's only representation


def _load_model_feature_mode(model_path: str) -> str:
    """
    Look for a `<model_stem>_meta.json` next to `model_path` (written by
    pipeline/train_improved.py) recording which feature representation
    this model expects. Falls back to "flatten" — the only representation
    plain pipeline.py-trained models use — when no metadata file exists,
    so older models keep working unchanged.
    """
    p = Path(model_path)
    meta_path = p.parent / (p.stem + "_meta.json")
    if not meta_path.exists():
        logger.info(
            "[MODEL] No %s found — assuming feature_mode='%s' "
            "(pipeline.py's default representation).",
            meta_path.name, DEFAULT_FEATURE_MODE,
        )
        return DEFAULT_FEATURE_MODE
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        mode = meta.get("feature_mode", DEFAULT_FEATURE_MODE)
        logger.info("[MODEL] Loaded %s → feature_mode='%s'", meta_path.name, mode)
        return mode
    except Exception as exc:
        logger.warning(
            "[MODEL] Failed to read %s (%s) — assuming feature_mode='%s'",
            meta_path.name, exc, DEFAULT_FEATURE_MODE,
        )
        return DEFAULT_FEATURE_MODE


def _aggregate_sequence_features(X_seq: np.ndarray) -> np.ndarray:
    """
    Must exactly mirror pipeline/train_improved.py's
    _aggregate_sequence_features — same stats, same concatenation order —
    or a model trained on one and served from the other will silently
    produce garbage predictions despite matching array shapes.
    """
    mean_  = X_seq.mean(axis=1)
    std_   = X_seq.std(axis=1)
    min_   = X_seq.min(axis=1)
    max_   = X_seq.max(axis=1)
    last_  = X_seq[:, -1, :]
    delta_ = X_seq[:, -1, :] - X_seq[:, 0, :]
    return np.concatenate([mean_, std_, min_, max_, last_, delta_], axis=1)


def _flatten_sequence_features(X_seq: np.ndarray) -> np.ndarray:
    return X_seq.reshape(len(X_seq), -1)


def _build_model_features(X_seq: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode == "aggregate":
        return _aggregate_sequence_features(X_seq)
    elif feature_mode == "flatten":
        return _flatten_sequence_features(X_seq)
    elif feature_mode == "both":
        return np.concatenate(
            [_aggregate_sequence_features(X_seq), _flatten_sequence_features(X_seq)], axis=1
        )
    else:
        logger.warning(
            "[MODEL] Unknown feature_mode '%s' — falling back to flatten.", feature_mode
        )
        return _flatten_sequence_features(X_seq)


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLING RATE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_sampling_rate(csv_path: str) -> Tuple[int, str]:
    logger.info("[DETECT] Reading header + 3 rows from %s ...", csv_path)
    df_head = pd.read_csv(csv_path, nrows=3, low_memory=False)

    ecg_cols = sorted(
        [c for c in df_head.columns if c.startswith("ecgData[")],
        key=lambda c: int(c.split("[")[1].rstrip("]")),
    )
    n_cols = len(ecg_cols)

    logger.info("[DETECT] Found %d ecgData[] columns (first=%s  last=%s)",
                n_cols,
                ecg_cols[0]  if ecg_cols else "none",
                ecg_cols[-1] if ecg_cols else "none")

    if n_cols == 0:
        raise ValueError(
            "No ecgData[] columns found. "
            "Check that this is an arrhythmia results CSV with ecgData[N] columns."
        )

    hz = KNOWN_SAMPLE_COUNTS.get(n_cols)
    if hz is None:
        d125 = abs(n_cols / 125.0 - 30.0)
        d250 = abs(n_cols / 250.0 - 30.0)
        hz   = 125 if d125 <= d250 else 250
        logger.warning("[DETECT] Non-standard column count %d — best guess: %d Hz", n_cols, hz)

    ecg_sample = df_head[ecg_cols].iloc[0].values.astype(float)
    ecg_sample[np.isnan(ecg_sample)] = 0.0
    if len(ecg_sample) >= 256:
        f, pxx = welch(ecg_sample, fs=hz,
                       nperseg=min(512, len(ecg_sample) // 4), nfft=1024)
        total    = float(np.sum(pxx)) or 1.0
        hf_frac  = float(np.sum(pxx[f > 62.5])) / total
        if hz == 125 and hf_frac > 0.05:
            logger.warning(
                "[DETECT] Column count says 125 Hz but %.1f%% power above 62.5 Hz. "
                "Use --force-hz 250 to override.", hf_frac * 100)
        else:
            logger.info("[DETECT] Spectral check: %.1f%% power above 62.5 Hz — "
                        "consistent with %d Hz.", hf_frac * 100, hz)

    evidence = f"{n_cols} ecgData[] columns = {n_cols // hz}s × {hz} Hz per segment"
    logger.info("[DETECT] ─────────────────────────────────────────")
    logger.info("[DETECT]  Detected : %d Hz", hz)
    logger.info("[DETECT]  Evidence : %s", evidence)
    logger.info("[DETECT] ─────────────────────────────────────────")
    return hz, evidence


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _bandpass(sig: np.ndarray, fs: int,
              lo: float = 0.5, hi: float = 40.0, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    hi  = min(hi, nyq - 0.1)
    lo  = min(lo, hi - 0.1)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    if HAS_NK:
        try:
            _, info = nk.ecg_process(ecg, sampling_rate=fs)
            peaks   = np.array(info["ECG_R_Peaks"], dtype=int)
            if len(peaks) >= 2:
                hr = 60.0 / (np.mean(np.diff(peaks)) / fs + 1e-9)
                if 30 <= hr <= 200:
                    # ── Half-rate correction ──────────────────────────────────
                    # If detected HR is physiologically implausible for sleep
                    # (< 35 bpm) but the signal has enough amplitude to suggest
                    # real cardiac activity, try inserting midpoint peaks between
                    # existing ones. This catches T/P-wave locking where every
                    # other beat is missed.
                    if hr < 45 and len(peaks) >= 3:
                        midpoints = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
                        # Only insert midpoint if local signal has a peak there
                        valid_mids = []
                        win = max(1, int(fs * 0.08))
                        for mp in midpoints:
                            lo = max(0, mp - win)
                            hi = min(len(ecg), mp + win)
                            local = ecg[lo:hi]
                            if len(local) > 0 and ecg[mp] >= np.percentile(local, 60):
                                valid_mids.append(mp)
                        if len(valid_mids) >= len(peaks) * 0.5:
                            corrected = np.sort(np.concatenate([peaks, valid_mids]))
                            corrected_hr = 60.0 / (np.mean(np.diff(corrected)) / fs + 1e-9)
                            if 40 <= corrected_hr <= 120:
                                logger.debug(
                                    "_detect_r_peaks: half-rate corrected "
                                    "%.0f→%.0f bpm", hr, corrected_hr)
                                return corrected
                    return peaks
                logger.debug("nk peaks imply %.0f BPM — scipy fallback", hr)
        except Exception as exc:
            logger.debug("nk.ecg_process: %s — scipy fallback", exc)

    min_dist = int(fs * 0.4)
    thr      = float(np.mean(ecg) + 0.15 * np.std(ecg))
    peaks, _ = find_peaks(ecg, distance=min_dist, height=thr)
    if len(peaks) < 5:
        thr      = float(np.percentile(ecg, 60))
        peaks, _ = find_peaks(ecg, distance=min_dist, height=thr)
    
    # ── Half-rate correction for scipy fallback too ──────────────────────────
    if len(peaks) >= 3:
        hr = 60.0 / (np.mean(np.diff(peaks)) / fs + 1e-9)
        if hr < 45 and len(peaks) >= 3:
            midpoints = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
            valid_mids = []
            win = max(1, int(fs * 0.08))
            for mp in midpoints:
                lo = max(0, mp - win)
                hi = min(len(ecg), mp + win)
                local = ecg[lo:hi]
                if len(local) > 0 and ecg[mp] >= np.percentile(local, 60):
                    valid_mids.append(mp)
            if len(valid_mids) >= len(peaks) * 0.5:
                corrected = np.sort(np.concatenate([peaks, valid_mids]))
                corrected_hr = 60.0 / (np.mean(np.diff(corrected)) / fs + 1e-9)
                if 40 <= corrected_hr <= 120:
                    logger.debug(
                        "_detect_r_peaks (scipy): half-rate corrected "
                        "%.0f→%.0f bpm", hr, corrected_hr)
                    return corrected
    
    return peaks


def _downsample_250_to_125(ecg_250: np.ndarray) -> np.ndarray:
    t_raw    = np.arange(len(ecg_250)) / 250.0
    t_target = np.arange(SAMPLES_125)  / 125.0
    try:
        path_a = CubicSpline(t_raw, ecg_250, bc_type="not-a-knot")(t_target)
    except Exception:
        path_a = np.interp(t_target, t_raw, ecg_250)
    path_b  = resample_poly(ecg_250, up=1, down=2).astype(float)
    min_len = min(len(path_a), len(path_b), SAMPLES_125)
    fused   = 0.5 * (path_a[:min_len] + path_b[:min_len])
    if len(fused) < SAMPLES_125:
        fused = np.pad(fused, (0, SAMPLES_125 - len(fused)), mode="edge")
    return fused[:SAMPLES_125]


def _get_ecg(row: pd.Series, hz: int) -> np.ndarray:
    cols = ECG_COLS_250 if hz == 250 else ECG_COLS_125
    ecg  = row[cols].values.astype(float)
    if np.isnan(ecg).any():
        ecg = pd.Series(ecg).ffill().bfill().fillna(0.0).values
    if hz == 250:
        ecg = _downsample_250_to_125(ecg)
    return _bandpass(ecg, FS_ECG)


def _compute_edr(ecg: np.ndarray, r_peaks: np.ndarray,
                 fs_ecg: int = FS_ECG, fs_resp: int = FS_RESP) -> np.ndarray:
    if len(r_peaks) < 8:
        return np.zeros(int(len(ecg) * fs_resp / fs_ecg))
    t_peaks   = r_peaks / fs_ecg
    t_uniform = np.arange(0, len(ecg) / fs_ecg, 1.0 / fs_resp)

    def _env(raw_v, times):
        if len(raw_v) < 6:
            return np.zeros_like(t_uniform)
        v = raw_v - np.polyval(np.polyfit(times, raw_v, 1), times)
        s = np.interp(t_uniform, times, v)
        return (s - np.mean(s)) / (np.std(s) + 1e-9)

    qrs_win = max(1, int(0.06 * fs_ecg))
    areas   = [np.sum(np.abs(ecg[max(0, r - qrs_win):min(len(ecg), r + qrs_win)]))
               for r in r_peaks]
    m3 = _env(np.array(areas, dtype=float), t_peaks)
    beats = [ecg[r - qrs_win:r + qrs_win]
             for r in r_peaks if r - qrs_win >= 0 and r + qrs_win <= len(ecg)]
    m4 = m3
    if len(beats) >= 8:
        X = np.array(beats, dtype=float)
        X -= X.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            m4 = _env(U[:, 0] * S[0], t_peaks[:len(beats)])
        except np.linalg.LinAlgError:
            pass
    n = min(len(m3), len(m4))
    fused = np.median(np.vstack([m3[:n], m4[:n]]), axis=0)
    nyq = fs_resp / 2.0
    b, a = butter(3, [0.1 / nyq, 0.5 / nyq], btype="band")
    out = np.zeros_like(t_uniform)
    out[:n] = filtfilt(b, a, fused)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_features(
    ecg: np.ndarray,
    baseline: Dict[str, float],
    row: Optional[pd.Series] = None,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Extract APNEA_FEATURE_COLS from a 30-second ECG at 125 Hz.

    SpO2 features are read from `row` when has_spo2=1 (written by mongo_infer.py),
    otherwise the previous neutral defaults are used so ECG-only recordings
    continue to work unchanged.

    Returns (feats, r_peaks, ecg_hr_bpm).
    """
    feats: Dict[str, float] = {}
    r_peaks = _detect_r_peaks(ecg, FS_ECG)

    # ── HRV ──────────────────────────────────────────────────────────────────
    if len(r_peaks) >= 3:
        rr_ms    = np.diff(r_peaks) / FS_ECG * 1000.0
        rr_diffs = np.diff(rr_ms)
        feats["rr_mean"]     = float(np.mean(rr_ms))
        feats["rr_std"]      = float(np.std(rr_ms))
        feats["rmssd"]       = float(np.sqrt(np.mean(rr_diffs ** 2)))
        feats["pnn50"]       = float(np.sum(np.abs(rr_diffs) > 50) / max(len(rr_ms), 1))
        feats["mean_hr"]     = float(60000.0 / (np.mean(rr_ms) + 1e-6))
        feats["hr_range"]    = float(
            60000.0 / (np.min(rr_ms) + 1e-6) - 60000.0 / (np.max(rr_ms) + 1e-6))
        feats["lf_hf_ratio"] = 0.0
        ecg_hr = feats["mean_hr"]
    else:
        for k in ("rr_mean", "rr_std", "rmssd", "pnn50",
                  "mean_hr", "hr_range", "lf_hf_ratio"):
            feats[k] = 0.0
        ecg_hr = 0.0

    # ── EDR / resp ────────────────────────────────────────────────────────────
    resp = _compute_edr(ecg, r_peaks)
    feats["resp_amplitude_mean"] = float(np.mean(np.abs(resp)))
    feats["resp_amplitude_std"]  = float(np.std(resp))
    threshold  = np.mean(resp) - 1.5 * np.std(resp)
    suppressed = resp < threshold
    max_run = cur = 0
    for v in suppressed:
        if v:
            cur += 1; max_run = max(max_run, cur)
        else:
            cur = 0
    feats["flatline_duration_s"] = float(max_run / FS_RESP)

    try:
        w       = tukey(len(resp), alpha=0.1)
        nperseg = min(len(resp), max(8, int(FS_RESP * 60)))
        f, pxx  = welch(resp * w, fs=FS_RESP, nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=2048)
        inn     = (f >= 0.1) & (f <= 0.6)
        feats["resp_rate_bpm"] = (float(f[inn][np.argmax(pxx[inn])] * 60.0)
                                  if np.any(inn) else 0.0)
        rp2, _ = find_peaks(resp, distance=int(FS_RESP * 1.5))
        feats["resp_rate_variability"] = (float(np.std(np.diff(rp2) / FS_RESP))
                                          if len(rp2) >= 2 else 0.0)
    except Exception:
        feats["resp_rate_bpm"] = feats["resp_rate_variability"] = 0.0

    # ── SpO2 features — read from CSV row when has_spo2=1 ────────────────────
    has_spo2 = 0
    if row is not None:
        try:
            has_spo2 = int(float(row.get("has_spo2", 0)))
        except (TypeError, ValueError):
            has_spo2 = 0

    if has_spo2 == 1 and row is not None:
        # Real SpO2 data written by mongo_infer.py — use it directly
        def _safe(col, default):
            try:
                v = float(row.get(col, default))
                return v if np.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        feats.update({
            "spo2_mean":           _safe("spo2_mean",           97.0),
            "spo2_min":            _safe("spo2_min",            97.0),
            "spo2_delta_index":    _safe("spo2_delta_index",    0.0),
            "odi":                 _safe("odi",                  0.0),
            "t90":                 _safe("t90",                  0.0),
            "spo2_approx_entropy": _safe("spo2_approx_entropy", 0.0),
        })
    else:
        # ECG-only path — neutral defaults so the scaler / model still work
        feats.update({
            "spo2_mean": 97.0, "spo2_min": 97.0, "spo2_delta_index": 0.0,
            "odi": 0.0, "t90": 0.0, "spo2_approx_entropy": 0.0,
        })

    # ── ABP + cross-signal — not available at wearable inference time ─────────
    feats.update({
        "map_mean": 0.0, "map_std": 0.0, "map_variability": 0.0,
        "sbp_max": 0.0, "dbp_min": 0.0, "pulse_pressure": 0.0,
        "resp_spo2_lag_s": 0.0, "ptt_ms": 0.0, "ecg_resp_coherence": 0.0,
    })

    # ── Modality flags ────────────────────────────────────────────────────────
    feats["has_spo2"]    = has_spo2
    feats["has_abp"]     = 0
    feats["has_resp_gt"] = 0

    return feats, r_peaks, ecg_hr


# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_baseline(df: pd.DataFrame, hz: int) -> Dict[str, float]:
    hr_vals, rmssd_vals, rr_vals = [], [], []
    for _, row in df.head(10).iterrows():
        sub_hrs = [row.get(c, np.nan) for c in HR_SUBSEG_COLS]
        valid   = [v for v in sub_hrs if pd.notna(v)]
        if valid:
            hr_vals.append(float(np.mean(valid)))
            rr_vals.append(60000.0 / (np.mean(valid) + 1e-6))
        ecg = _get_ecg(row, hz)
        rp  = _detect_r_peaks(ecg, FS_ECG)
        if len(rp) >= 3:
            rr = np.diff(rp) / FS_ECG * 1000.0
            if len(rr) >= 2:
                rmssd_vals.append(float(np.sqrt(np.mean(np.diff(rr) ** 2))))
    return {
        "baseline_spo2":    97.0,
        "baseline_rmssd":   float(np.mean(rmssd_vals)) if rmssd_vals else 35.0,
        "baseline_rr_ms":   float(np.mean(rr_vals))    if rr_vals    else 833.0,
        "baseline_map_std": 5.0,
        "baseline_sbp":     120.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  AHI DURATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _recording_duration_min(adm_df: pd.DataFrame, n_segs: int) -> float:
    """
    Return recording duration in minutes.

    Prefer actual first/last timestamps from the DataFrame — this correctly
    accounts for packet gaps that would otherwise DEFLATE AHI when using
    n_segs × 30s as the denominator (gap recordings have wall_min > scored_min).

    Falls back to n_segs × 30s only when timestamps are clearly wrong:
      - too_short: wall_min < 90% of scored_min (clock skew / corrupt timestamps)
      - too_long:  wall_min > 1440 min = 24 h (impossible for one admission)
    """
    ts_col = "timestamp"
    if ts_col not in adm_df.columns:
        return n_segs * SEGMENT_LEN_S / 60.0

    ts = pd.to_datetime(adm_df[ts_col], utc=True, errors="coerce").dropna()
    if len(ts) < 2:
        return n_segs * SEGMENT_LEN_S / 60.0

    wall_min   = (ts.max() - ts.min()).total_seconds() / 60.0
    scored_min = n_segs * SEGMENT_LEN_S / 60.0

    # wall_min >= scored_min is EXPECTED when there are recording gaps.
    # Only reject timestamps that are clearly impossible:
    too_short = wall_min < scored_min * 0.90   # >10% shorter than scored → corrupt
    too_long  = wall_min > 1440.0              # >24h is impossible for one admission

    if too_short or too_long:
        logger.warning(
            "[AHI] Timestamp-derived duration %.1f min rejected "
            "(scored=%.1f min, too_short=%s, too_long=%s) — "
            "falling back to segment count.",
            wall_min, scored_min, too_short, too_long,
        )
        return scored_min

    logger.info(
        "[AHI] Duration from timestamps: %.1f min  "
        "(segment-count estimate was %.1f min, recording gap = %.1f min)",
        wall_min, scored_min, max(0.0, wall_min - scored_min),
    )
    return wall_min


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-ADMISSION INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sequences(X: np.ndarray, t: int) -> np.ndarray:
    if len(X) <= t:
        return np.empty((0, t, X.shape[1]))
    return np.array([X[i:i + t] for i in range(len(X) - t)])


def _run_one_admission(
    adm_id:    str,
    adm_df:    pd.DataFrame,
    model,
    scaler,
    threshold: float,
    out_dir:   str,
    hz:        int,
    model_path: str,
    feature_mode: str = DEFAULT_FEATURE_MODE,
) -> Dict:
    n = len(adm_df)
    ds_note = "" if hz == 125 else "  (250→125 Hz dual-path downsample)"
    logger.info("[%s] %d segments  (%.1f min)%s",
                adm_id, n, n * SEGMENT_LEN_S / 60.0, ds_note)

    # Track SpO2 coverage
    if "has_spo2" in adm_df.columns:
        n_spo2 = int((adm_df["has_spo2"].astype(float) == 1).sum())
        logger.info("[%s] SpO2 available in %d / %d segments (%.0f%%)",
                    adm_id, n_spo2, n, 100.0 * n_spo2 / max(n, 1))
    else:
        n_spo2 = 0
        logger.info("[%s] No has_spo2 column — ECG-only mode", adm_id)

    baseline   = _compute_baseline(adm_df, hz)
    rows_out   = []
    n_flag_hr  = 0
    n_flag_q   = 0
    inference_start = time.time() 

    for seg_i, (_, row) in enumerate(adm_df.iterrows()):
        ecg = _get_ecg(row, hz)

        sub_hrs       = [row.get(c, np.nan) for c in HR_SUBSEG_COLS]
        sub_hrs_valid = [v for v in sub_hrs if pd.notna(v)]
        ref_hr  = float(np.mean(sub_hrs_valid)) if sub_hrs_valid else np.nan
        ref_std = float(np.std(sub_hrs_valid))  if len(sub_hrs_valid) > 1 else 0.0

        sig_qual = str(row.get(SIG_QUAL_COL, "unknown")).lower()
        qual_ok  = sig_qual in ("acceptable", "good", "excellent")

        # Pass the full row so SpO2 columns can be read when has_spo2=1
        feats, r_peaks, ecg_hr = _extract_features(ecg, baseline, row=row)

        if pd.notna(ref_hr) and ecg_hr > 0:
            hr_diff = abs(ecg_hr - ref_hr)
            hr_ok   = hr_diff <= HR_TOLERANCE
        else:
            hr_diff = np.nan
            hr_ok   = True

        if not qual_ok:
            quality_flag = f"LOW_QUALITY_SIGNAL_{sig_qual.upper()}"
            n_flag_q += 1
        elif not hr_ok:
            quality_flag = "LOW_QUALITY_HR_MISMATCH"
            n_flag_hr += 1
            logger.warning(
                "[HR-GATE] %s seg=%d  ECG=%.1f bpm  ref=%.1f bpm  "
                "diff=%.1f > %.0f bpm  src=%dHz → FLAGGED",
                adm_id, seg_i, ecg_hr, ref_hr, hr_diff, HR_TOLERANCE, hz)
        else:
            quality_flag = "OK"

        rhythm_labels = [str(row.get(c, "")) for c in RHYTHM_COLS]
        ectopy_labels = [str(row.get(c, "")) for c in ECTOPY_COLS]
        feats.update({
            "segment_idx":    seg_i,
            "start_time_s":   seg_i * SEGMENT_LEN_S,
            "timestamp":      row.get("timestamp", ""),
            "fs_source_hz":   hz,
            "ecg_hr_bpm":     round(ecg_hr, 2),
            "ref_hr_bpm":     round(ref_hr, 2) if pd.notna(ref_hr) else np.nan,
            "ref_hr_std":     round(ref_std, 2),
            "hr_diff_bpm":    round(hr_diff, 2) if pd.notna(hr_diff) else np.nan,
            "hr_gate_pass":   int(hr_ok),
            "signal_quality": sig_qual,
            "quality_flag":   quality_flag,
            "dominant_rhythm": max(set(rhythm_labels), key=rhythm_labels.count),
            "ectopy_present":  int(any(
                v not in ("", "nan", "None") for v in ectopy_labels)),
            "overall_hr_bpm":  row.get("analysis.heart_rate_bpm", np.nan),
        })
        rows_out.append(feats)

        if (seg_i + 1) % 100 == 0 or seg_i == n - 1:
            logger.info("[%s] Features: %d / %d  "
                        "(hr_flagged=%d  qual_flagged=%d)",
                        adm_id, seg_i + 1, n, n_flag_hr, n_flag_q)

    feat_df = pd.DataFrame(rows_out)
    total_flagged = n_flag_hr + n_flag_q
    gate_pct      = 100.0 * total_flagged / max(n, 1)
    logger.info("[%s] Quality gate: %d / %d flagged (%.1f%%)",
                adm_id, total_flagged, n, gate_pct)
    if gate_pct > 20:
        logger.warning("[%s] %.0f%% of segments flagged — check R-peak detection.",
                       adm_id, gate_pct)

    for c in APNEA_FEATURE_COLS:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    is_flagged = (feat_df["quality_flag"] != "OK").values

    # ── Fix 2: Interpolate flagged segment features ──────────────────────────
    # Interpolate flagged segment features so they don't corrupt adjacent
    # sequence windows. Flagged segments are still excluded from prediction
    # by the is_flagged gate below — this only affects sequence context.
    feat_matrix = feat_df[APNEA_FEATURE_COLS].copy()
    if is_flagged.any():
        feat_matrix.loc[is_flagged, :] = np.nan
        feat_matrix = feat_matrix.interpolate(method='linear', limit_direction='both')
        logger.info("[%s] Interpolated features for %d flagged segments "
                    "to prevent sequence contamination",
                    adm_id, int(is_flagged.sum()))

    X_scaled = scaler.transform(feat_matrix.fillna(0.0).values.astype(float))
    X_seq    = _build_sequences(X_scaled, TIMESTEPS)

    prob_col = np.full(n, np.nan)
    pred_col = np.full(n, np.nan)

    if len(X_seq) > 0:
        logger.info("[%s] Running model on %d sequences (feature_mode=%s) ...",
                    adm_id, len(X_seq), feature_mode)
        
        if hasattr(model, "predict_proba"):
            # Build features the way THIS model was actually trained on —
            # previously this always flattened, which silently breaks (shape
            # mismatch, or worse a shape coincidence producing garbage
            # predictions) for models trained with train_improved.py's
            # --feature-mode aggregate. See _load_model_feature_mode().
            X_seq_model = _build_model_features(X_seq, feature_mode)
            y_prob = model.predict_proba(X_seq_model)[:, 1]
            logger.info("[%s] Using predict_proba for tree model (features=%d cols)",
                        adm_id, X_seq_model.shape[1])
        else:
            y_prob = model.predict(X_seq, verbose=0, batch_size=64).flatten()

        # ── SpO2 fusion ───────────────────────────────────────────────────────
        # Compute per-segment directional SpO2 delta (positive = desaturation).
        # spo2_mean vs 90th-pct baseline gives the drop magnitude; only applied
        # when has_spo2=1. Segments without SpO2 pass through unchanged.
        if "has_spo2" in feat_df.columns and "spo2_mean" in feat_df.columns:
            spo2_mask = feat_df["has_spo2"].astype(float) == 1
            if spo2_mask.any():
                raw_baseline = float(
                    feat_df.loc[spo2_mask, "spo2_mean"].quantile(0.95)
                )
                spo2_baseline = max(raw_baseline, 97.0)
            else:
                spo2_baseline = 97.0

            # Store raw ECG-only probability for audit before fusion
            feat_df["xgb_prob_raw"] = np.nan

            def _fuse(ecg_prob: float, seg_idx: int) -> float:
                """
                Adjust ECG probability using SpO2 evidence.
                - has_spo2=0 → unchanged (no penalty for missing data)
                - desaturation ≥3% → boost (AASM apnea criterion threshold)
                - desaturation 1.5-3% → mild boost
                - SpO2 rising → suppress (contradicts apnea hypothesis)
                - flat SpO2 → mild suppress ONLY for borderline ECG predictions
                Adjustment scales with ECG uncertainty — maximum effect at
                prob=0.5, near-zero effect at prob=0.0 or prob=1.0.
                """
                if seg_idx >= len(feat_df):
                    return ecg_prob
                row_s = feat_df.iloc[seg_idx]
                if int(float(row_s.get("has_spo2", 0))) == 0:
                    return ecg_prob   # no SpO2 — ECG stands alone

                # Positive delta = SpO2 dropped below patient baseline
                delta = spo2_baseline - float(row_s.get("spo2_mean", spo2_baseline))

                # Scale by ECG uncertainty: 1.0 at threshold, 0.0 at certainty
                uncertainty = 1.0 - abs(ecg_prob - 0.5) * 2.0

                if delta >= 4.0:
                    adj = +0.12 * uncertainty   # significant desaturation
                elif delta >= 3.0:
                    adj = +0.06 * uncertainty   # borderline — small boost only
                elif delta <= -1.5:
                    adj = -0.08 * uncertainty   # SpO2 clearly rising — suppress
                else:
                    # Flat SpO2 during a predicted apnea event is mild contradicting evidence
                    # only suppress if ECG confidence is low (near threshold)
                    # High-confidence ECG predictions (prob > 0.75) are not touched
                    if ecg_prob < 0.70:
                        adj = -0.06 * uncertainty
                    else:
                        adj = 0.0

                return float(np.clip(ecg_prob + adj, 0.0, 1.0))

            n_fused_up   = 0
            n_fused_down = 0

            for j, yp in enumerate(y_prob):
                si = j + TIMESTEPS
                if si >= n:
                    continue
                if is_flagged[max(0, si - TIMESTEPS):si + 1].any():
                    continue

                feat_df.at[feat_df.index[si], "xgb_prob_raw"] = yp
                fused = _fuse(yp, si)

                if fused > yp + 0.01:
                    n_fused_up += 1
                elif fused < yp - 0.01:
                    n_fused_down += 1

                prob_col[si] = fused
                pred_col[si] = int(fused > threshold)

            logger.info(
                "[%s] SpO2 fusion: baseline=%.1f%%  boosted=%d  suppressed=%d  "
                "no-SpO2 unchanged=%d",
                adm_id, spo2_baseline, n_fused_up, n_fused_down,
                int((feat_df["has_spo2"].astype(float) == 0).sum()),
            )
        else:
            # No SpO2 columns in CSV — original path unchanged
            for j, yp in enumerate(y_prob):
                si = j + TIMESTEPS
                if si >= n:
                    continue
                if is_flagged[max(0, si - TIMESTEPS):si + 1].any():
                    continue
                prob_col[si] = yp
                pred_col[si] = int(yp > threshold)
    else:
        logger.warning(
            "[%s] Only %d segments — LSTM needs ≥%d for any predictions.",
            adm_id, n, TIMESTEPS + 1)

    feat_df["apnea_prob"]  = prob_col
    feat_df["apnea_pred"]  = pred_col
    feat_df["apnea_label"] = feat_df.apply(
        lambda r: ("APNEA"    if r["apnea_pred"] == 1.0
                   else "normal" if r["apnea_pred"] == 0.0
                   else r["quality_flag"]),
        axis=1)
    feat_df["admission_id"] = adm_id

    out_csv = os.path.join(out_dir, f"infer_results_{adm_id}.csv")
    feat_df.to_csv(out_csv, index=False)
    logger.info("[%s] Saved → %s", adm_id, out_csv)

    # ── Summary stats ─────────────────────────────────────────────────────────
    scored    = feat_df["apnea_pred"].notna()
    n_scored  = int(scored.sum())
    n_apnea   = int(feat_df.loc[scored, "apnea_pred"].sum())
    apnea_pct = 100.0 * n_apnea / max(n_scored, 1)

    # Use timestamp-derived duration for AHI denominator
    dur_min   = _recording_duration_min(adm_df, n)
    ahi_proxy = n_apnea / max(dur_min / 60.0, 1e-6)

    # ── Log fusion impact ─────────────────────────────────────────────────────
    if "xgb_prob_raw" in feat_df.columns:
        raw_apnea = int((feat_df["xgb_prob_raw"] > threshold).sum())
        if raw_apnea != n_apnea:
            logger.info(
                "[%s] SpO2 fusion changed apnea count: %d → %d  "
                "(AHI change: %.1f → %.1f /hr)",
                adm_id, raw_apnea, n_apnea,
                raw_apnea / max(dur_min / 60.0, 1e-6),
                ahi_proxy,
            )
    else:
        logger.info(
            "[%s] SpO2 fusion: apnea count unchanged (%d)  "
            "(SpO2 available in %.0f%% of segments)",
            adm_id, n_apnea,
            100.0 * (feat_df["has_spo2"].astype(float) == 1).sum() / max(n, 1)
        )

    mean_prob = float(np.nanmean(prob_col))
    severity  = ("Normal"       if ahi_proxy < 5  else
                 "Mild OSA"     if ahi_proxy < 15 else
                 "Moderate OSA" if ahi_proxy < 30 else
                 "Severe OSA")
    hr_diffs  = feat_df["hr_diff_bpm"].dropna()

    apnea_rows = feat_df[feat_df["apnea_pred"] == 1.0].head(5)
    if len(apnea_rows):
        logger.info("[%s] First apnea detections:", adm_id)
        for _, r in apnea_rows.iterrows():
            logger.info(
                "  seg=%3d  t=%s  prob=%.3f  ecg_hr=%.1f  ref_hr=%.1f  "
                "resp=%.1f bpm  spo2_mean=%.1f  has_spo2=%d  rhythm=%s  src=%dHz",
                int(r["segment_idx"]),
                _fmt_ts_ist(r.get("timestamp", "")),
                r["apnea_prob"], r.get("ecg_hr_bpm", 0.0),
                r.get("ref_hr_bpm", float("nan")),
                r.get("resp_rate_bpm", 0.0),
                r.get("spo2_mean", 97.0),
                int(r.get("has_spo2", 0)),
                r.get("dominant_rhythm", ""),
                int(r.get("fs_source_hz", hz)))
    else:
        logger.info("[%s] No apnea detected above threshold %.2f", adm_id, threshold)

    # ── Calculate inference time ──────────────────────────────────────────────
    inference_time_ms = round((time.time() - inference_start) * 1000, 1)
    
    # ── Get model version from path ──────────────────────────────────────────
    model_version = os.path.basename(model_path).replace('.pkl', '')
    
    # ── Calculate SpO2 coverage percentage ──────────────────────────────────
    spo2_coverage_pct = round(100.0 * n_spo2 / max(n, 1), 1)

    return {
        "admission_id":     adm_id,
        "fs_source_hz":     hz,
        "total_segments":   n,
        "flagged_segments": total_flagged,
        "flagged_hr":       n_flag_hr,
        "flagged_quality":  n_flag_q,
        "scored_segments":  n_scored,
        "n_apnea":          n_apnea,
        "n_normal":         n_scored - n_apnea,
        "apnea_pct":        round(apnea_pct, 1),
        "mean_apnea_prob":  round(mean_prob, 3),
        "max_apnea_prob":   round(float(np.nanmax(prob_col)), 3) if n_scored > 0 else 0.0,
        "min_apnea_prob":   round(float(np.nanmin(prob_col[~np.isnan(prob_col)])), 3) if n_scored > 0 else 0.0,
        "ahi_proxy":        round(ahi_proxy, 1),
        "severity":         severity,
        "duration_min":     round(dur_min, 1),
        "threshold":        threshold,
        "mean_hr_diff_bpm": round(float(hr_diffs.mean()), 2) if len(hr_diffs) else 0.0,
        "max_hr_diff_bpm":  round(float(hr_diffs.max()),  2) if len(hr_diffs) else 0.0,
        "model_version": model_version,
        "inference_time_ms": inference_time_ms,
        "spo2_coverage_pct": spo2_coverage_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(
    csv_path:     str,
    model_path:   str,
    scaler_path:  str,
    threshold:    float,
    out_dir:      str,
    admission_id: Optional[str],
    force_hz:     Optional[int] = None,
    chunk_rows:   int = 500,
    feature_mode: Optional[str] = None,
) -> None:

    os.makedirs(out_dir, exist_ok=True)

    if force_hz is not None:
        hz       = force_hz
        evidence = f"Forced by --force-hz {force_hz}"
        logger.info("[DETECT] Sampling rate forced to %d Hz", hz)
    else:
        try:
            hz, evidence = _detect_sampling_rate(csv_path)
        except Exception as exc:
            logger.error("[DETECT] %s", exc)
            logger.error("Use --force-hz 125 or --force-hz 250 to skip detection.")
            return

    for path, label in [(model_path, "Model"), (scaler_path, "Scaler")]:
        if not os.path.exists(path):
            logger.error("%s not found: '%s' — run pipeline.py --save-model", label, path)
            return

    # ── Feature representation: explicit --feature-mode wins, otherwise
    # auto-detect from the model's companion metadata file ────────────────────
    resolved_feature_mode = feature_mode or _load_model_feature_mode(model_path)
    if feature_mode is not None:
        logger.info("[MODEL] feature_mode explicitly overridden via --feature-mode='%s'",
                    feature_mode)

    # ── MODEL LOADING (XGBoost .pkl) ─────────────────────────────────────────
    logger.info("Loading model from %s", model_path)
    try:
        if model_path.endswith(".pkl"):
            # XGBoost / LightGBM tree model
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model_type = "xgboost"
            logger.info("Model type: XGBoost (tree-based)")
        else:
            logger.error(
                "Unsupported model format: '%s'. Only .pkl (XGBoost) is supported.",
                model_path,
            )
            return
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Include SpO2 feature columns in the load set so they arrive in the DataFrame
    ecg_cols_to_load = ECG_COLS_250 if hz == 250 else ECG_COLS_125
    needed_cols = (
        ["admissionId", "timestamp", SIG_QUAL_COL,
         "analysis.heart_rate_bpm", "analysis.background_rhythm"]
        + HR_SUBSEG_COLS + RHYTHM_COLS + ECTOPY_COLS
        + SPO2_CSV_COLS                    # ← SpO2 features + has_spo2 flag
        + ecg_cols_to_load
    )

    logger.info("[LOAD] Streaming %s (%d Hz, %d ECG cols/row) ...",
                csv_path, hz, len(ecg_cols_to_load))
    admission_chunks: Dict[str, List[pd.DataFrame]] = {}
    total_rows = 0

    for chunk in pd.read_csv(csv_path,
                              usecols=lambda c: c in needed_cols,
                              chunksize=chunk_rows, low_memory=False):
        total_rows += len(chunk)
        if admission_id:
            chunk = chunk[chunk["admissionId"] == admission_id]
            if chunk.empty:
                continue
        for adm, grp in chunk.groupby("admissionId"):
            admission_chunks.setdefault(adm, []).append(grp)
        if total_rows % 5000 < chunk_rows:
            logger.info("[LOAD] Streamed %d rows ...", total_rows)

    logger.info("[LOAD] Total rows: %d  |  Admissions: %d",
                total_rows, len(admission_chunks))

    if not admission_chunks:
        if admission_id:
            logger.error("admissionId '%s' not found in CSV.", admission_id)
        else:
            logger.error("No data loaded.")
        return

    all_summaries = []
    for adm_id, chunks in admission_chunks.items():
        adm_df = pd.concat(chunks, ignore_index=True)
        try:
            adm_df = adm_df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass

        logger.info("=" * 55)
        logger.info("  Admission: %s  (%d segments)", adm_id, len(adm_df))
        logger.info("=" * 55)

        summary = _run_one_admission(
            adm_id, adm_df, model, scaler, threshold, out_dir, hz,
            model_path=model_path,
            feature_mode=resolved_feature_mode,
        )
        if summary:
            all_summaries.append(summary)

    if all_summaries:
        ds_note = ("250→125 Hz dual-path downsample (cubic spline + polyphase FIR)"
                   if hz == 250 else "125 Hz native")
        lines = [
            "=" * 60,
            "  APNEA INFERENCE SUMMARY",
            "=" * 60,
            f"  Source     : {csv_path}",
            f"  Model      : {model_path}",
            f"  Model type : {model_type}",
            f"  Feature mode : {resolved_feature_mode}",
            f"  Threshold  : {threshold}",
            f"  HR gate    : ±{HR_TOLERANCE} bpm",
            f"  ECG input  : {hz} Hz  ({ds_note})",
            "",
        ]
        for s in all_summaries:
            lines += [
                f"  ── {s['admission_id']} " + "─" * max(0, 38 - len(s['admission_id'])),
                f"  Duration (wall)   : {s['duration_min']} min",
                f"  Total segments    : {s['total_segments']}",
                f"  Flagged (skipped) : {s['flagged_segments']}"
                f"  ({s['flagged_hr']} HR mismatch, {s['flagged_quality']} poor signal)",
                f"  Scored by model   : {s['scored_segments']}",
                f"  Apnea detected    : {s['n_apnea']}  ({s['apnea_pct']}%)",
                f"  Mean apnea prob   : {s['mean_apnea_prob']}",
                f"  AHI proxy         : {s['ahi_proxy']} /hr  → {s['severity']}",
                f"  Mean HR diff      : {s['mean_hr_diff_bpm']} bpm",
                f"  Max HR diff       : {s['max_hr_diff_bpm']} bpm",
                f"  Model version     : {s.get('model_version', 'unknown')}",
                f"  Inference time    : {s.get('inference_time_ms', 0)} ms",
                f"  SpO2 coverage     : {s.get('spo2_coverage_pct', 0)}%",
                "",
            ]
        lines += [
            "  AHI: <5 Normal | 5-15 Mild | 15-30 Moderate | >30 Severe",
            "  NOTE: Research prototype. Not for clinical use.",
            "=" * 60,
        ]
        summary_text = "\n".join(lines)
        logger.info("\n%s", summary_text)

        txt_path = os.path.join(out_dir, "infer_summary.txt")
        with open(txt_path, "w") as f:
            f.write(summary_text + "\n")

        csv_sum = os.path.join(out_dir, "infer_summary.csv")
        pd.DataFrame(all_summaries).to_csv(csv_sum, index=False)
        logger.info("Summary → %s  |  %s", txt_path, csv_sum)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apnea inference — auto-detects 125 Hz or 250 Hz ECG CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python infer.py --csv /path/to/file.csv
  python infer.py --csv /path/to/file.csv --force-hz 250
  python infer.py --csv /path/to/file.csv --admission ADM914251465
  python infer.py --csv /path/to/file.csv --threshold 0.40 --out-dir results/
  python infer.py --csv /path/to/file.csv --feature-mode aggregate
        """,
    )
    p.add_argument("--csv",       required=True)
    p.add_argument("--model",     default="apnea_model.keras")
    p.add_argument("--scaler",    default="apnea_scaler.pkl")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--out-dir",   default="infer_output")
    p.add_argument("--admission", default=None)
    p.add_argument("--force-hz",  type=int, choices=[125, 250], default=None)
    p.add_argument("--feature-mode", choices=["flatten", "aggregate", "both"], default=None,
                    help="Override auto-detected feature representation. Normally "
                         "left unset — infer.py reads this automatically from "
                         "<model_stem>_meta.json next to --model (written by "
                         "train_improved.py). Only set this if that file is "
                         "missing/wrong for some reason.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        csv_path     = args.csv,
        model_path   = args.model,
        scaler_path  = args.scaler,
        threshold    = args.threshold,
        out_dir      = args.out_dir,
        admission_id = args.admission,
        force_hz     = args.force_hz,
        feature_mode = args.feature_mode,
    )


if __name__ == "__main__":
    main()
