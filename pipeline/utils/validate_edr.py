"""
validate_edr_slpdb.py
=====================
Validates EDR accuracy against the MIT-BIH Polysomnographic Database (SLPDB).
Streams data directly from PhysioNet - no manual download required.

Optimisations over v1
---------------------
  • pn_dir= fix   — correct wfdb parameter for PhysioNet streaming
  • Local cache   — records downloaded once to ~/.cache/slpdb, re-used on every
                    subsequent run (near-instant reads)
  • Parallel load — ThreadPoolExecutor fetches / processes multiple records
                    simultaneously (default 6 workers)

Usage
-----
python validate_edr_slpdb.py
python validate_edr_slpdb.py --out-dir edr_validation_slpdb/
python validate_edr_slpdb.py --records slp01a slp02a slp03   # specific records
python validate_edr_slpdb.py --max-segments 5                # quick smoke-test
python validate_edr_slpdb.py --workers 8                     # more parallelism
python validate_edr_slpdb.py --no-cache                      # force re-download
"""

import argparse
import logging
import os
import warnings
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample, welch
from scipy.signal.windows import tukey
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import neurokit2 as nk
    HAS_NK = True
except ImportError:
    HAS_NK = False

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FS_ECG        = 125
FS_RESP       = 4
SEGMENT_LEN_S = 30
CACHE_DIR     = os.path.expanduser("~/.cache/slpdb")

# All 18 SLPDB records
SLPDB_RECORDS = [
    'slp01a', 'slp01b', 'slp02a', 'slp02b', 'slp03', 'slp04',
    'slp14',  'slp16',  'slp32',  'slp37',  'slp41', 'slp45',
    'slp48',  'slp59',  'slp60',  'slp61',  'slp66', 'slp67x',
]

# ECG / Respiration channel indices per record
SIGNAL_MAP = {
    'default': {'ecg': 0, 'resp': 3},
    'slp32':   {'ecg': 0, 'resp': 2},
    'slp37':   {'ecg': 0, 'resp': 3},
    'slp41':   {'ecg': 0, 'resp': 4},
    'slp45':   {'ecg': 0, 'resp': 4},
    'slp48':   {'ecg': 0, 'resp': 4},
    'slp59':   {'ecg': 0, 'resp': 3},
    'slp60':   {'ecg': 0, 'resp': 3},
    'slp61':   {'ecg': 0, 'resp': 3},
    'slp66':   {'ecg': 0, 'resp': 4},
    'slp67x':  {'ecg': 0, 'resp': 3},
}


# ── Data loading (with cache + pn_dir fix) ────────────────────────────────────

def _ensure_cached(record_name: str) -> str:
    """
    Return the local path for a record, downloading it first if needed.
    Uses ~/.cache/slpdb as the cache directory.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_hea = os.path.join(CACHE_DIR, record_name + ".hea")
    if not os.path.exists(local_hea):
        logger.info(f"[CACHE] Downloading {record_name} from PhysioNet...")
        try:
            wfdb.dl_database('slpdb/1.0.0', dl_dir=CACHE_DIR,
                             records=[record_name])
        except Exception:
            # dl_database may not support per-record selection on older wfdb;
            # fall back to streaming via pn_dir at read time.
            logger.warning(f"[CACHE] dl_database failed for {record_name} — "
                           "will stream instead.")
            return None
    return os.path.join(CACHE_DIR, record_name)


def _load_slpdb_record(
    record_name: str,
    use_cache: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[object], Optional[int]]:
    """
    Load a SLPDB record.  Tries local cache first; falls back to PhysioNet
    streaming via the correct ``pn_dir`` parameter.
    """
    try:
        local_path = _ensure_cached(record_name) if use_cache else None

        if local_path and os.path.exists(local_path + ".hea"):
            record = wfdb.rdrecord(local_path)
            source = "cache"
        else:
            # pn_dir is the correct keyword for PhysioNet remote access
            record = wfdb.rdrecord(record_name, pn_dir='slpdb/1.0.0')
            source = "stream"

        sig_map  = SIGNAL_MAP.get(record_name, SIGNAL_MAP['default'])
        ecg_idx  = sig_map['ecg']
        resp_idx = sig_map['resp']

        n_sigs = record.p_signal.shape[1]
        if ecg_idx >= n_sigs:
            logger.warning(f"{record_name}: ECG index {ecg_idx} out of range ({n_sigs} sigs)")
            return None, None, None, None
        if resp_idx >= n_sigs:
            logger.warning(f"{record_name}: Resp index {resp_idx} out of range ({n_sigs} sigs)")
            return None, None, None, None

        ecg_signal  = record.p_signal[:, ecg_idx]
        resp_signal = record.p_signal[:, resp_idx]
        fs          = record.fs

        # Load stage / apnea annotations (.st extension)
        try:
            if local_path and os.path.exists(local_path + ".st"):
                annotation = wfdb.rdann(local_path, 'st')
            else:
                annotation = wfdb.rdann(record_name, 'st', pn_dir='slpdb/1.0.0')
        except Exception as e:
            logger.warning(f"{record_name}: Could not load .st annotations: {e}")
            annotation = None

        logger.info(f"[LOAD] {record_name} ({source}): fs={fs} Hz  "
                    f"ECG ch={ecg_idx}  Resp ch={resp_idx}  "
                    f"dur={len(ecg_signal)/fs/60:.1f} min")
        return ecg_signal, resp_signal, annotation, fs

    except Exception as exc:
        logger.error(f"Failed to load {record_name}: {exc}")
        return None, None, None, None


# ── Signal processing helpers ─────────────────────────────────────────────────

def _resample_signal(sig: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    if fs_orig == fs_target:
        return sig
    n_target = int(len(sig) * fs_target / fs_orig)
    return resample(sig, n_target)


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
            return info["ECG_R_Peaks"]
        except Exception:
            pass
    peaks, _ = find_peaks(ecg, distance=int(fs * 0.4), height=float(np.std(ecg)))
    return peaks


def _compute_edr(ecg: np.ndarray, r_peaks: np.ndarray,
                 fs_ecg: int, fs_resp: int = 4) -> np.ndarray:
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

    areas = [np.sum(np.abs(ecg[max(0, r - qrs_win):min(len(ecg), r + qrs_win)]))
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

    n     = min(len(m3), len(m4))
    fused = np.median(np.vstack([m3[:n], m4[:n]]), axis=0)
    nyq   = fs_resp / 2.0
    b, a  = butter(3, [0.1 / nyq, 0.5 / nyq], btype="band")
    out   = np.zeros_like(t_uniform)
    out[:n] = filtfilt(b, a, fused)
    return out


def _compute_edr_snr(edr: np.ndarray, fs: int) -> float:
    try:
        f, pxx = welch(edr, fs=fs, nperseg=min(len(edr), 64), nfft=512)
        inn   = (f >= 0.1) & (f <= 0.6)
        total = float(np.sum(pxx))
        if total < 1e-12:
            return 0.0
        return float(np.sum(pxx[inn]) / total)
    except Exception:
        return 0.0


def _resp_rate_welch(sig: np.ndarray, fs: int) -> float:
    try:
        w       = tukey(len(sig), alpha=0.1)
        nperseg = min(len(sig), max(8, int(fs * 60)))
        f, pxx  = welch(sig * w, fs=fs, nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=2048)
        inn = (f >= 0.1) & (f <= 0.6)
        return float(f[inn][np.argmax(pxx[inn])] * 60.0) if np.any(inn) else 0.0
    except Exception:
        return 0.0


def _apnea_flag(sig: np.ndarray, fs: int,
                baseline_std: Optional[float] = None) -> bool:
    if len(sig) < fs * 10:
        return False
    ref_std = baseline_std if baseline_std is not None else float(np.std(sig))
    thr     = float(np.mean(sig)) - 1.5 * ref_std
    max_run = cur = 0
    for v in sig < thr:
        if v:
            cur     += 1
            max_run  = max(max_run, cur)
        else:
            cur = 0
    return max_run >= (10 * fs)


def _normalise(sig: np.ndarray) -> np.ndarray:
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-9)


def _get_apnea_label(annotation, seg_start_sample: int,
                     seg_end_sample: int, fs: int) -> int:
    if annotation is None:
        return 0
    for ann_sample, ann_symbol in zip(annotation.sample, annotation.symbol):
        if ann_symbol in ['A', 'APNEA']:
            ann_end = ann_sample + fs * 5
            if ann_sample < seg_end_sample and ann_end > seg_start_sample:
                return 1
    return 0


# ── Segment-level comparison ──────────────────────────────────────────────────

def _compare_segment(
    ecg_seg: np.ndarray,
    resp_gt: np.ndarray,
    record: str,
    seg_idx: int,
    baseline_std: Optional[float] = None,
    apnea_label: Optional[int] = None,
) -> Optional[Dict]:

    r_peaks = _detect_r_peaks(ecg_seg, FS_ECG)
    if len(r_peaks) < 8:
        return None

    edr = _compute_edr(ecg_seg, r_peaks, FS_ECG, FS_RESP)
    n   = min(len(edr), len(resp_gt))
    edr_seg = edr[:n]
    gt_seg  = resp_gt[:n]

    rr_edr = _resp_rate_welch(edr_seg, FS_RESP)
    rr_gt  = _resp_rate_welch(gt_seg,  FS_RESP)
    snr    = _compute_edr_snr(edr_seg, FS_RESP)

    edr_n = _normalise(edr_seg)
    gt_n  = _normalise(gt_seg)

    try:
        corr, _ = pearsonr(edr_n, gt_n)
    except Exception:
        corr = 0.0

    rmse = float(np.sqrt(np.mean((edr_n - gt_n) ** 2)))

    apnea_edr = _apnea_flag(edr_seg, FS_RESP, baseline_std)
    apnea_gt  = apnea_label if apnea_label is not None else _apnea_flag(gt_seg, FS_RESP, baseline_std)

    return {
        "record":           record,
        "segment_idx":      seg_idx,
        "rr_edr_bpm":       round(rr_edr, 2),
        "rr_gt_bpm":        round(rr_gt,  2),
        "rr_error_bpm":     round(rr_edr - rr_gt, 2),
        "rr_abs_error_bpm": round(abs(rr_edr - rr_gt), 2),
        "edr_snr":          round(snr, 4),
        "waveform_corr":    round(float(corr), 4),
        "waveform_rmse":    round(rmse, 4),
        "apnea_edr":        int(apnea_edr),
        "apnea_gt":         int(apnea_gt),
        "apnea_match":      int(apnea_edr == apnea_gt),
        "_edr_wave":        edr_n.tolist(),
        "_gt_wave":         gt_n.tolist(),
    }


# ── Per-record processing (runs in a thread) ──────────────────────────────────

def _process_record(record_name: str, max_segments: int,
                    use_cache: bool) -> List[Dict]:
    """Load, segment, and compare one SLPDB record. Thread-safe."""
    ecg_signal, resp_signal, annotation, fs_orig = _load_slpdb_record(
        record_name, use_cache=use_cache
    )
    if ecg_signal is None or resp_signal is None:
        return []

    ecg_resampled  = _resample_signal(ecg_signal,  fs_orig, FS_ECG)
    resp_resampled = _resample_signal(resp_signal, fs_orig, FS_RESP)

    samples_per_seg_ecg  = FS_ECG  * SEGMENT_LEN_S
    samples_per_seg_resp = FS_RESP * SEGMENT_LEN_S

    n_segs = min(
        len(ecg_resampled)  // samples_per_seg_ecg,
        len(resp_resampled) // samples_per_seg_resp,
        max_segments,
    )
    if n_segs == 0:
        logger.warning(f"{record_name}: not enough samples for one segment")
        return []

    # Baseline respiratory std from first 5 clean segments
    resp_stds = []
    for i in range(min(5, n_segs)):
        s = i * samples_per_seg_resp
        seg = resp_resampled[s: s + samples_per_seg_resp]
        if len(seg) == samples_per_seg_resp:
            resp_stds.append(float(np.std(seg)))
    baseline_resp_std = float(np.mean(resp_stds)) if resp_stds else None

    rec_results = []
    for i in range(n_segs):
        s_ecg  = i * samples_per_seg_ecg
        s_resp = i * samples_per_seg_resp

        ecg_seg  = _bandpass(ecg_resampled[s_ecg:  s_ecg  + samples_per_seg_ecg],  FS_ECG)
        resp_seg = resp_resampled[s_resp: s_resp + samples_per_seg_resp]

        apnea_label = (
            _get_apnea_label(annotation, s_resp, s_resp + samples_per_seg_resp, FS_RESP)
            if annotation else None
        )

        result = _compare_segment(ecg_seg, resp_seg, record_name, i,
                                  baseline_resp_std, apnea_label)
        if result:
            rec_results.append(result)

    logger.info(f"[VAL] {record_name}: {len(rec_results)} / {n_segs} segments")
    return rec_results


# ── Summary statistics ────────────────────────────────────────────────────────

def _compute_summary_stats(df: pd.DataFrame) -> Dict:
    errors = df["rr_error_bpm"].dropna()
    abs_e  = df["rr_abs_error_bpm"].dropna()
    corrs  = df["waveform_corr"].dropna()
    snrs   = df["edr_snr"].dropna()

    stats = {
        "n_segments":            len(df),
        "n_records":             df["record"].nunique(),
        "mae_bpm":               round(float(abs_e.mean()), 3),
        "rmse_bpm":              round(float(np.sqrt((errors**2).mean())), 3),
        "mean_error_bpm":        round(float(errors.mean()), 3),
        "std_error_bpm":         round(float(errors.std()), 3),
        "median_abs_error":      round(float(abs_e.median()), 3),
        "pct_within_2bpm":       round(100.0 * (abs_e <= 2.0).mean(), 1),
        "pct_within_4bpm":       round(100.0 * (abs_e <= 4.0).mean(), 1),
        "pct_within_6bpm":       round(100.0 * (abs_e <= 6.0).mean(), 1),
        "mean_waveform_corr":    round(float(corrs.mean()), 4),
        "median_waveform_corr":  round(float(corrs.median()), 4),
        "pct_corr_above_0.5":    round(100.0 * (corrs > 0.5).mean(), 1),
        "pct_corr_above_0.7":    round(100.0 * (corrs > 0.7).mean(), 1),
        "mean_edr_snr":          round(float(snrs.mean()), 4),
        "pct_snr_above_0.5":     round(100.0 * (snrs > 0.5).mean(), 1),
        "apnea_accuracy_pct":    round(100.0 * df["apnea_match"].mean(), 1),
        "apnea_sensitivity":     _apnea_sensitivity(df),
        "apnea_specificity":     _apnea_specificity(df),
    }

    df2 = df.copy()
    df2["snr_quartile"] = pd.qcut(df2["edr_snr"], 4,
                                  labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    stats["mae_by_snr_quartile"] = (
        df2.groupby("snr_quartile", observed=True)["rr_abs_error_bpm"]
        .mean().round(2).to_dict()
    )

    bins   = [0, 10, 15, 20, 25, 60]
    labels = ["<10", "10-15", "15-20", "20-25", ">25"]
    df2["rr_range"] = pd.cut(df2["rr_gt_bpm"], bins=bins, labels=labels)
    stats["mae_by_rr_range"] = (
        df2.groupby("rr_range", observed=True)["rr_abs_error_bpm"]
        .mean().round(2).to_dict()
    )
    return stats


def _apnea_sensitivity(df: pd.DataFrame) -> float:
    tp = ((df["apnea_edr"] == 1) & (df["apnea_gt"] == 1)).sum()
    fn = ((df["apnea_edr"] == 0) & (df["apnea_gt"] == 1)).sum()
    return round(100.0 * tp / max(tp + fn, 1), 1)


def _apnea_specificity(df: pd.DataFrame) -> float:
    tn = ((df["apnea_edr"] == 0) & (df["apnea_gt"] == 0)).sum()
    fp = ((df["apnea_edr"] == 1) & (df["apnea_gt"] == 0)).sum()
    return round(100.0 * tn / max(tn + fp, 1), 1)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_bland_altman(df: pd.DataFrame, out_dir: str) -> None:
    mean_rr = (df["rr_edr_bpm"] + df["rr_gt_bpm"]) / 2.0
    diff_rr = df["rr_edr_bpm"] - df["rr_gt_bpm"]
    md, sd  = diff_rr.mean(), diff_rr.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(mean_rr, diff_rr, c=df["edr_snr"], cmap="RdYlGn",
                    alpha=0.5, s=15, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="EDR SNR")
    ax.axhline(md, color="black", lw=1.5, label=f"Bias = {md:.2f} bpm")
    ax.axhline(md + 1.96 * sd, color="#d7301f", lw=1.2, linestyle="--",
               label=f"+1.96 SD = {md + 1.96 * sd:.2f}")
    ax.axhline(md - 1.96 * sd, color="#d7301f", lw=1.2, linestyle="--",
               label=f"-1.96 SD = {md - 1.96 * sd:.2f}")
    ax.fill_between([mean_rr.min(), mean_rr.max()],
                    md - 1.96 * sd, md + 1.96 * sd, alpha=0.07, color="#d7301f")
    ax.set_xlabel("Mean RR (bpm)")
    ax.set_ylabel("EDR − GT RR (bpm)")
    ax.set_title("Bland-Altman (colour = EDR SNR)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bland_altman.png"), dpi=150)
    plt.close(fig)
    logger.info("[PLOT] Bland-Altman saved")


def _plot_rr_scatter(df: pd.DataFrame, out_dir: str) -> None:
    lo = min(df["rr_gt_bpm"].min(), df["rr_edr_bpm"].min()) - 1
    hi = max(df["rr_gt_bpm"].max(), df["rr_edr_bpm"].max()) + 1
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df["rr_gt_bpm"], df["rr_edr_bpm"],
                    c=df["edr_snr"], cmap="RdYlGn",
                    alpha=0.4, s=14, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="EDR SNR")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Identity")
    ax.plot([lo, hi], [lo + 2, hi + 2], ":", lw=0.8, color="#d7301f")
    ax.plot([lo, hi], [lo - 2, hi - 2], ":", lw=0.8, color="#d7301f", label="±2 bpm")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("GT RR (bpm)"); ax.set_ylabel("EDR RR (bpm)")
    ax.set_title("EDR vs GT Respiratory Rate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rr_scatter.png"), dpi=150)
    plt.close(fig)
    logger.info("[PLOT] RR scatter saved")


def _plot_error_distribution(df: pd.DataFrame, out_dir: str) -> None:
    errors = df["rr_error_bpm"].dropna()
    abs_e  = df["rr_abs_error_bpm"].dropna().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(errors, bins=30, color="#2171b5", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="black", lw=1.5, linestyle="--")
    axes[0].axvline(errors.mean(), color="#d7301f", lw=1.5,
                    label=f"Mean = {errors.mean():.2f} bpm")
    axes[0].set_xlabel("EDR RR − GT RR (bpm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Signed Error Distribution")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    cdf = np.arange(1, len(abs_e) + 1) / len(abs_e)
    axes[1].plot(abs_e, cdf * 100, color="#2171b5", lw=2)
    for thr, col in [(2, "#41ab5d"), (4, "#fd8d3c"), (6, "#d7301f")]:
        pct = 100.0 * (abs_e <= thr).mean()
        axes[1].axvline(thr, color=col, lw=1.2, linestyle="--",
                        label=f"≤{thr} bpm: {pct:.0f}%")
    axes[1].set_xlabel("|Error| (bpm)")
    axes[1].set_ylabel("Cumulative %")
    axes[1].set_title("Absolute Error CDF")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_distribution.png"), dpi=150)
    plt.close(fig)
    logger.info("[PLOT] Error distribution saved")


def _plot_snr_vs_error(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["edr_snr"], df["rr_abs_error_bpm"],
               alpha=0.3, s=12, color="#2171b5")
    df_s = df.sort_values("edr_snr")
    roll = df_s["rr_abs_error_bpm"].rolling(30, center=True, min_periods=5).mean()
    ax.plot(df_s["edr_snr"], roll, color="#d7301f", lw=2, label="Rolling mean")
    ax.axvline(0.3, color="#fd8d3c", lw=1.2, linestyle="--", label="SNR=0.3 gate")
    ax.axhline(4,   color="#41ab5d", lw=1.0, linestyle=":", label="4 bpm threshold")
    ax.set_xlabel("EDR In-band SNR")
    ax.set_ylabel("|RR Error| (bpm)")
    ax.set_title("EDR SNR vs RR Error")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snr_vs_error.png"), dpi=150)
    plt.close(fig)
    logger.info("[PLOT] SNR vs error saved")


def _plot_per_record_mae(df: pd.DataFrame, out_dir: str) -> None:
    rec = (df.groupby("record")["rr_abs_error_bpm"]
           .agg(["mean", "count"])
           .rename(columns={"mean": "mae", "count": "n"})
           .sort_values("mae"))
    fig, ax = plt.subplots(figsize=(max(6, len(rec) * 0.8 + 2), 5))
    colors = ["#41ab5d" if m <= 2 else "#fd8d3c" if m <= 4 else "#d7301f"
              for m in rec["mae"]]
    bars = ax.bar(range(len(rec)), rec["mae"], color=colors, edgecolor="white", width=0.7)
    ax.axhline(2, color="#41ab5d", lw=1.2, linestyle="--", label="2 bpm target")
    ax.axhline(4, color="#fd8d3c", lw=1.2, linestyle="--", label="4 bpm warning")
    ax.set_xticks(range(len(rec)))
    ax.set_xticklabels([f"{r}\n(n={n})" for r, n in zip(rec.index, rec["n"])],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("EDR Accuracy per SLPDB Record")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, rec["mae"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_record_mae.png"), dpi=150)
    plt.close(fig)
    logger.info("[PLOT] Per-record MAE saved")


def _plot_correlation_distribution(df: pd.DataFrame, out_dir: str) -> None:
    corrs = df["waveform_corr"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corrs, bins=30, color="#6baed6", edgecolor="white", alpha=0.85)
    ax.axvline(corrs.median(), color="#d7301f", lw=1.5,
               label=f"Median r = {corrs.median():.3f}")
    ax.axvline(0.5, color="#fd8d3c", lw=1.2, linestyle="--", label="r = 0.50")
    ax.axvline(0.7, color="#41ab5d", lw=1.2, linestyle="--", label="r = 0.70")
    ax.set_xlabel("Pearson r (EDR vs GT Resp)")
    ax.set_ylabel("Count")
    ax.set_title("Waveform Correlation Distribution")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "waveform_correlation.png"), dpi=150)
    plt.close(fig)
    logger.info("[PLOT] Correlation distribution saved")


def _plot_waveform_samples(df: pd.DataFrame, out_dir: str, n_samples: int = 6) -> None:
    df_v = df[df["edr_snr"].notna()].copy().sort_values("edr_snr").reset_index(drop=True)
    if len(df_v) == 0:
        return

    n   = len(df_v)
    n3  = max(1, n_samples // 3)
    idx = list(set(
        df_v.head(n3).index.tolist() +
        df_v.iloc[n // 2 - n3 // 2: n // 2 + n3 // 2].index.tolist() +
        df_v.tail(n3).index.tolist()
    ))[:n_samples]

    rows    = [df_v.iloc[i] for i in idx]
    n_plots = len(rows)
    fig, axes = plt.subplots(n_plots, 1, figsize=(11, 2.8 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    t = np.arange(SEGMENT_LEN_S * FS_RESP) / FS_RESP
    for ax, row in zip(axes, rows):
        edr_w = np.array(row.get("_edr_wave", []))
        gt_w  = np.array(row.get("_gt_wave",  []))
        if len(edr_w) == 0 or len(gt_w) == 0:
            continue
        n_pts = min(len(t), len(edr_w), len(gt_w))
        ax.plot(t[:n_pts], gt_w[:n_pts],  color="#2ca02c", lw=1.4, alpha=0.85,
                label=f"GT Resp  {row.get('rr_gt_bpm', 0):.1f} bpm")
        ax.plot(t[:n_pts], edr_w[:n_pts], color="#d62728", lw=1.2, linestyle="--",
                alpha=0.85,
                label=f"EDR  {row.get('rr_edr_bpm', 0):.1f} bpm  "
                      f"SNR={row.get('edr_snr', 0):.2f}  "
                      f"r={row.get('waveform_corr', 0):.2f}")
        ax.set_title(f"{row.get('record', '')} seg={int(row.get('segment_idx', 0))}  "
                     f"|err|={row.get('rr_abs_error_bpm', 0):.1f} bpm", fontsize=9)
        ax.set_ylabel("Norm"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("EDR vs GT Resp Waveform (sorted by SNR)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "waveform_sample.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    logger.info("[PLOT] Waveform overlay saved")


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(stats: Dict, df: pd.DataFrame, out_dir: str) -> None:
    lines = [
        "=" * 62,
        "  EDR VALIDATION REPORT (SLPDB - Sleep Apnea Database)",
        "=" * 62,
        "",
        f"  Segments analysed  : {stats['n_segments']}",
        f"  Records            : {stats['n_records']}",
        "",
        "  ── Respiratory Rate Accuracy ────────────────────────────",
        f"  MAE                : {stats['mae_bpm']} bpm",
        f"  RMSE               : {stats['rmse_bpm']} bpm",
        f"  Mean signed error  : {stats['mean_error_bpm']} bpm  "
        f"({'EDR overestimates' if stats['mean_error_bpm'] > 0 else 'EDR underestimates'})",
        f"  Std of error       : {stats['std_error_bpm']} bpm",
        f"  Median |error|     : {stats['median_abs_error']} bpm",
        "",
        f"  Within ±2 bpm      : {stats['pct_within_2bpm']}%",
        f"  Within ±4 bpm      : {stats['pct_within_4bpm']}%",
        f"  Within ±6 bpm      : {stats['pct_within_6bpm']}%",
        "",
        "  ── EDR Signal Quality (SNR) ─────────────────────────────",
        f"  Mean in-band SNR   : {stats['mean_edr_snr']}",
        f"  SNR > 0.5          : {stats['pct_snr_above_0.5']}%",
        "",
        "  MAE by SNR quartile:",
    ]
    for q, mae in stats["mae_by_snr_quartile"].items():
        lines.append(f"    {q:<12} → MAE = {mae} bpm")

    lines += ["", "  MAE by GT respiratory rate range:"]
    for rng, mae in stats["mae_by_rr_range"].items():
        lines.append(f"    {rng:<10} bpm  → MAE = {mae} bpm")

    lines += [
        "",
        "  ── Waveform Shape Agreement ─────────────────────────────",
        f"  Mean Pearson r     : {stats['mean_waveform_corr']}",
        f"  Median Pearson r   : {stats['median_waveform_corr']}",
        f"  Corr > 0.50        : {stats['pct_corr_above_0.5']}%",
        f"  Corr > 0.70        : {stats['pct_corr_above_0.7']}%",
        "",
        "  ── Apnea Detection Agreement (vs Expert Annotations) ────",
        f"  Accuracy           : {stats['apnea_accuracy_pct']}%",
        f"  Sensitivity        : {stats['apnea_sensitivity']}%",
        f"  Specificity        : {stats['apnea_specificity']}%",
        "",
        "  ── Interpretation ───────────────────────────────────────",
        "  MAE ≤ 2 bpm : Excellent | MAE ≤ 4 bpm : Acceptable | MAE > 4 bpm : Poor",
        "=" * 62,
    ]

    report = "\n".join(lines)
    logger.info("\n%s", report)
    with open(os.path.join(out_dir, "summary_report.txt"), "w") as f:
        f.write(report + "\n")
    logger.info("[REPORT] Saved")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_validation_slpdb(
    out_dir:                 str       = "edr_validation_slpdb",
    records:                 List[str] = None,
    max_segments_per_record: int       = 20,
    workers:                 int       = 6,
    use_cache:               bool      = True,
) -> None:
    """
    Run EDR validation on SLPDB.

    Args:
        out_dir:                 Output directory for results.
        records:                 Record names to process (default: all 18).
        max_segments_per_record: Maximum 30-second segments per record.
        workers:                 Thread-pool size for parallel loading.
        use_cache:               Cache records to ~/.cache/slpdb for fast re-runs.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not HAS_WFDB:
        logger.error("wfdb not installed — run: pip install wfdb")
        return
    if not HAS_MPL:
        logger.error("matplotlib not installed — run: pip install matplotlib")
        return

    if records is None:
        records = SLPDB_RECORDS

    logger.info(f"Processing {len(records)} records  |  workers={workers}  |  "
                f"cache={'on' if use_cache else 'off'}  |  "
                f"max_segments={max_segments_per_record}")

    all_results: List[Dict] = []

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {
            pool.submit(_process_record, rec, max_segments_per_record, use_cache): rec
            for rec in records
        }
        for fut in futures.as_completed(fut_map):
            rec = fut_map[fut]
            try:
                results = fut.result()
                all_results.extend(results)
            except Exception as exc:
                logger.error(f"[FAIL] {rec}: {exc}")

    if not all_results:
        logger.error("No segments compared — check connectivity / record names")
        return

    n_recs = len(set(r["record"] for r in all_results))
    logger.info(f"\n[VAL] Total: {len(all_results)} segments across {n_recs} records")

    df_full = pd.DataFrame(all_results)
    df_csv  = df_full.drop(columns=["_edr_wave", "_gt_wave"], errors="ignore")
    df_csv.to_csv(os.path.join(out_dir, "edr_validation_segments.csv"), index=False)

    stats = _compute_summary_stats(df_csv)

    _plot_bland_altman(df_csv, out_dir)
    _plot_rr_scatter(df_csv, out_dir)
    _plot_error_distribution(df_csv, out_dir)
    _plot_snr_vs_error(df_csv, out_dir)
    _plot_per_record_mae(df_csv, out_dir)
    _plot_correlation_distribution(df_csv, out_dir)
    _plot_waveform_samples(df_full, out_dir)
    _write_report(stats, df_csv, out_dir)

    logger.info(f"\n[VAL] Done! Results saved to {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Validate EDR against SLPDB (MIT-BIH Polysomnographic Database)"
    )
    p.add_argument("--out-dir",      default="edr_validation_slpdb",
                   help="Output directory for results")
    p.add_argument("--records",      nargs="+", default=None,
                   help="Specific records to process (default: all 18)")
    p.add_argument("--max-segments", type=int, default=20,
                   help="Max 30-second segments per record (use 5 for quick tests)")
    p.add_argument("--workers",      type=int, default=6,
                   help="Parallel workers for record loading (default: 6)")
    p.add_argument("--no-cache",     action="store_true",
                   help="Disable local caching; always stream from PhysioNet")
    return p.parse_args()


def main():
    args = _parse_args()
    run_validation_slpdb(
        out_dir=args.out_dir,
        records=args.records,
        max_segments_per_record=args.max_segments,
        workers=args.workers,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()

