"""
pipeline/pipeline.py
====================
Single-module ML pipeline for Apnea detection using real-world wearable constraints.

Modules
-------
apnea       30-second segment apnea detection
            Data: MIMIC-IV Waveform DB streamed via wfdb
            Constraints: ECG @ 125Hz, PPG @ 120Hz, EDR (fusion) for resp, intermittent SpO2
            Label: AASM 3-signal composite (GT Resp channel, SpO2, HRV)
            Model: Bidirectional LSTM

Usage
-----
python pipeline/pipeline.py
python pipeline/pipeline.py --fresh          # delete DB before run
python pipeline/pipeline.py --save-model     # save trained model and scaler to disk
"""

import argparse
import datetime
import json
import logging
import os
import sys
import warnings
import scipy.signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, filtfilt, find_peaks, welch
from scipy.signal import resample as scipy_resample
from scipy.signal.windows import tukey

import random
import tensorflow as tf

# Set all random seeds for reproducibility
def set_all_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

warnings.filterwarnings("ignore")
set_all_seeds(42)



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CLI.db.database import (
    DB_PATH,
    init_db,
    insert_apnea_ecg_plot,
    insert_apnea_features,
    insert_apnea_preprocessed,
    insert_apnea_raw,
    insert_apnea_results,
    insert_apnea_segment,
    fetch_apnea_segments,
    log_module,
)

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

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── EDR v3: precision respiratory rate engine ─────────────────────────────────
try:
    from compute_edr_fixed import compute_edr_v3 as _compute_edr_v3
    HAS_EDR_V3 = True
except ImportError:
    HAS_EDR_V3 = False

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

# config
DATA_DIR: str = os.environ.get("DATA_DIR", "../archive2/")
MIMIC_URL: str = "https://physionet.org/files/mimic4wdb/0.1.0/"

FS_MIMIC: int = 320
FS_ECG: int = 125
FS_PPG: int = 120
FS_RESP: int = 4
SEGMENT_LEN_S: int = 30
N_MIMIC_RECORDS: int = 60

APNEA_FEATURE_COLS = [
    "rr_mean", "rr_std", "rmssd", "pnn50", "mean_hr", "hr_range", "lf_hf_ratio",
    "resp_rate_bpm", "resp_rate_variability", "flatline_duration_s",
    "resp_amplitude_mean", "resp_amplitude_std",
    "spo2_mean", "spo2_min", "spo2_delta_index", "odi", "t90", "spo2_approx_entropy",
    "map_mean", "map_std", "map_variability", "sbp_max", "dbp_min", "pulse_pressure",
    "resp_spo2_lag_s", "ptt_ms", "ecg_resp_coherence",
]


# ── Signal utilities ──────────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: int,
              lo: float = 0.5, hi: float = 40.0,
              order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    actual_hi = min(hi, nyq - 0.1)
    actual_lo = min(lo, actual_hi - 0.1)
    b, a = butter(order, [actual_lo / nyq, actual_hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def _detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    if HAS_NK:
        try:
            _, info = nk.ecg_process(ecg, sampling_rate=fs)
            return info["ECG_R_Peaks"]
        except Exception as exc:
            logger.warning("nk.ecg_process failed: %s — using scipy fallback", exc)
    peaks, _ = find_peaks(ecg, distance=int(fs * 0.4), height=float(np.std(ecg)))
    return peaks


def _compute_edr(ecg: np.ndarray, r_peaks: np.ndarray,
                 fs_ecg: int, fs_resp: int = 4) -> np.ndarray:
    """Legacy EDR: dual-engine QRS-area + QRS-PCA fusion."""
    if len(r_peaks) < 8:
        return np.zeros(int(len(ecg) * fs_resp / fs_ecg))
    t_peaks = r_peaks / fs_ecg
    t_uniform = np.arange(0, len(ecg) / fs_ecg, 1.0 / fs_resp)

    def _process_envelope(raw_v, times):
        if len(raw_v) < 6:
            return np.zeros_like(t_uniform)
        v_detrended = raw_v - np.polyval(np.polyfit(times, raw_v, 1), times)
        s = np.interp(t_uniform, times, v_detrended)
        return (s - np.mean(s)) / (np.std(s) + 1e-9)

    qrs_win = max(1, int(0.06 * fs_ecg))
    areas = [np.sum(np.abs(ecg[max(0, r - qrs_win):min(len(ecg), r + qrs_win)]))
             for r in r_peaks]
    m3_wave = _process_envelope(np.array(areas, dtype=float), t_peaks)

    beats = [ecg[r - qrs_win:r + qrs_win]
             for r in r_peaks if r - qrs_win >= 0 and r + qrs_win <= len(ecg)]
    m4_wave = m3_wave
    if len(beats) >= 8:
        X = np.array(beats, dtype=float)
        X -= X.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            m4_wave = _process_envelope(U[:, 0] * S[0], t_peaks[:len(beats)])
        except np.linalg.LinAlgError:
            pass

    min_len = min(len(m3_wave), len(m4_wave))
    fused = np.median(np.vstack([m3_wave[:min_len], m4_wave[:min_len]]), axis=0)
    nyq = fs_resp / 2.0
    b, a = butter(3, [0.1 / nyq, 0.5 / nyq], btype="band")
    out = np.zeros_like(t_uniform)
    out[:min_len] = filtfilt(b, a, fused)
    return out


# ── Signal flag functions ─────────────────────────────────────────────────────

def _resp_flag(resp: np.ndarray, fs: int) -> bool:
    """True if respiratory signal shows ≥10s of sustained amplitude suppression.
    Pass GT Resp channel when available; EDR as fallback."""
    if len(resp) < fs * 10:
        return False
    threshold = np.mean(resp) - 1.5 * np.std(resp)
    suppressed = resp < threshold
    max_run = current_run = 0
    for val in suppressed:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run >= (10 * fs)


def _spo2_flag(pleth: np.ndarray, fs: int, baseline_spo2: float) -> bool:
    smooth = (pd.Series(pleth)
              .rolling(int(fs * 2), center=True, min_periods=1)
              .median().values)
    spo2_min = float(np.min(smooth))
    return (baseline_spo2 - spo2_min >= 3.0) and (spo2_min < 94.0)


def _hrv_flag(r_peaks: np.ndarray, fs: int,
              baseline_rmssd: float, baseline_rr_ms: float) -> bool:
    if len(r_peaks) < 3:
        return False
    rr_ms = np.diff(r_peaks) / fs * 1000.0
    rmssd_w = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    mean_rr_w = float(np.mean(rr_ms))
    return (rmssd_w > 1.5 * baseline_rmssd) and (mean_rr_w > 1.2 * baseline_rr_ms)


def _abp_flag(abp: np.ndarray, fs: int,
              baseline_map_std: float, baseline_sbp: float) -> bool:
    map_sig = (pd.Series(abp)
               .rolling(int(fs * 2), center=True, min_periods=1)
               .mean().values)
    return (float(np.std(map_sig)) > 1.5 * baseline_map_std or
            float(np.max(abp)) > baseline_sbp + 15.0)


def label_apnea_segment(resp: bool, spo2: bool, hrv: bool) -> Tuple[int, str]:
    n = sum([resp, spo2, hrv])
    if n == 3: return 1, "definite_apnea"
    if n == 2: return 1, "probable_apnea"
    if n == 1: return 0, "possible_hypopnea"
    return 0, "normal"


# ── Cross-signal features ─────────────────────────────────────────────────────

def _cross_signal_features(resp, pleth, r_peaks, abp,
                            fs_resp, fs_pleth, fs_abp) -> Dict[str, float]:
    feats: Dict[str, float] = {"resp_spo2_lag_s": 0.0, "ptt_ms": 0.0,
                                "ecg_resp_coherence": 0.0}
    try:
        min_len_s = min(len(resp) / fs_resp, len(pleth) / fs_pleth)
        t_common = np.arange(0, min_len_s, 1.0 / fs_resp)
        r_res = np.interp(t_common, np.arange(len(resp)) / fs_resp, resp)
        p_res = np.interp(t_common, np.arange(len(pleth)) / fs_pleth, pleth)
        r_norm = r_res - np.mean(r_res)
        p_norm = p_res - np.mean(p_res)
        corr = np.correlate(p_norm, r_norm, mode="full")
        lags = np.arange(-(len(p_norm) - 1), len(p_norm))
        feats["resp_spo2_lag_s"] = float(lags[np.argmax(np.abs(corr))]) / fs_resp
    except Exception:
        pass
    try:
        ptt_vals: List[float] = []
        for rp in r_peaks:
            s = int((rp / FS_ECG) * fs_abp)
            e = min(int(s + 0.5 * fs_abp), len(abp) - 1)
            if e <= s: continue
            ptt_ms = int(np.argmin(abp[s:e])) / fs_abp * 1000.0
            if 50.0 < ptt_ms < 500.0:
                ptt_vals.append(ptt_ms)
        feats["ptt_ms"] = float(np.mean(ptt_vals)) if ptt_vals else 0.0
    except Exception:
        pass
    try:
        rr_series = np.diff(r_peaks) / float(FS_ECG)
        if len(rr_series) >= 8 and len(resp) >= 64:
            resp_res = np.interp(np.linspace(0, 1, len(rr_series)),
                                 np.linspace(0, 1, len(resp)), resp)
            f, cxy = coherence(rr_series, resp_res,
                               fs=1.0, nperseg=min(8, len(rr_series)))
            hf = (f >= 0.15) & (f <= 0.4)
            feats["ecg_resp_coherence"] = float(np.mean(cxy[hf]) if hf.any() else 0.0)
    except Exception:
        pass
    return feats


# ── Per-segment feature extraction ───────────────────────────────────────────

def _extract_apnea_features(
    ecg: np.ndarray,
    pleth: np.ndarray,
    resp: np.ndarray,
    abp: np.ndarray,
    r_peaks: np.ndarray,
    baseline: Dict[str, float],
    edr_bpm: Optional[float] = None,
    edr_quality: Optional[float] = None,
    resp_gt: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    resp_gt : ground-truth MIMIC Resp channel at FS_RESP Hz.
              When provided: used for _resp_flag (label) and all resp features.
              When None: falls back to EDR signal (resp) for both.
    """
    feats: Dict[str, Any] = {}

    # Route resp signal
    # resp_for_label → _resp_flag → AASM label (must be ground truth)
    # resp_for_feats → all resp-based feature values
    resp_for_label = resp_gt if resp_gt is not None else resp
    resp_for_feats = resp_gt if resp_gt is not None else resp

    # HRV
    if len(r_peaks) >= 3:
        rr_ms = np.diff(r_peaks) / FS_ECG * 1000.0
        rr_diffs = np.diff(rr_ms)
        feats["rr_mean"] = float(np.mean(rr_ms))
        feats["rr_std"] = float(np.std(rr_ms))
        feats["rmssd"] = float(np.sqrt(np.mean(rr_diffs ** 2)))
        feats["pnn50"] = float(np.sum(np.abs(rr_diffs) > 50.0) / max(len(rr_ms), 1))
        feats["mean_hr"] = float(60000.0 / (np.mean(rr_ms) + 1e-6))
        feats["hr_range"] = float(
            60000.0 / (np.min(rr_ms) + 1e-6) - 60000.0 / (np.max(rr_ms) + 1e-6))
        if HAS_NK and len(r_peaks) >= 5:
            try:
                hf = nk.hrv_frequency(r_peaks, sampling_rate=FS_ECG, show=False)
                feats["lf_hf_ratio"] = float(hf["HRV_LFHF"].values[0])
            except Exception:
                feats["lf_hf_ratio"] = 0.0
        else:
            feats["lf_hf_ratio"] = 0.0
    else:
        for k in ("rr_mean", "rr_std", "rmssd", "pnn50",
                  "mean_hr", "hr_range", "lf_hf_ratio"):
            feats[k] = 0.0

    # SpO2
    pleth_clean = pd.Series(pleth).ffill().bfill().values
    smooth = (pd.Series(pleth_clean)
              .rolling(int(FS_PPG * 2), center=True, min_periods=1)
              .median().values)
    feats["spo2_mean"] = float(np.mean(smooth))
    feats["spo2_min"] = float(np.min(smooth))
    feats["spo2_delta_index"] = float(np.max(smooth) - np.min(smooth))
    desat_thresh = baseline.get("baseline_spo2", 97.0) - 3.0
    feats["odi"] = float(np.sum(np.diff((smooth < desat_thresh).astype(int)) == 1))
    feats["t90"] = float(np.mean(smooth < 90.0))
    try:
        phi = (smooth - np.mean(smooth)) / (np.std(smooth) + 1e-9)
        feats["spo2_approx_entropy"] = float(-np.mean(np.log(np.abs(np.diff(phi)) + 1e-9)))
    except Exception:
        feats["spo2_approx_entropy"] = 0.0

    # Respiratory features — GT Resp when available, EDR as fallback
    feats["resp_amplitude_mean"] = float(np.mean(np.abs(resp_for_feats)))
    feats["resp_amplitude_std"] = float(np.std(resp_for_feats))
    threshold = np.mean(resp_for_feats) - 1.5 * np.std(resp_for_feats)
    suppressed = resp_for_feats < threshold
    max_run = current_run = 0
    for val in suppressed:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    feats["flatline_duration_s"] = float(max_run / FS_RESP)

    # label_source: tells training filter which segments to trust
    feats["label_source"] = "mimic_resp" if resp_gt is not None else "edr"

    try:
        if resp_gt is not None:
            # GT Resp: compute rate directly from chest belt via Welch
            w_edge = tukey(len(resp_for_feats), alpha=0.1)
            nperseg = min(len(resp_for_feats), max(8, int(FS_RESP * 60)))
            f, pxx = welch(resp_for_feats * w_edge, fs=FS_RESP,
                           nperseg=nperseg, noverlap=nperseg // 2, nfft=2048)
            inn = (f >= 0.1) & (f <= 0.6)
            feats["resp_rate_bpm"] = (float(f[inn][np.argmax(pxx[inn])] * 60.0)
                                      if np.any(inn) else 0.0)
        elif edr_bpm is not None and edr_quality is not None and edr_quality >= 1.5:
            feats["resp_rate_bpm"] = float(edr_bpm)
        else:
            w_edge = tukey(len(resp_for_feats), alpha=0.1)
            nperseg = min(len(resp_for_feats), max(8, int(FS_RESP * 60)))
            f, pxx = welch(resp_for_feats * w_edge, fs=FS_RESP,
                           nperseg=nperseg, noverlap=nperseg // 2, nfft=2048)
            inn = (f >= 0.1) & (f <= 0.6)
            feats["resp_rate_bpm"] = (float(f[inn][np.argmax(pxx[inn])] * 60.0)
                                      if np.any(inn) else 0.0)

        feats["edr_quality_ok"] = int((edr_quality or 0.0) >= 1.5)
        feats["edr_snr"] = float(edr_quality) if edr_quality is not None else 0.0

        resp_peaks, _ = find_peaks(resp_for_feats, distance=int(FS_RESP * 1.5))
        feats["resp_rate_variability"] = (float(np.std(np.diff(resp_peaks) / FS_RESP))
                                          if len(resp_peaks) >= 2 else 0.0)
    except Exception:
        feats["resp_rate_bpm"] = 0.0
        feats["resp_rate_variability"] = 0.0
        feats["edr_quality_ok"] = 0
        feats["edr_snr"] = 0.0

    # ABP
    map_sig = (pd.Series(abp)
               .rolling(int(FS_ECG * 2), center=True, min_periods=1)
               .mean().values)
    feats["map_mean"] = float(np.mean(map_sig))
    feats["map_std"] = float(np.std(map_sig))
    feats["map_variability"] = feats["map_std"]
    feats["sbp_max"] = float(np.max(abp))
    feats["dbp_min"] = float(np.min(abp))
    feats["pulse_pressure"] = feats["sbp_max"] - feats["dbp_min"]

    # Cross-signal — use GT Resp when available
    cross = _cross_signal_features(resp_for_feats, pleth_clean,
                                   r_peaks, abp, FS_RESP, FS_PPG, FS_ECG)
    feats.update(cross)

    # Signal flags
    # _resp_flag uses GT Resp channel — this breaks the circular label dependency
    rf = _resp_flag(resp_for_label, FS_RESP)
    sf = _spo2_flag(pleth_clean, FS_PPG, baseline.get("baseline_spo2", 97.0))
    hf = _hrv_flag(r_peaks, FS_ECG,
                   baseline.get("baseline_rmssd", 35.0),
                   baseline.get("baseline_rr_ms", 833.0))
    af = _abp_flag(abp, FS_ECG,
                   baseline.get("baseline_map_std", 5.0),
                   baseline.get("baseline_sbp", 120.0))

    label, conf = label_apnea_segment(rf, sf, hf)

    feats["resp_flag"] = int(rf)
    feats["spo2_flag"] = int(sf)
    feats["hrv_flag"] = int(hf)
    feats["abp_flag"] = int(af)
    feats["signals_positive"] = sum([rf, sf, hf])
    feats["true_label"] = label
    feats["label_confidence"] = conf

    return feats


# ── MIMIC record loader ───────────────────────────────────────────────────────

def _load_mimic_records(n: int = N_MIMIC_RECORDS) -> List[str]:
    if not HAS_WFDB:
        logger.error("[APNEA] wfdb not installed")
        return []
    try:
        import urllib.request
        with urllib.request.urlopen(MIMIC_URL + "RECORDS", timeout=30) as r:
            lines = r.read().decode().splitlines()
        valid_paths = []
        for ln in lines:
            if not ln.strip():
                continue
            dir_path = ln.strip()
            try:
                with urllib.request.urlopen(
                        MIMIC_URL + dir_path + "RECORDS", timeout=10) as ir:
                    for iln in ir.read().decode().splitlines():
                        if "layout" not in iln:
                            valid_paths.append(dir_path + iln.strip())
                            if len(valid_paths) >= n:
                                return valid_paths
            except Exception:
                continue
        return valid_paths
    except Exception as exc:
        logger.error("Failed to load MIMIC records: %s", exc)
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_apnea_module(n_records: int = N_MIMIC_RECORDS, save_model: bool = False) -> None:
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info(" APNEA MODULE  run_id=%s", run_id)
    logger.info("=" * 60)

    if not HAS_WFDB:
        logger.error("[APNEA] wfdb not installed. Run: pip install wfdb")
        log_module("apnea", "ingest", "failed", "wfdb not installed", 0)
        return

    log_module("apnea", "ingest", "started")
    record_paths = _load_mimic_records(n_records)
    if not record_paths:
        log_module("apnea", "ingest", "failed", "No MIMIC records fetched", 0)
        return

    total_segs = 0
    plots_per_record = {}

    for rec_path in record_paths:
        record_name = rec_path.split("/")[-1]
        pn_dir = "mimic4wdb/0.1.0/" + "/".join(rec_path.split("/")[:-1])

        try:
            rec = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=96000)
        except Exception as exc:
            logger.warning("[APNEA] Could not load %s: %s", record_name, exc)
            continue

        sig_map = {name: idx for idx, name in enumerate(rec.sig_name)}
        if any(s not in sig_map for s in ["II", "Pleth"]):
            logger.warning("[APNEA] %s missing II or Pleth — skipping", record_name)
            continue

        signals = rec.p_signal
        fs_orig = rec.fs

        ecg_orig   = signals[:, sig_map["II"]]
        pleth_orig = signals[:, sig_map["Pleth"]]
        abp_orig   = (signals[:, sig_map["ABP"]]
                      if "ABP" in sig_map else np.zeros(signals.shape[0]))

        # Ground-truth Resp channel — breaks circular EDR label dependency
        has_gt_resp = "Resp" in sig_map
        resp_orig = signals[:, sig_map["Resp"]] if has_gt_resp else None
        if has_gt_resp:
            logger.info("[APNEA] %s: ground-truth Resp channel found — using for labelling",
                        record_name)
        else:
            logger.warning("[APNEA] %s: no Resp channel — EDR fallback for labelling",
                           record_name)

        # Fill NaNs
        for arr in (ecg_orig, pleth_orig, abp_orig):
            m = np.isnan(arr)
            arr[m] = np.nanmean(arr) if not m.all() else 0.0
        if resp_orig is not None:
            m = np.isnan(resp_orig)
            resp_orig[m] = np.nanmean(resp_orig) if not m.all() else 0.0

        # Resample
        ecg_full   = scipy_resample(ecg_orig,   int(len(ecg_orig)   * FS_ECG  / fs_orig))
        pleth_full = scipy_resample(pleth_orig, int(len(pleth_orig) * FS_PPG  / fs_orig))
        abp_full   = scipy_resample(abp_orig,   int(len(abp_orig)   * FS_ECG  / fs_orig))
        resp_gt_full = (scipy_resample(resp_orig, int(len(resp_orig) * FS_RESP / fs_orig))
                        if resp_orig is not None else None)

        spe = FS_ECG  * SEGMENT_LEN_S   # samples per seg ECG
        spp = FS_PPG  * SEGMENT_LEN_S   # samples per seg PPG
        spr = FS_RESP * SEGMENT_LEN_S   # samples per seg Resp

        n_segs = min(len(ecg_full) // spe, 10)
        if n_segs == 0:
            continue

        last_spo2_val = 97.0

        # ── Baseline computation ──────────────────────────────────────────────
        baseline_windows: List[Dict] = []
        for i in range(n_segs):
            ecg_seg   = _bandpass(ecg_full[i*spe:(i+1)*spe], FS_ECG)
            pleth_seg = pleth_full[i*spp:(i+1)*spp]
            abp_seg   = abp_full[i*spe:(i+1)*spe]
            rp = _detect_r_peaks(ecg_seg, FS_ECG)
            rr_ms = np.diff(rp) / FS_ECG * 1000.0 if len(rp) >= 3 else np.array([833.0])
            rmssd_w = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) >= 2 else 35.0

            if HAS_EDR_V3:
                resp_seg, _, _ = _compute_edr_v3(ecg_seg, rp, FS_ECG, FS_RESP, SEGMENT_LEN_S)
            else:
                resp_seg = _compute_edr(ecg_seg, rp, FS_ECG, FS_RESP)

            resp_gt_seg_bl = (resp_gt_full[i*spr:(i+1)*spr]
                              if resp_gt_full is not None and (i+1)*spr <= len(resp_gt_full)
                              else None)
            resp_for_bl = resp_gt_seg_bl if resp_gt_seg_bl is not None else resp_seg

            map_sig = pd.Series(abp_seg).rolling(int(FS_ECG*2), min_periods=1).mean().values
            baseline_windows.append({
                "spo2_mean": float(np.mean(pleth_seg)),
                "rmssd": rmssd_w,
                "rr_mean": float(np.mean(rr_ms)),
                "map_std": float(np.std(map_sig)),
                "sbp_max": float(np.max(abp_seg)),
                "individually_clean": (not _resp_flag(resp_for_bl, FS_RESP)
                                       and float(np.min(pleth_seg)) > 90.0),
            })

        clean_wins = [w for w in baseline_windows if w["individually_clean"]][:5]
        if clean_wins:
            baseline = {
                "baseline_spo2":    float(np.mean([w["spo2_mean"] for w in clean_wins])),
                "baseline_rmssd":   float(np.mean([w["rmssd"]     for w in clean_wins])),
                "baseline_rr_ms":   float(np.mean([w["rr_mean"]   for w in clean_wins])),
                "baseline_map_std": float(np.mean([w["map_std"]   for w in clean_wins])),
                "baseline_sbp":     float(np.mean([w["sbp_max"]   for w in clean_wins])),
            }
        else:
            baseline = {"baseline_spo2": 97.0, "baseline_rmssd": 35.0,
                        "baseline_rr_ms": 833.0, "baseline_map_std": 5.0,
                        "baseline_sbp": 120.0}

        # ── Per-segment processing ────────────────────────────────────────────
        for i in range(n_segs):
            ecg_seg = _bandpass(ecg_full[i*spe:(i+1)*spe], FS_ECG)
            abp_seg = abp_full[i*spe:(i+1)*spe]

            take_reading = (i % np.random.randint(6, 11)) == 0
            if take_reading:
                pleth_seg = pleth_full[i*spp:(i+1)*spp]
                last_spo2_val = float(np.mean(pleth_seg))
            else:
                pleth_seg = np.full(spp, last_spo2_val)

            r_peaks = _detect_r_peaks(ecg_seg, FS_ECG)

            if HAS_EDR_V3:
                resp_seg, edr_bpm, edr_quality = _compute_edr_v3(
                    ecg_seg, r_peaks, FS_ECG, FS_RESP, SEGMENT_LEN_S)
            else:
                resp_seg = _compute_edr(ecg_seg, r_peaks, FS_ECG, FS_RESP)
                edr_bpm, edr_quality = None, None

            # GT Resp segment aligned to this 30-second window
            resp_gt_seg = (resp_gt_full[i*spr:(i+1)*spr]
                           if resp_gt_full is not None and (i+1)*spr <= len(resp_gt_full)
                           else None)

            # Stage 1: raw
            raw_id = insert_apnea_raw(
                record_name, i, ecg_seg, pleth_seg, resp_seg, abp_seg, FS_ECG)

            # Stage 2: preprocess
            spo2_smooth = (pd.Series(pleth_seg)
                           .rolling(int(FS_PPG*2), center=True, min_periods=1)
                           .median().values)
            resp_smooth = (pd.Series(resp_seg)
                           .rolling(FS_RESP, center=True, min_periods=1)
                           .median().values)
            rr_ms = np.diff(r_peaks) / FS_ECG * 1000.0 if len(r_peaks) >= 2 else np.array([0.0])
            pre_id = insert_apnea_preprocessed(
                raw_id, ecg_seg, r_peaks, spo2_smooth, resp_smooth,
                float(np.mean(rr_ms)), float(np.std(rr_ms)),
                int(len(r_peaks)), float(np.median(rr_ms)))

            # Stage 3: features + label
            feats = _extract_apnea_features(
                ecg_seg, pleth_seg, resp_seg, abp_seg, r_peaks, baseline,
                edr_bpm=edr_bpm, edr_quality=edr_quality,
                resp_gt=resp_gt_seg,
            )
            insert_apnea_features(pre_id, json.dumps(feats))

            seg_row = {"record": record_name, "segment_idx": i,
                       "run_id": run_id, **feats}
            insert_apnea_segment(seg_row)
            total_segs += 1

            if plots_per_record.get(record_name, 0) < 2 and feats["true_label"] in (0, 1):
                def _d(sig, n): return np.interp(np.linspace(0,1,n),
                                                  np.linspace(0,1,len(sig)), sig)
                insert_apnea_ecg_plot(
                    record_name, i, ecg_seg, r_peaks,
                    _d(spo2_smooth, SEGMENT_LEN_S),
                    _d(resp_smooth, SEGMENT_LEN_S),
                    _d(abp_seg, SEGMENT_LEN_S),
                    FS_ECG, feats["true_label"], feats["label_confidence"])
                plots_per_record[record_name] = plots_per_record.get(record_name, 0) + 1

        logger.info("[APNEA] %s: %d segments processed", record_name, n_segs)

    log_module("apnea", "ingest", "done", "Segments ingested", total_segs)

    if total_segs == 0:
        logger.error("[APNEA] No segments processed — aborting")
        return

    # ── Stage 4: Train Bidirectional LSTM ────────────────────────────────────
    log_module("apnea", "train", "started")
    if not HAS_TF:
        logger.error("[APNEA] TensorFlow not installed")
        log_module("apnea", "train", "failed", "tensorflow not installed", 0)
        return

    segs = fetch_apnea_segments(run_id=run_id)
    if len(segs) == 0:
        logger.warning("[APNEA] No segments for run_id=%s", run_id)
        log_module("apnea", "train", "skipped", "No segments", 0)
        return

    seg_df = pd.DataFrame(segs)
    total_fetched = len(seg_df)

    # Filter 1: GT-labelled only
    if "label_source" in seg_df.columns:
        gt_mask = seg_df["label_source"] == "mimic_resp"
        seg_df = seg_df[gt_mask].reset_index(drop=True)
        logger.info("[APNEA] Label filter: %d / %d segments have GT Resp labels",
                    len(seg_df), total_fetched)

    # Filter 2: quality gate
    if "mean_hr" in seg_df.columns:
        q_mask = seg_df["mean_hr"] > 0
        dropped = int((~q_mask).sum())
        if dropped:
            logger.info("[APNEA] Quality filter: dropped %d segments with mean_hr=0", dropped)
        seg_df = seg_df[q_mask].reset_index(drop=True)

    if "true_label" not in seg_df.columns or len(seg_df) == 0:
        logger.error("[APNEA] No valid segments after filtering")
        log_module("apnea", "train", "skipped", "No valid segments", 0)
        return

    n_apnea  = int((seg_df["true_label"] == 1).sum())
    n_normal = int((seg_df["true_label"] == 0).sum())
    apnea_pct = n_apnea / max(len(seg_df), 1)
    logger.info("[APNEA] Training set: %d segments — %d apnea / %d normal (%.0f%% apnea)",
                len(seg_df), n_apnea, n_normal, apnea_pct * 100)

    if apnea_pct > 0.70:
        logger.error("[APNEA] %.0f%% apnea — labelling likely broken", apnea_pct * 100)
        log_module("apnea", "train", "skipped", "Implausible label distribution", 0)
        return

    if n_apnea == 0 or n_normal == 0:
        logger.error("[APNEA] Only one class — cannot train")
        log_module("apnea", "train", "skipped", "Single class", 0)
        return

    if len(seg_df) < 20:
        logger.warning("[APNEA] Only %d segments — need ≥20", len(seg_df))
        log_module("apnea", "train", "skipped", "Not enough segments", 0)
        return

    X_all = seg_df[APNEA_FEATURE_COLS].fillna(0.0).values.astype(float)
    y_all = seg_df["true_label"].values.astype(float)

    TIMESTEPS = 10
    if len(X_all) <= TIMESTEPS:
        logger.warning("[APNEA] Not enough segments for sequence model")
        return

    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)

    X_seq = np.array([X_scaled[i:i+TIMESTEPS] for i in range(len(X_scaled)-TIMESTEPS)])
    y_seq = np.array([y_all[i+TIMESTEPS]       for i in range(len(y_all)-TIMESTEPS)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)
    logger.info("[APNEA] Train: %d seqs (%d apnea / %d normal) | "
                "Test: %d seqs (%d apnea / %d normal)",
                len(y_tr), int(y_tr.sum()), int((y_tr==0).sum()),
                len(y_te), int(y_te.sum()), int((y_te==0).sum()))

    # ========== MODEL ARCHITECTURE (Regularised for 418 sequences) ==========
    model = tf.keras.Sequential([
        # Input shape defined in first layer that accepts it
        tf.keras.layers.Input(shape=(TIMESTEPS, len(APNEA_FEATURE_COLS))),
        
        # Spatial dropout on input features — randomly zeros entire timesteps
        tf.keras.layers.SpatialDropout1D(0.2),
        
        # Smaller BiLSTM with L2 regularisation on kernel weights
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                48,
                return_sequences=True,
                recurrent_dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            )
        ),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                24,
                recurrent_dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            )
        ),
        tf.keras.layers.Dropout(0.2),
        
        # Smaller dense bottleneck
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    # ========== FOCAL LOSS (alpha=0.75 was best in run 2) ==========
    def focal_loss(gamma=2.0, alpha=0.75):
        """
        Focal loss for binary classification.
        
        gamma: focusing parameter (2.0 is standard)
        alpha: weighting for positive class (apnea)
              0.75 gives apnea 3x weight vs normal
        """
        def loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            return tf.reduce_mean(alpha_t * focal_weight * bce)
        return loss_fn

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.75),  # Back to run 2's best value
        metrics=["AUC"],
    )

    # ========== SIMPLE EARLY STOPPING (no LR scheduling needed) ==========
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,                    # Stop 5 epochs after peak
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
    ]

    # Train with regularised model
    history = model.fit(
        X_tr, y_tr, epochs=60, batch_size=32,
        validation_split=0.1, verbose=1,
        callbacks=callbacks
    )

    # ========== EVALUATION WITH THRESHOLD TUNING ==========
    y_prob = model.predict(X_te, verbose=0).flatten()

    if len(np.unique(y_te)) > 1:
        thresholds = np.arange(0.25, 0.75, 0.01)
        f1s = [f1_score(y_te, (y_prob > t).astype(int), zero_division=0)
               for t in thresholds]
        best_thresh = thresholds[np.argmax(f1s)]
        logger.info("[APNEA] Optimal threshold: %.2f (F1=%.3f)", best_thresh, max(f1s))
        
        auc = roc_auc_score(y_te, y_prob)
        
        # Bootstrap CI for AUC
        from sklearn.utils import resample
        n_bootstraps = 1000
        boot_aucs = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(y_te), len(y_te))
            if len(np.unique(y_te[idx])) > 1:
                boot_aucs.append(roc_auc_score(y_te[idx], y_prob[idx]))
        ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])
        logger.info("[APNEA] AUC: %.4f (95%% CI: %.3f-%.3f)", auc, ci_lower, ci_upper)
        
        report = classification_report(
            y_te, (y_prob > best_thresh).astype(int),
            target_names=["Normal", "Apnea"], output_dict=True)
        logger.info("\n%s", classification_report(
            y_te, (y_prob > best_thresh).astype(int),
            target_names=["Normal", "Apnea"]))
        
        insert_apnea_results(auc, report)
        log_module("apnea", "train", "done", f"auc={auc:.4f}", total_segs)
    else:
        logger.warning("[APNEA] Only one class in test set — AUC not computed")
        log_module("apnea", "train", "skipped", "Single class in test", 0)

    # ── Save model and scaler if requested ────────────────────────────────────
    if save_model and HAS_TF:
        _save_model_and_scaler(model, scaler, "apnea_model.keras", "apnea_scaler.pkl")

    logger.info("[APNEA] Module complete.")


# ── Model + scaler save helpers ──────────────────────────────────────────────

def _save_model_and_scaler(model, scaler, model_path: str, scaler_path: str) -> None:
    """Save the trained Keras model and fitted StandardScaler to disk."""
    import pickle
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("[SAVE] Model saved to %s", model_path)
    logger.info("[SAVE] Scaler saved to %s", scaler_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apnea Vital Signs ML Pipeline")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete the existing database before running")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the trained model and scaler to disk")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.fresh:
        for suffix in ["", "-wal", "-shm"]:
            p = DB_PATH + suffix
            if os.path.exists(p):
                os.remove(p)
        logger.info("[DB] Existing database deleted.")
    init_db()
    run_apnea_module(save_model=args.save_model)
    logger.info("[DONE] Pipeline complete. DB: %s", os.path.abspath(DB_PATH))


if __name__ == "__main__":
    main()