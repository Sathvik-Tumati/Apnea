from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import butter, coherence, filtfilt, find_peaks, welch
from scipy.signal import resample as scipy_resample
from scipy.signal.windows import tukey
try:
    import neurokit2 as nk
    HAS_NK = True
except Exception:
    HAS_NK = False
try:
    from pipeline.compute_edr_fixed import compute_edr_v3 as _compute_edr_v3
    HAS_EDR_V3 = True
except ImportError:
    HAS_EDR_V3 = False

from pipeline.modules.config import *
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _bandpass(
    signal: np.ndarray, fs: int,
    lo: float = 0.5, hi: float = 40.0, order: int = 3,
) -> np.ndarray:
    nyq = fs / 2.0
    actual_hi = min(hi, nyq - 0.1)
    actual_lo = min(lo, actual_hi - 0.1)
    b, a = butter(order, [actual_lo / nyq, actual_hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def _detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    # Guard against NaN-contaminated segments
    if not np.isfinite(ecg).all():
        ecg = np.where(np.isfinite(ecg), ecg, 0.0)
    if HAS_NK:
        try:
            _, info = nk.ecg_process(ecg, sampling_rate=fs)
            peaks = info["ECG_R_Peaks"]
            # Physiological sanity check: 30–200 bpm
            if len(peaks) >= 2:
                rr_ms = np.diff(peaks) / fs * 1000.0
                peaks = peaks[
                    np.concatenate([[True], (rr_ms >= 300) & (rr_ms <= 2000)])
                ]
            return peaks
        except Exception as exc:
            logger.warning("nk.ecg_process failed: %s — scipy fallback", exc)
    peaks, _ = find_peaks(ecg, distance=int(fs * 0.4), height=float(np.std(ecg)))
    return peaks


def _compute_edr(
    ecg: np.ndarray, r_peaks: np.ndarray,
    fs_ecg: int = FS_ECG, fs_resp: int = FS_RESP,
) -> np.ndarray:
    """Dual-engine QRS-area + QRS-PCA EDR fusion."""
    if len(r_peaks) < 8:
        return np.zeros(int(len(ecg) * fs_resp / fs_ecg))
    t_peaks  = r_peaks / fs_ecg
    t_uniform = np.arange(0, len(ecg) / fs_ecg, 1.0 / fs_resp)

    def _process_envelope(raw_v: np.ndarray, times: np.ndarray) -> np.ndarray:
        if len(raw_v) < 6:
            return np.zeros_like(t_uniform)
        detrended = raw_v - np.polyval(np.polyfit(times, raw_v, 1), times)
        s = np.interp(t_uniform, times, detrended)
        return (s - np.mean(s)) / (np.std(s) + 1e-9)

    qws = max(1, int(0.06 * fs_ecg))
    areas = [
        np.sum(np.abs(ecg[max(0, r - qws): min(len(ecg), r + qws)]))
        for r in r_peaks
    ]
    m3 = _process_envelope(np.array(areas, dtype=float), t_peaks)

    beats = [
        ecg[r - qws: r + qws]
        for r in r_peaks
        if r - qws >= 0 and r + qws <= len(ecg)
    ]
    m4 = m3
    if len(beats) >= 8:
        X = np.array(beats, dtype=float)
        X -= X.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            m4 = _process_envelope(U[:, 0] * S[0], t_peaks[: len(beats)])
        except np.linalg.LinAlgError:
            pass

    n = min(len(m3), len(m4))
    fused = np.median(np.vstack([m3[:n], m4[:n]]), axis=0)
    nyq = fs_resp / 2.0
    b, a = butter(3, [0.1 / nyq, 0.5 / nyq], btype="band")
    out = np.zeros_like(t_uniform)
    out[:n] = filtfilt(b, a, fused)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  LABELLING FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def _resp_flag_edr(resp: np.ndarray, fs: int) -> bool:
    """For EDR (amplitude-modulated ECG-derived respiration): detect flatline/cessation."""
    if len(resp) < fs * 10:
        return False
    threshold = np.mean(resp) - 1.5 * np.std(resp)
    suppressed = resp < threshold
    max_run = cur = 0
    for v in suppressed:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run >= (10 * fs)


def _resp_flag_gt(resp: np.ndarray, fs: int) -> bool:
    """
    Detect apnea in MIMIC GT Resp channel (chest impedance/bellows).
    Uses signal-adaptive prominence to handle cross-record amplitude variation.
    """
    if len(resp) < fs * 15:
        return False

    resp_std = np.std(resp)
    if resp_std < 0.05:
        # Completely flat signal — either sensor off or sustained apnea
        # Only call it apnea if signal isn't all-NaN-filled zeros
        return resp_std > 0.001

    # Adaptive prominence: 20% of signal std, minimum 0.05
    prominence = max(0.05, 0.20 * resp_std)

    # Cardiac contamination filter:
    # Real breathing is 0.1–0.6 Hz (6–36 breaths/min)
    # Minimum inter-breath distance = 1.67s at 36 bpm
    min_breath_distance = int(fs * 1.67)

    peaks, props = find_peaks(resp, distance=min_breath_distance, prominence=prominence)

    if len(peaks) < 3:
        # Fewer than 3 peaks in the window — likely apnea or sensor issue
        # Confirm by checking if signal has any meaningful oscillation at all
        if resp_std < 0.10:
            return False  # Sensor likely disconnected
        return True  # Real signal but no breathing detected

    # Calculate breath amplitudes from prominence values
    breath_amplitudes = props["prominences"]
    baseline_amplitude = np.percentile(breath_amplitudes, 75)

    if baseline_amplitude < 0.10 * resp_std:
        return False  # Peaks too small to be real breaths

    # Apnea = breath amplitude drops to <30% of baseline
    low_amp_threshold = 0.30 * baseline_amplitude
    low_amp_mask = breath_amplitudes < low_amp_threshold

    # Check for runs of low-amplitude breaths
    max_run = cur = 0
    for v in low_amp_mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    # Need at least 3 consecutive suppressed breaths (~10–15s depending on rate)
    if max_run >= 3:
        return True

    # Secondary check: look for 10-second windows with no peaks at all
    window_size = int(fs * 10)
    for i in range(0, len(resp) - window_size, window_size // 2):
        window = resp[i: i + window_size]
        w_peaks, _ = find_peaks(window, distance=min_breath_distance, prominence=prominence)
        if len(w_peaks) == 0 and np.std(window) > 0.10 * resp_std:
            return True

    return False


def _spo2_flag(pleth: np.ndarray, fs: int, baseline_spo2: float) -> bool:
    smooth = (
        pd.Series(pleth)
        .rolling(int(fs * 2), center=True, min_periods=1)
        .median()
        .values
    )
    return (baseline_spo2 - float(np.min(smooth)) >= 3.0) and (float(np.min(smooth)) < 94.0)


def _hrv_flag(
    r_peaks: np.ndarray, fs: int,
    baseline_rmssd: float, baseline_rr_ms: float,
) -> bool:
    if len(r_peaks) < 3:
        return False
    rr_ms    = np.diff(r_peaks) / fs * 1000.0
    rmssd_w  = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    mean_rr_w = float(np.mean(rr_ms))
    return (rmssd_w > 1.5 * baseline_rmssd) and (mean_rr_w > 1.2 * baseline_rr_ms)


def _abp_flag(
    abp: np.ndarray, fs: int,
    baseline_map_std: float, baseline_sbp: float,
) -> bool:
    map_sig = (
        pd.Series(abp)
        .rolling(int(fs * 2), center=True, min_periods=1)
        .mean()
        .values
    )
    return (
        float(np.std(map_sig)) > 1.5 * baseline_map_std
        or float(np.max(abp)) > baseline_sbp + 15.0
    )


def label_apnea_segment(resp: bool, spo2: bool, hrv: bool, has_gt_resp: bool = False) -> Tuple[int, str]:
    """
    Label apnea segment based on available signals.
    
    Args:
        resp: Respiratory suppression flag (from GT Resp or EDR)
        spo2: SpO2 desaturation flag
        hrv: HRV/autonomic activation flag
        has_gt_resp: Whether resp came from GT Resp channel (if True, resp alone is sufficient)
    """
    if not resp:
        return 0, "normal"
    
    # If we have GT Resp showing suppression, that's ground truth apnea
    if has_gt_resp:
        return 1, "gt_resp_apnea"
    
    # For EDR, require corroboration
    n = sum([resp, spo2, hrv])
    if n == 3: 
        return 1, "definite_apnea"
    if n == 2: 
        return 1, "probable_apnea"
    return 0, "possible_hypopnea"   # EDR alone = not enough


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-SIGNAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def _cross_signal_features(
    resp, pleth, r_peaks, abp,
    fs_resp, fs_pleth, fs_abp,
) -> Dict[str, float]:
    feats: Dict[str, float] = {
        "resp_spo2_lag_s": 0.0, "ptt_ms": 0.0, "ecg_resp_coherence": 0.0,
    }
    try:
        min_len_s = min(len(resp) / fs_resp, len(pleth) / fs_pleth)
        t_common  = np.arange(0, min_len_s, 1.0 / fs_resp)
        r_res = np.interp(t_common, np.arange(len(resp))   / fs_resp,  resp)
        p_res = np.interp(t_common, np.arange(len(pleth)) / fs_pleth, pleth)
        corr  = np.correlate(p_res - np.mean(p_res), r_res - np.mean(r_res), mode="full")
        lags  = np.arange(-(len(p_res) - 1), len(p_res))
        feats["resp_spo2_lag_s"] = float(lags[np.argmax(np.abs(corr))]) / fs_resp
    except Exception:
        pass
    try:
        ptt_vals: List[float] = []
        for rp in r_peaks:
            s = int((rp / FS_ECG) * fs_abp)
            e = min(int(s + 0.5 * fs_abp), len(abp) - 1)
            if e <= s:
                continue
            ptt_ms = int(np.argmin(abp[s:e])) / fs_abp * 1000.0
            if 50.0 < ptt_ms < 500.0:
                ptt_vals.append(ptt_ms)
        feats["ptt_ms"] = float(np.mean(ptt_vals)) if ptt_vals else 0.0
    except Exception:
        pass
    try:
        rr_series = np.diff(r_peaks) / float(FS_ECG)
        if len(rr_series) >= 8 and len(resp) >= 64:
            resp_res = np.interp(
                np.linspace(0, 1, len(rr_series)),
                np.linspace(0, 1, len(resp)), resp,
            )
            f, cxy = coherence(rr_series, resp_res, fs=1.0, nperseg=min(8, len(rr_series)))
            hf = (f >= 0.15) & (f <= 0.4)
            feats["ecg_resp_coherence"] = float(np.mean(cxy[hf]) if hf.any() else 0.0)
    except Exception:
        pass
    return feats


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SEGMENT FEATURE EXTRACTION  (MIMIC)
# ══════════════════════════════════════════════════════════════════════════════

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
    has_abp_signal: bool = True,
) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}

    resp_for_label = resp_gt if resp_gt is not None else resp
    resp_for_feats = resp_gt if resp_gt is not None else resp

    # ── HRV ──────────────────────────────────────────────────────────────────
    if len(r_peaks) >= 3:
        rr_ms    = np.diff(r_peaks) / FS_ECG * 1000.0
        rr_diffs = np.diff(rr_ms)
        feats["rr_mean"]     = float(np.mean(rr_ms))
        feats["rr_std"]      = float(np.std(rr_ms))
        feats["rmssd"]       = float(np.sqrt(np.mean(rr_diffs ** 2)))
        feats["pnn50"]       = float(np.sum(np.abs(rr_diffs) > 50.0) / max(len(rr_ms), 1))
        feats["mean_hr"]     = float(60000.0 / (np.mean(rr_ms) + 1e-6))
        feats["hr_range"]    = float(
            60000.0 / (np.min(rr_ms) + 1e-6) - 60000.0 / (np.max(rr_ms) + 1e-6)
        )
        if HAS_NK and len(r_peaks) >= 5:
            try:
                hf_df = nk.hrv_frequency(r_peaks, sampling_rate=FS_ECG, show=False)
                feats["lf_hf_ratio"] = float(hf_df["HRV_LFHF"].values[0])
            except Exception:
                feats["lf_hf_ratio"] = 0.0
        else:
            feats["lf_hf_ratio"] = 0.0
    else:
        for k in ("rr_mean", "rr_std", "rmssd", "pnn50",
                  "mean_hr", "hr_range", "lf_hf_ratio"):
            feats[k] = 0.0

    # ── SpO2 ──────────────────────────────────────────────────────────────────
    pleth_clean = pd.Series(pleth).ffill().bfill().values
    smooth = (
        pd.Series(pleth_clean)
        .rolling(int(FS_PPG * 2), center=True, min_periods=1)
        .median()
        .values
    )
    feats["spo2_mean"]          = float(np.mean(smooth))
    feats["spo2_min"]           = float(np.min(smooth))
    feats["spo2_delta_index"]   = float(np.max(smooth) - np.min(smooth))
    desat_thresh                = baseline.get("baseline_spo2", 97.0) - 3.0
    feats["odi"]                = float(np.sum(np.diff((smooth < desat_thresh).astype(int)) == 1))
    feats["t90"]                = float(np.mean(smooth < 90.0))
    # spo2_approx_entropy: Pincus ApEn — same computation as mongo_infer.py
    # (the previous formula used log-mean-abs-derivative which is always
    # negative, while production uses true ApEn which is always positive;
    # the sign mismatch caused the scaler to invert this feature's direction)
    def _apen(sig: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
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
    feats["spo2_approx_entropy"] = _apen(smooth)

    # ── Respiratory features ──────────────────────────────────────────────────
    feats["resp_amplitude_mean"] = float(np.mean(np.abs(resp_for_feats)))
    feats["resp_amplitude_std"]  = float(np.std(resp_for_feats))

    threshold  = np.mean(resp_for_feats) - 1.5 * np.std(resp_for_feats)
    suppressed = resp_for_feats < threshold
    max_run = cur = 0
    for v in suppressed:
        if v:
            cur += 1; max_run = max(max_run, cur)
        else:
            cur = 0
    feats["flatline_duration_s"] = float(max_run / FS_RESP)
    feats["label_source"]        = "mimic_resp" if resp_gt is not None else "edr"

    try:
        if resp_gt is not None:
            w_edge  = tukey(len(resp_for_feats), alpha=0.1)
            nperseg = min(len(resp_for_feats), max(8, int(FS_RESP * 60)))
            f, pxx  = welch(resp_for_feats * w_edge, fs=FS_RESP,
                            nperseg=nperseg, noverlap=nperseg // 2, nfft=2048)
            inn = (f >= 0.1) & (f <= 0.6)
            feats["resp_rate_bpm"] = float(f[inn][np.argmax(pxx[inn])] * 60.0) if np.any(inn) else 0.0
        elif edr_bpm is not None and (edr_quality or 0.0) >= 1.5:
            feats["resp_rate_bpm"] = float(edr_bpm)
        else:
            w_edge  = tukey(len(resp_for_feats), alpha=0.1)
            nperseg = min(len(resp_for_feats), max(8, int(FS_RESP * 60)))
            f, pxx  = welch(resp_for_feats * w_edge, fs=FS_RESP,
                            nperseg=nperseg, noverlap=nperseg // 2, nfft=2048)
            inn = (f >= 0.1) & (f <= 0.6)
            feats["resp_rate_bpm"] = float(f[inn][np.argmax(pxx[inn])] * 60.0) if np.any(inn) else 0.0

        feats["edr_quality_ok"] = int((edr_quality or 0.0) >= 1.5)
        feats["edr_snr"]        = float(edr_quality) if edr_quality is not None else 0.0

        resp_peaks, _ = find_peaks(resp_for_feats, distance=int(FS_RESP * 1.5))
        feats["resp_rate_variability"] = (
            float(np.std(np.diff(resp_peaks) / FS_RESP))
            if len(resp_peaks) >= 2 else 0.0
        )
    except Exception:
        feats["resp_rate_bpm"]         = 0.0
        feats["resp_rate_variability"] = 0.0
        feats["edr_quality_ok"]        = 0
        feats["edr_snr"]               = 0.0

    # ── ABP ───────────────────────────────────────────────────────────────────
    map_sig          = pd.Series(abp).rolling(int(FS_ECG * 2), center=True, min_periods=1).mean().values
    feats["map_mean"]       = float(np.mean(map_sig))
    feats["map_std"]        = float(np.std(map_sig))
    feats["map_variability"] = feats["map_std"]
    feats["sbp_max"]        = float(np.max(abp))
    feats["dbp_min"]        = float(np.min(abp))
    feats["pulse_pressure"] = feats["sbp_max"] - feats["dbp_min"]

    # ── Cross-signal ──────────────────────────────────────────────────────────
    cross = _cross_signal_features(
        resp_for_feats, pleth_clean, r_peaks, abp, FS_RESP, FS_PPG, FS_ECG
    )
    feats.update(cross)

    # ── Modality flags ────────────────────────────────────────────────────────
    feats["has_spo2"]    = 1
    feats["has_abp"]     = int(has_abp_signal)
    feats["has_resp_gt"] = int(resp_gt is not None)

    # ── Labels ────────────────────────────────────────────────────────────────
    # Choose the right resp_flag function based on signal source
    if resp_gt is not None:
        rf = _resp_flag_gt(resp_for_label, FS_RESP)
    else:
        rf = _resp_flag_edr(resp_for_label, FS_RESP)
    
    sf = _spo2_flag(pleth_clean, FS_PPG, baseline.get("baseline_spo2", 97.0))
    hf = _hrv_flag(r_peaks, FS_ECG,
                   baseline.get("baseline_rmssd", 35.0),
                   baseline.get("baseline_rr_ms", 833.0))
    af = _abp_flag(abp, FS_ECG,
                   baseline.get("baseline_map_std", 5.0),
                   baseline.get("baseline_sbp", 120.0))

    has_gt_resp = resp_gt is not None
    label, conf = label_apnea_segment(rf, sf, hf, has_gt_resp)
    feats["resp_flag"]         = int(rf)
    feats["spo2_flag"]         = int(sf)
    feats["hrv_flag"]          = int(hf)
    feats["abp_flag"]          = int(af)
    feats["signals_positive"]  = sum([rf, sf, hf])
    feats["true_label"]        = label
    feats["label_confidence"]  = conf
    feats["data_source"]       = "mimic"

    return feats


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SEGMENT FEATURE EXTRACTION  (SLPDB — ECG ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_features_slpdb(ecg_seg: np.ndarray) -> Dict[str, float]:
    """
    Extract APNEA_FEATURE_COLS from a 30-second 125 Hz ECG segment.
    SpO2 / ABP columns are zeroed; modality flags mark them as absent.
    """
    feats: Dict[str, float] = {}
    r_peaks = _detect_r_peaks(ecg_seg, FS_ECG)

    # HRV
    if len(r_peaks) >= 3:
        rr_ms    = np.diff(r_peaks) / FS_ECG * 1000.0
        rr_diffs = np.diff(rr_ms)
        feats["rr_mean"]     = float(np.mean(rr_ms))
        feats["rr_std"]      = float(np.std(rr_ms))
        feats["rmssd"]       = float(np.sqrt(np.mean(rr_diffs ** 2)))
        feats["pnn50"]       = float(np.sum(np.abs(rr_diffs) > 50) / max(len(rr_ms), 1))
        feats["mean_hr"]     = float(60000.0 / (np.mean(rr_ms) + 1e-6))
        feats["hr_range"]    = float(
            60000.0 / (np.min(rr_ms) + 1e-6) - 60000.0 / (np.max(rr_ms) + 1e-6)
        )
        feats["lf_hf_ratio"] = 0.0
    else:
        for k in ("rr_mean", "rr_std", "rmssd", "pnn50",
                  "mean_hr", "hr_range", "lf_hf_ratio"):
            feats[k] = 0.0

    # EDR-derived respiratory features
    resp = _compute_edr(ecg_seg, r_peaks, FS_ECG, FS_RESP)
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
        w_edge  = tukey(len(resp), alpha=0.1)
        nperseg = min(len(resp), max(8, int(FS_RESP * 60)))
        f, pxx  = welch(resp * w_edge, fs=FS_RESP,
                        nperseg=nperseg, noverlap=nperseg // 2, nfft=2048)
        inn = (f >= 0.1) & (f <= 0.6)
        feats["resp_rate_bpm"] = float(f[inn][np.argmax(pxx[inn])] * 60.0) if np.any(inn) else 0.0
        rp2, _  = find_peaks(resp, distance=int(FS_RESP * 1.5))
        feats["resp_rate_variability"] = (
            float(np.std(np.diff(rp2) / FS_RESP)) if len(rp2) >= 2 else 0.0
        )
    except Exception:
        feats["resp_rate_bpm"]         = 0.0
        feats["resp_rate_variability"] = 0.0

    # SpO2 / ABP / cross — absent; use 0 so scaler maps to a distinct value
    feats.update({k: 0.0 for k in SPO2_FEATURE_COLS + ABP_FEATURE_COLS + CROSS_FEATURE_COLS})

    # Modality flags — ECG only
    feats["has_spo2"]    = 0
    feats["has_abp"]     = 0
    feats["has_resp_gt"] = 0

    feats["label_source"] = "slpdb_annotation"
    feats["data_source"]  = "slpdb"

    return feats


