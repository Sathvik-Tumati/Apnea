"""
pipeline/pipeline.py
====================
Three-module ML pipeline for clinical vital-sign prediction.

Modules
-------
arrhythmia  Beat-level ECG classification (N / VEB / SVEB / F / Q)
            Data: MIT-BIH / INCART / SCD-Holter CSVs
            Model: RandomForestClassifier

apnea       30-second segment apnea detection
            Data: MIMIC-IV Waveform DB streamed via wfdb
            Label: AASM multi-signal composite (2-of-4 signals)
            Model: Bidirectional LSTM

sepsis      ICU sepsis early warning
            Data: sepsis_icu_synthetic.csv
            Model: GradientBoostingClassifier

Usage
-----
python pipeline/pipeline.py                  # run all three
python pipeline/pipeline.py --module arrhythmia
python pipeline/pipeline.py --module apnea
python pipeline/pipeline.py --module sepsis
python pipeline/pipeline.py --fresh          # delete DB before run
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, filtfilt, find_peaks

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CLI.db.database import (
    DB_PATH,
    _j,
    init_db,
    insert_apnea_ecg_plot,
    insert_apnea_features,
    insert_apnea_preprocessed,
    insert_apnea_raw,
    insert_apnea_results,
    insert_apnea_segment,
    insert_arr_ecg_plot,
    insert_arr_features,
    insert_arr_predictions,
    insert_arr_preprocessed,
    insert_arr_raw,
    insert_arr_results,
    insert_sep_features,
    insert_sep_predictions,
    insert_sep_preprocessed,
    insert_sep_raw,
    insert_sep_results,
    insert_sep_vitals_plot,
    fetch_arr_features,
    fetch_sep_features,
    fetch_apnea_segments,
    log_module,
)

# ── optional heavy deps ───────────────────────────────────────────────────────
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

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR: str = os.environ.get("DATA_DIR", "/Users/sathvik/Desktop/Internship/archive2")
MIMIC_URL: str = "https://physionet.org/files/mimic4wdb/0.1.0/"
FS_MIMIC: int = 320
SEGMENT_LEN_S: int = 30
BEATS_PER_TYPE: int = 3000
N_MIMIC_RECORDS: int = 10

ECG_FILES: List[Tuple[str, str]] = [
    ("MIT-BIH_Arrhythmia_Database.csv",
     "MIT-BIH Arrhythmia Database.csv"),
    ("MIT-BIH_Supraventricular_Arrhythmia_Database.csv",
     "MIT-BIH Supraventricular Arrhythmia Database.csv"),
    ("INCART_2-lead_Arrhythmia_Database.csv",
     "INCART 2-lead Arrhythmia Database.csv"),
    ("Sudden_Cardiac_Death_Holter_Database.csv",
     "Sudden Cardiac Death Holter Database.csv"),
]

BEAT_COLS: List[str] = [
    "0_pre-RR", "0_post-RR", "0_pPeak", "0_tPeak", "0_rPeak",
    "0_sPeak", "0_qPeak", "0_qrs_interval", "0_pq_interval",
    "0_qt_interval", "0_st_interval",
    "0_qrs_morph0", "0_qrs_morph1", "0_qrs_morph2",
    "0_qrs_morph3", "0_qrs_morph4",
    "1_pre-RR", "1_post-RR", "1_pPeak", "1_tPeak", "1_rPeak",
    "1_sPeak", "1_qPeak", "1_qrs_interval", "1_pq_interval",
    "1_qt_interval", "1_st_interval",
    "1_qrs_morph0", "1_qrs_morph1", "1_qrs_morph2",
    "1_qrs_morph3", "1_qrs_morph4",
]

SEPSIS_COLS: List[str] = [
    "hr_mean", "hr_max", "hr_min", "hr_std",
    "sbp_mean", "sbp_min", "sbp_std",
    "dbp_mean", "dbp_min", "map_mean",
    "temp_celsius_mean", "temp_celsius_max", "temp_celsius_std",
    "spo2_mean", "spo2_min", "spo2_std",
    "respiratory_rate_mean", "respiratory_rate_max", "respiratory_rate_std",
    "wbc", "lactate_mmol", "creatinine", "platelet_count",
    "bilirubin_total", "glucose", "ph_arterial", "pao2_fio2_ratio",
    "sofa_score", "apache_iv", "qsofa", "sirs_criteria",
    "vasopressors_flag", "mechanical_ventilation",
]

APNEA_FEATURE_COLS: List[str] = [
    "rr_mean", "rr_std", "rmssd", "pnn50", "lf_hf_ratio",
    "mean_hr", "hr_range",
    "spo2_mean", "spo2_min", "spo2_delta_index", "odi", "t90",
    "spo2_approx_entropy",
    "resp_amplitude_mean", "resp_amplitude_std",
    "flatline_duration_s", "resp_rate_bpm", "resp_rate_variability",
    "map_mean", "map_std", "sbp_max", "dbp_min",
    "pulse_pressure", "map_variability",
    "resp_spo2_lag_s", "ptt_ms", "ecg_resp_coherence",
    "resp_flag", "spo2_flag", "hrv_flag", "abp_flag", "signals_positive",
]


# ── shared signal utilities ───────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, fs: int,
              lo: float = 0.5, hi: float = 40.0,
              order: int = 3) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    # Ensure hi is strictly less than nyquist frequency to prevent ValueError
    actual_hi = min(hi, nyq - 0.1)
    # Ensure lo is valid relative to actual_hi
    actual_lo = min(lo, actual_hi - 0.1)
    
    b, a = butter(order, [actual_lo / nyq, actual_hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def _detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peaks using neurokit2 if available, else scipy."""
    if HAS_NK:
        try:
            _, info = nk.ecg_process(ecg, sampling_rate=fs)
            return info["ECG_R_Peaks"]
        except Exception as exc:
            logger.warning("nk.ecg_process failed: %s — using scipy fallback", exc)
    peaks, _ = find_peaks(ecg, distance=int(fs * 0.4),
                          height=float(np.std(ecg)))
    return peaks


def _resolve(underscore: str, spaced: str) -> Optional[str]:
    """Return the first existing path from two name variants."""
    for name in [underscore, spaced]:
        p = os.path.join(DATA_DIR, name)
        if os.path.isfile(p):
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — ARRHYTHMIA
# ═══════════════════════════════════════════════════════════════════════════════

def _arr_preprocess_row(raw: Dict) -> Dict[str, float]:
    """Derive cleaned features from one raw ECG beat dict."""
    def g(k: str) -> float:
        try:
            return float(raw.get(k) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    rr_pre = g("0_pre-RR")
    rr_post = g("0_post-RR")
    rr_ratio = rr_pre / (rr_post + 1e-6)
    return {
        "rr_ratio": rr_ratio,
        "rr_diff": rr_post - rr_pre,
        "rr_symmetry": abs(rr_ratio - 1.0),
        "qrs_amplitude": g("0_rPeak") - g("0_sPeak"),
        "qrs_diff_leads": g("0_rPeak") - g("1_rPeak"),
        "st_diff_leads": g("0_st_interval") - g("1_st_interval"),
        "p_absent": int(abs(g("0_pPeak")) < 0.05),
        "qtc_approx": g("0_qt_interval") / (rr_pre ** 0.5 + 1e-6),
    }


def _arr_save_ecg_plot(df: pd.DataFrame, le: LabelEncoder) -> None:
    """Save one annotated beat per class into arrhythmia_ecg_plot."""
    for beat_type in le.classes_:
        sub = df[df["type"] == beat_type]
        if sub.empty:
            continue
        row = sub.iloc[0]
        record = str(row.get("record", "unknown"))
        # Reconstruct a synthetic beat from morph columns
        morph_cols = [c for c in df.columns if "morph" in c and c.startswith("0_")]
        ecg_segment = np.array(
            [float(row.get(c) or 0.0) for c in morph_cols], dtype=float
        )
        r_idx = np.array([len(ecg_segment) // 2])
        insert_arr_ecg_plot(
            record=record,
            beat_type=beat_type,
            ecg=ecg_segment,
            r_peaks=r_idx,
            p_peaks=np.array([max(0, r_idx[0] - 4)]),
            q_peaks=np.array([max(0, r_idx[0] - 2)]),
            s_peaks=np.array([min(len(ecg_segment) - 1, r_idx[0] + 2)]),
            t_peaks=np.array([min(len(ecg_segment) - 1, r_idx[0] + 6)]),
            fs=360,
        )
    logger.info("[ARR] ECG plot segments saved for all beat types")


def run_arrhythmia_module(beats_per_type: int = BEATS_PER_TYPE) -> None:
    """Run the full arrhythmia pipeline: ingest → preprocess → features → train."""
    logger.info("=" * 60)
    logger.info(" ARRHYTHMIA MODULE")
    logger.info("=" * 60)

    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    log_module("arrhythmia", "ingest", "started")
    all_dfs: List[pd.DataFrame] = []

    for u_name, s_name in ECG_FILES:
        path = _resolve(u_name, s_name)
        if not path:
            logger.warning("[ARR] %s not found — skipping", u_name)
            continue
        df = pd.read_csv(path, low_memory=False)
        if "type" not in df.columns:
            logger.warning("[ARR] %s missing 'type' column — skipping", path)
            continue
        df = df[df["type"].isin(["N", "SVEB", "VEB", "F", "Q"])]
        df = df.groupby("type", group_keys=False).apply(
            lambda x: x.sample(min(len(x), beats_per_type), random_state=42)
        )
        source = Path(path).stem
        raw_rows = [
            (source, str(row.get("record", "")), row["type"],
             json.dumps({k: (None if pd.isna(v) else v)
                         for k, v in row.items()}))
            for _, row in df.iterrows()
        ]
        insert_arr_raw(raw_rows)
        all_dfs.append(df)
        logger.info("[ARR] %s: %d beats ingested", source, len(df))

    if not all_dfs:
        logger.error("[ARR] No ECG files loaded — aborting arrhythmia module")
        log_module("arrhythmia", "ingest", "failed", "No ECG files found", 0)
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    log_module("arrhythmia", "ingest", "done", "Raw beats stored",
               len(combined_df))

    # ── Stage 2: Preprocess ───────────────────────────────────────────────────
    log_module("arrhythmia", "preprocess", "started")
    import sqlite3 as _sql
    con = _sql.connect(DB_PATH)
    con.row_factory = _sql.Row
    raw_rows_db = con.execute(
        "SELECT id, beat_type, raw_json FROM arrhythmia_raw"
        " WHERE id NOT IN (SELECT raw_id FROM arrhythmia_preprocessed)"
    ).fetchall()
    con.close()

    pre_rows: List[tuple] = []
    for r in raw_rows_db:
        raw = json.loads(r["raw_json"])
        p = _arr_preprocess_row(raw)
        pre_rows.append((
            r["id"], r["beat_type"],
            p["rr_ratio"], p["rr_diff"], p["rr_symmetry"],
            p["qrs_amplitude"], p["qrs_diff_leads"],
            p["st_diff_leads"], p["p_absent"], p["qtc_approx"],
        ))

    insert_arr_preprocessed(pre_rows)
    logger.info("[ARR] %d preprocessed rows stored", len(pre_rows))
    log_module("arrhythmia", "preprocess", "done", "", len(pre_rows))

    # ── Stage 3: Feature extraction ───────────────────────────────────────────
    log_module("arrhythmia", "features", "started")
    import sqlite3 as _sql
    con = _sql.connect(DB_PATH)
    con.row_factory = _sql.Row
    pre_df = pd.read_sql("""
        SELECT p.id, p.beat_type, p.rr_ratio, p.rr_diff, p.rr_symmetry,
               p.qrs_amplitude, p.qrs_diff_leads, p.st_diff_leads,
               p.p_absent, p.qtc_approx, r.raw_json
        FROM arrhythmia_preprocessed p
        JOIN arrhythmia_raw r ON r.id = p.raw_id
        WHERE p.id NOT IN (SELECT preprocessed_id FROM arrhythmia_features)
    """, con)
    con.close()

    raw_feat_df = (
        pre_df["raw_json"]
        .apply(lambda x: {c: float(json.loads(x).get(c) or 0.0)
                          for c in BEAT_COLS})
        .apply(pd.Series)
    )
    feat_df = pd.concat(
        [pre_df.drop(columns=["raw_json"]), raw_feat_df], axis=1
    ).fillna(0.0)

    feat_cols = [c for c in feat_df.columns if c not in ("id", "beat_type")]
    feat_rows: List[tuple] = []
    for i, (_, row) in enumerate(feat_df.iterrows()):
        feat_rows.append((
            int(row["id"]),
            row["beat_type"],
            feat_df[feat_cols].iloc[i].to_json(),
        ))

    insert_arr_features(feat_rows)
    logger.info("[ARR] %d feature rows stored", len(feat_rows))
    log_module("arrhythmia", "features", "done", "", len(feat_rows))

    # ── Stage 4: Train & predict ──────────────────────────────────────────────
    log_module("arrhythmia", "train", "started")
    feat_load = fetch_arr_features()
    X = feat_load["feature_json"].apply(json.loads).apply(pd.Series).fillna(0.0)
    le = LabelEncoder()
    y = le.fit_transform(feat_load["beat_type"])

    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, feat_load["id"],
        test_size=0.2, stratify=y, random_state=42,
    )
    scaler = StandardScaler()
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    model.fit(scaler.fit_transform(X_tr), y_tr)
    y_pred = model.predict(scaler.transform(X_te))
    y_prob = model.predict_proba(scaler.transform(X_te))
    report = classification_report(y_te, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)
    accuracy = report["accuracy"]
    logger.info("[ARR] Accuracy: %.4f", accuracy)
    logger.info("\n%s", classification_report(y_te, y_pred,
                                              target_names=le.classes_))

    insert_arr_results(accuracy, report)
    pred_rows = [
        (str(int(fid)), le.classes_[yt], le.classes_[yp], float(ypa.max()))
        for fid, yt, yp, ypa in zip(id_te, y_te, y_pred, y_prob)
    ]
    insert_arr_predictions(pred_rows)

    # Save ECG plot segments
    _arr_save_ecg_plot(combined_df, le)

    log_module("arrhythmia", "train", "done",
               f"acc={accuracy:.4f}", len(pred_rows))
    logger.info("[ARR] Module complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — APNEA  (MIMIC-IV Waveform, AASM multi-signal labelling)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Signal flag functions ─────────────────────────────────────────────────────

def _resp_flag(resp: np.ndarray, fs: int) -> bool:
    """True if Resp shows ≥10s of sustained amplitude suppression."""
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


def _spo2_flag(pleth: np.ndarray, fs: int,
               baseline_spo2: float) -> bool:
    """True if Pleth-derived SpO2 drops ≥3% from baseline AND min < 94%."""
    smoothed = (
        pd.Series(pleth)
        .rolling(int(fs * 2), center=True, min_periods=1)
        .median()
        .values
    )
    spo2_min = float(np.min(smoothed))
    delta = baseline_spo2 - spo2_min
    return (delta >= 3.0) and (spo2_min < 94.0)


def _hrv_flag(r_peaks: np.ndarray, fs: int,
              baseline_rmssd: float,
              baseline_rr_ms: float) -> bool:
    """True if HRV shows autonomic signature of apnea."""
    if len(r_peaks) < 3:
        return False
    rr_ms = np.diff(r_peaks) / fs * 1000.0
    rmssd_w = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    mean_rr_w = float(np.mean(rr_ms))
    hrv_surge = rmssd_w > 1.5 * baseline_rmssd
    bradycardia = mean_rr_w > 1.2 * baseline_rr_ms
    return hrv_surge or bradycardia


def _abp_flag(abp: np.ndarray, fs: int,
              baseline_map_std: float,
              baseline_sbp: float) -> bool:
    """True if ABP shows haemodynamic apnea signature."""
    map_sig = (
        pd.Series(abp)
        .rolling(int(fs * 2), center=True, min_periods=1)
        .mean()
        .values
    )
    map_std_w = float(np.std(map_sig))
    peak_sbp_w = float(np.max(abp))
    pressure_var = map_std_w > 1.5 * baseline_map_std
    bp_surge = peak_sbp_w > baseline_sbp + 15.0
    return pressure_var or bp_surge


def label_apnea_segment(
    resp: bool, spo2: bool, hrv: bool, abp: bool
) -> Tuple[int, str]:
    """AASM-aligned composite apnea labelling (2-of-4 threshold)."""
    n = sum([resp, spo2, hrv, abp])
    if n >= 3:
        return 1, "definite_apnea"
    if n == 2:
        return 1, "probable_apnea"
    if n == 1:
        return 0, "possible_hypopnea"
    return 0, "normal"


# ── Cross-signal features ─────────────────────────────────────────────────────

def _cross_signal_features(
    resp: np.ndarray,
    pleth: np.ndarray,
    r_peaks: np.ndarray,
    abp: np.ndarray,
    fs: int,
) -> Dict[str, float]:
    """Compute resp_spo2_lag, PTT, and ECG-Resp coherence."""
    feats: Dict[str, float] = {
        "resp_spo2_lag_s": 0.0,
        "ptt_ms": 0.0,
        "ecg_resp_coherence": 0.0,
    }

    # resp_spo2_lag
    try:
        min_len = min(len(resp), len(pleth))
        r_norm = resp[:min_len] - np.mean(resp[:min_len])
        p_norm = pleth[:min_len] - np.mean(pleth[:min_len])
        corr = np.correlate(p_norm, r_norm, mode="full")
        lags = np.arange(-(min_len - 1), min_len)
        feats["resp_spo2_lag_s"] = float(lags[np.argmax(np.abs(corr))]) / fs
    except Exception:
        pass

    # PTT
    try:
        ptt_vals: List[float] = []
        for rp in r_peaks:
            s_start = int(rp)
            s_end = min(int(rp + 0.5 * fs), len(abp) - 1)
            if s_end <= s_start:
                continue
            foot = int(np.argmin(abp[s_start:s_end]))
            ptt_ms = foot / fs * 1000.0
            if 50.0 < ptt_ms < 500.0:
                ptt_vals.append(ptt_ms)
        feats["ptt_ms"] = float(np.mean(ptt_vals)) if ptt_vals else 0.0
    except Exception:
        pass

    # RSA coherence
    try:
        rr_series = np.diff(r_peaks) / float(fs)
        if len(rr_series) >= 8 and len(resp) >= 64:
            resp_res = np.interp(
                np.linspace(0, 1, len(rr_series)),
                np.linspace(0, 1, len(resp)),
                resp,
            )
            f, cxy = coherence(rr_series, resp_res,
                               fs=1.0, nperseg=min(8, len(rr_series)))
            hf = (f >= 0.15) & (f <= 0.4)
            feats["ecg_resp_coherence"] = float(
                np.mean(cxy[hf]) if hf.any() else 0.0
            )
    except Exception:
        pass

    return feats


# ── Per-segment full feature extraction ──────────────────────────────────────

def _extract_apnea_features(
    ecg: np.ndarray,
    pleth: np.ndarray,
    resp: np.ndarray,
    abp: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    baseline: Dict[str, float],
) -> Dict[str, Any]:
    """Extract all 33 features + all 4 signal flags for one 30s segment."""
    feats: Dict[str, Any] = {}

    # HRV
    if len(r_peaks) >= 3:
        rr_ms = np.diff(r_peaks) / fs * 1000.0
        rr_diffs = np.diff(rr_ms)
        feats["rr_mean"] = float(np.mean(rr_ms))
        feats["rr_std"] = float(np.std(rr_ms))
        feats["rmssd"] = float(np.sqrt(np.mean(rr_diffs ** 2)))
        feats["pnn50"] = float(
            np.sum(np.abs(rr_diffs) > 50.0) / max(len(rr_ms), 1)
        )
        feats["mean_hr"] = float(60000.0 / (np.mean(rr_ms) + 1e-6))
        feats["hr_range"] = float(
            60000.0 / (np.min(rr_ms) + 1e-6) - 60000.0 / (np.max(rr_ms) + 1e-6)
        )
        if HAS_NK and len(r_peaks) >= 5:
            try:
                hf = nk.hrv_frequency(r_peaks, sampling_rate=fs, show=False)
                feats["lf_hf_ratio"] = float(hf["HRV_LFHF"].values[0])
            except Exception:
                feats["lf_hf_ratio"] = 0.0
        else:
            feats["lf_hf_ratio"] = 0.0
    else:
        for k in ("rr_mean", "rr_std", "rmssd", "pnn50",
                  "mean_hr", "hr_range", "lf_hf_ratio"):
            feats[k] = 0.0

    # SpO2 / Pleth
    smooth = (
        pd.Series(pleth)
        .rolling(int(fs * 2), center=True, min_periods=1)
        .median()
        .values
    )
    feats["spo2_mean"] = float(np.mean(smooth))
    feats["spo2_min"] = float(np.min(smooth))
    feats["spo2_delta_index"] = float(np.max(smooth) - np.min(smooth))
    desat_thresh = baseline.get("baseline_spo2", 97.0) - 3.0
    feats["odi"] = float(np.sum(np.diff((smooth < desat_thresh).astype(int)) == 1))
    feats["t90"] = float(np.mean(smooth < 90.0))
    try:
        m = np.mean(smooth)
        s = np.std(smooth)
        phi = (smooth - m) / (s + 1e-9)
        feats["spo2_approx_entropy"] = float(
            -np.mean(np.log(np.abs(np.diff(phi)) + 1e-9))
        )
    except Exception:
        feats["spo2_approx_entropy"] = 0.0

    # Resp
    feats["resp_amplitude_mean"] = float(np.mean(np.abs(resp)))
    feats["resp_amplitude_std"] = float(np.std(resp))
    threshold = np.mean(resp) - 1.5 * np.std(resp)
    suppressed = resp < threshold
    max_run = current_run = 0
    for val in suppressed:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    feats["flatline_duration_s"] = float(max_run / fs)
    try:
        resp_peaks, _ = find_peaks(resp, distance=int(fs * 1.5))
        feats["resp_rate_bpm"] = float(
            len(resp_peaks) / (len(resp) / fs) * 60.0
        )
        if len(resp_peaks) >= 2:
            rri = np.diff(resp_peaks) / fs
            feats["resp_rate_variability"] = float(np.std(rri))
        else:
            feats["resp_rate_variability"] = 0.0
    except Exception:
        feats["resp_rate_bpm"] = 0.0
        feats["resp_rate_variability"] = 0.0

    # ABP
    map_sig = (
        pd.Series(abp)
        .rolling(int(fs * 2), center=True, min_periods=1)
        .mean()
        .values
    )
    feats["map_mean"] = float(np.mean(map_sig))
    feats["map_std"] = float(np.std(map_sig))
    feats["map_variability"] = feats["map_std"]
    feats["sbp_max"] = float(np.max(abp))
    feats["dbp_min"] = float(np.min(abp))
    feats["pulse_pressure"] = feats["sbp_max"] - feats["dbp_min"]

    # Cross-signal
    cross = _cross_signal_features(resp, pleth, r_peaks, abp, fs)
    feats.update(cross)

    # Signal flags
    rf = _resp_flag(resp, fs)
    sf = _spo2_flag(pleth, fs, baseline.get("baseline_spo2", 97.0))
    hf = _hrv_flag(r_peaks, fs,
                   baseline.get("baseline_rmssd", 35.0),
                   baseline.get("baseline_rr_ms", 833.0))
    af = _abp_flag(abp, fs,
                   baseline.get("baseline_map_std", 5.0),
                   baseline.get("baseline_sbp", 120.0))
    label, conf = label_apnea_segment(rf, sf, hf, af)

    feats["resp_flag"] = int(rf)
    feats["spo2_flag"] = int(sf)
    feats["hrv_flag"] = int(hf)
    feats["abp_flag"] = int(af)
    feats["signals_positive"] = sum([rf, sf, hf, af])
    feats["true_label"] = label
    feats["label_confidence"] = conf

    return feats


def _load_mimic_records(n: int = N_MIMIC_RECORDS) -> List[str]:
    """Stream the RECORDS list from MIMIC-IV and return the first n paths."""
    if not HAS_WFDB:
        logger.error("[APNEA] wfdb not installed — cannot load MIMIC-IV")
        return []
    try:
        import urllib.request
        url = MIMIC_URL + "RECORDS"
        with urllib.request.urlopen(url, timeout=30) as resp:
            lines = resp.read().decode().splitlines()
        
        # lines look like: 'waves/p100/p10014354/'
        # we need to fetch the RECORDS file inside each to get the actual segments
        valid_paths = []
        for ln in lines:
            if not ln.strip(): continue
            dir_path = ln.strip()
            inner_url = MIMIC_URL + dir_path + "RECORDS"
            try:
                with urllib.request.urlopen(inner_url, timeout=10) as inner_resp:
                    inner_lines = inner_resp.read().decode().splitlines()
                    for iln in inner_lines:
                        if iln.strip():
                            valid_paths.append(dir_path + iln.strip())
            except Exception:
                pass
            if len(valid_paths) >= n:
                break
        return valid_paths[:n]
    except Exception as exc:
        logger.error("[APNEA] Failed to fetch RECORDS: %s", exc)
        return []


def run_apnea_module(n_records: int = N_MIMIC_RECORDS) -> None:
    """Run the full apnea pipeline on MIMIC-IV waveform records."""
    logger.info("=" * 60)
    logger.info(" APNEA MODULE")
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

    SIGNAL_NAMES = {"II": 0, "Pleth": 1, "Resp": 2, "ABP": 3}
    total_segs = 0
    plot_saved = 0
    samples_per_seg = FS_MIMIC * SEGMENT_LEN_S

    for rec_path in record_paths:
        record_name = rec_path.split("/")[-1]
        # Example rec_path: waves/p100/p10039708/83411188/83411188
        # pn_dir should be: mimic4wdb/0.1.0/waves/p100/p10039708/83411188
        pn_dir = "mimic4wdb/0.1.0/" + "/".join(rec_path.split("/")[:-1])
        
        try:
            # We want max 10 segments per patient. 
            # 10 segments * 30 seconds * 320 Hz = 96,000 samples.
            # Passing sampto ensures wfdb STOPS downloading after 96,000 samples.
            rec = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=96000)
        except Exception as exc:
            logger.warning("[APNEA] Could not load %s: %s", record_name, exc)
            continue

        sig_map = {name: idx for idx, name in enumerate(rec.sig_name)}
        missing = [s for s in SIGNAL_NAMES if s not in sig_map]
        if missing:
            logger.warning("[APNEA] %s missing signals %s — skipping",
                           record_name, missing)
            continue

        signals = rec.p_signal
        fs = rec.fs
        ecg_full = signals[:, sig_map["II"]]
        pleth_full = signals[:, sig_map["Pleth"]]
        resp_full = signals[:, sig_map["Resp"]]
        abp_full = signals[:, sig_map["ABP"]]

        # Fill NaNs
        for arr in (ecg_full, pleth_full, resp_full, abp_full):
            nan_mask = np.isnan(arr)
            arr[nan_mask] = np.nanmean(arr) if not np.all(nan_mask) else 0.0

        n_segs = len(ecg_full) // samples_per_seg
        if n_segs == 0:
            continue
        
        # Limit to 10 segments (5 minutes) per record
        n_segs = min(n_segs, 10)

        # Compute subject baseline from first 5 "quiet" windows
        baseline_windows: List[Dict] = []
        for i in range(min(n_segs, 20)):
            s = i * samples_per_seg
            e = s + samples_per_seg
            ecg_seg = _bandpass(ecg_full[s:e], fs)
            rp = _detect_r_peaks(ecg_seg, fs)
            rr_ms = np.diff(rp) / fs * 1000.0 if len(rp) >= 3 else np.array([833.0])
            rmssd_w = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) >= 2 else 35.0
            pleth_seg = pleth_full[s:e]
            abp_seg = abp_full[s:e]
            map_sig = pd.Series(abp_seg).rolling(int(fs * 2), min_periods=1).mean().values
            individually_clean = (
                not _resp_flag(resp_full[s:e], fs)
                and float(np.min(pleth_seg)) > 90.0
            )
            baseline_windows.append({
                "spo2_mean": float(np.mean(pleth_seg)),
                "rmssd": rmssd_w,
                "rr_mean": float(np.mean(rr_ms)),
                "map_std": float(np.std(map_sig)),
                "sbp_max": float(np.max(abp_seg)),
                "individually_clean": individually_clean,
            })

        clean_windows = [w for w in baseline_windows if w["individually_clean"]][:5]
        if clean_windows:
            baseline = {
                "baseline_spo2": float(np.mean([w["spo2_mean"] for w in clean_windows])),
                "baseline_rmssd": float(np.mean([w["rmssd"] for w in clean_windows])),
                "baseline_rr_ms": float(np.mean([w["rr_mean"] for w in clean_windows])),
                "baseline_map_std": float(np.mean([w["map_std"] for w in clean_windows])),
                "baseline_sbp": float(np.mean([w["sbp_max"] for w in clean_windows])),
            }
        else:
            baseline = {
                "baseline_spo2": 97.0,
                "baseline_rmssd": 35.0,
                "baseline_rr_ms": 833.0,
                "baseline_map_std": 5.0,
                "baseline_sbp": 120.0,
            }

        # Process each 30-second segment
        for i in range(n_segs):
            s = i * samples_per_seg
            e = s + samples_per_seg

            ecg_seg = _bandpass(ecg_full[s:e], fs)
            pleth_seg = pleth_full[s:e]
            resp_seg = resp_full[s:e]
            abp_seg = abp_full[s:e]

            r_peaks = _detect_r_peaks(ecg_seg, fs)

            # Stage 1: ingest raw
            raw_id = insert_apnea_raw(
                record_name, i, ecg_seg, pleth_seg, resp_seg, abp_seg, fs
            )

            # Stage 2: preprocess
            spo2_smooth = (
                pd.Series(pleth_seg)
                .rolling(int(fs * 2), center=True, min_periods=1)
                .median()
                .values
            )
            resp_smooth = (
                pd.Series(resp_seg)
                .rolling(int(fs * 1), center=True, min_periods=1)
                .median()
                .values
            )
            rr_ms = np.diff(r_peaks) / fs * 1000.0 if len(r_peaks) >= 2 else np.array([0.0])
            pre_id = insert_apnea_preprocessed(
                raw_id, ecg_seg, r_peaks, spo2_smooth, resp_smooth,
                float(np.mean(rr_ms)), float(np.std(rr_ms)),
                int(len(r_peaks)), float(np.median(rr_ms)),
            )

            # Stage 3: extract features + label
            feats = _extract_apnea_features(
                ecg_seg, pleth_seg, resp_seg, abp_seg, r_peaks, fs, baseline
            )
            insert_apnea_features(pre_id, json.dumps(feats))

            # Stage 1b: store labelled segment row
            seg_row = {
                "record": record_name,
                "segment_idx": i,
                **feats,
            }
            insert_apnea_segment(seg_row)
            total_segs += 1

            # Save one plot segment per label type (max 2 per record)
            if plot_saved < 8 and feats["true_label"] in (0, 1):
                spo2_1hz = np.interp(
                    np.linspace(0, 1, SEGMENT_LEN_S),
                    np.linspace(0, 1, len(spo2_smooth)),
                    spo2_smooth,
                )
                resp_1hz = np.interp(
                    np.linspace(0, 1, SEGMENT_LEN_S),
                    np.linspace(0, 1, len(resp_smooth)),
                    resp_smooth,
                )
                abp_1hz = np.interp(
                    np.linspace(0, 1, SEGMENT_LEN_S),
                    np.linspace(0, 1, len(abp_seg)),
                    abp_seg,
                )
                insert_apnea_ecg_plot(
                    record_name, i, ecg_seg, r_peaks,
                    spo2_1hz, resp_1hz, abp_1hz, fs,
                    feats["true_label"], feats["label_confidence"],
                )
                plot_saved += 1

        logger.info("[APNEA] %s: %d segments processed", record_name, n_segs)

    log_module("apnea", "ingest", "done", "Segments ingested", total_segs)

    if total_segs == 0:
        logger.error("[APNEA] No segments processed — aborting")
        return

    # ── Stage 4: Train Bidirectional LSTM ─────────────────────────────────────
    log_module("apnea", "train", "started")
    if not HAS_TF:
        logger.error("[APNEA] TensorFlow not installed — skipping LSTM training")
        log_module("apnea", "train", "failed", "tensorflow not installed", 0)
        return

    segs = fetch_apnea_segments()
    if len(segs) < 20:
        logger.warning("[APNEA] Only %d segments — need ≥20 for LSTM", len(segs))
        log_module("apnea", "train", "skipped", "Not enough segments", 0)
        return

    seg_df = pd.DataFrame(segs)
    X_all = seg_df[APNEA_FEATURE_COLS].fillna(0.0).values.astype(float)
    y_all = seg_df["true_label"].values.astype(float)

    TIMESTEPS = 10
    if len(X_all) <= TIMESTEPS:
        logger.warning("[APNEA] Not enough segments for sequence model")
        return

    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)

    X_seq = np.array([X_scaled[i:i + TIMESTEPS]
                      for i in range(len(X_scaled) - TIMESTEPS)])
    y_seq = np.array([y_all[i + TIMESTEPS]
                      for i in range(len(y_all) - TIMESTEPS)])

    split = int(0.8 * len(X_seq))
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True),
            input_shape=(TIMESTEPS, len(APNEA_FEATURE_COLS)),
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["AUC"])
    model.fit(X_tr, y_tr, epochs=10, batch_size=32,
              validation_split=0.1, verbose=0)

    y_prob = model.predict(X_te, verbose=0).flatten()

    if len(np.unique(y_te)) > 1:
        auc = roc_auc_score(y_te, y_prob)
        report = classification_report(
            y_te, (y_prob > 0.5).astype(int),
            target_names=["Normal", "Apnea"],
            output_dict=True,
        )
        logger.info("[APNEA] AUC-ROC: %.4f", auc)
        logger.info("\n%s", classification_report(
            y_te, (y_prob > 0.5).astype(int),
            target_names=["Normal", "Apnea"],
        ))
        insert_apnea_results(auc, report)
        log_module("apnea", "train", "done", f"auc={auc:.4f}", total_segs)
    else:
        logger.warning("[APNEA] Only one class in test set — AUC not computed")
        log_module("apnea", "train", "skipped", "Single class in test", 0)

    logger.info("[APNEA] Module complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — SEPSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _sep_preprocess_row(raw: Dict) -> Dict[str, Any]:
    """Derive additional clinical features for one sepsis patient row."""
    def g(k: str) -> float:
        try:
            return float(raw.get(k) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    sbp = g("sbp_mean")
    dbp = g("dbp_mean")
    return {
        "pulse_pressure": sbp - dbp,
        "shock_index": g("hr_mean") / (sbp + 1e-6),
        "spo2_rr_ratio": g("spo2_mean") / (g("respiratory_rate_mean") + 1e-6),
        "lactate_high": int(g("lactate_mmol") > 2.0),
        "map_low": int(g("map_mean") < 65.0),
    }


def run_sepsis_module() -> None:
    """Run the full sepsis pipeline: ingest → preprocess → features → train."""
    logger.info("=" * 60)
    logger.info(" SEPSIS MODULE")
    logger.info("=" * 60)

    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    log_module("sepsis", "ingest", "started")
    sep_path = _resolve("sepsis_icu_synthetic.csv", "sepsis icu synthetic.csv")
    if not sep_path:
        logger.error("[SEP] sepsis_icu_synthetic.csv not found in %s", DATA_DIR)
        log_module("sepsis", "ingest", "failed", "Sepsis CSV not found", 0)
        return

    df = pd.read_csv(sep_path, low_memory=False)
    logger.info("[SEP] Loaded %d patients | prevalence %.1f%%",
                len(df), df["sepsis_label"].mean() * 100)

    raw_rows = [
        (int(row["subject_id"]),
         json.dumps({k: (None if pd.isna(v) else v)
                     for k, v in row.items()}))
        for _, row in df.iterrows()
    ]
    insert_sep_raw(raw_rows)
    log_module("sepsis", "ingest", "done", "Raw patients stored", len(raw_rows))

    # ── Stage 2: Preprocess ───────────────────────────────────────────────────
    log_module("sepsis", "preprocess", "started")
    import sqlite3 as _sql
    con = _sql.connect(DB_PATH)
    con.row_factory = _sql.Row
    raw_db = con.execute(
        "SELECT id, subject_id, raw_json FROM sepsis_raw"
        " WHERE id NOT IN (SELECT raw_id FROM sepsis_preprocessed)"
    ).fetchall()
    con.close()

    pre_rows: List[tuple] = []
    for r in raw_db:
        raw = json.loads(r["raw_json"])
        p = _sep_preprocess_row(raw)
        pre_rows.append((
            r["id"], int(r["subject_id"]),
            p["pulse_pressure"], p["shock_index"],
            p["spo2_rr_ratio"], p["lactate_high"], p["map_low"],
        ))

    insert_sep_preprocessed(pre_rows)
    logger.info("[SEP] %d preprocessed rows stored", len(pre_rows))
    log_module("sepsis", "preprocess", "done", "", len(pre_rows))

    # ── Stage 3: Feature extraction ───────────────────────────────────────────
    log_module("sepsis", "features", "started")
    import sqlite3 as _sql
    con = _sql.connect(DB_PATH)
    con.row_factory = _sql.Row
    pre_df = pd.read_sql("""
        SELECT p.id, p.subject_id, p.pulse_pressure, p.shock_index,
               p.spo2_rr_ratio, p.lactate_high, p.map_low, r.raw_json
        FROM sepsis_preprocessed p
        JOIN sepsis_raw r ON r.id = p.raw_id
        WHERE p.id NOT IN (SELECT preprocessed_id FROM sepsis_features)
    """, con)
    con.close()

    def _parse_sep_cols(raw_json: str) -> Dict:
        d = json.loads(raw_json)
        out = {c: float(d.get(c) or 0.0)
               for c in SEPSIS_COLS if c in d}
        out["sepsis_label"] = int(d.get("sepsis_label") or 0)
        return out

    raw_feat = pre_df["raw_json"].apply(_parse_sep_cols).apply(pd.Series)
    sep_feat = pd.concat(
        [pre_df.drop(columns=["raw_json"]), raw_feat], axis=1
    ).fillna(0.0)
    # Add derived features
    sep_feat["shock_index_derived"] = (
        sep_feat.get("hr_mean", pd.Series(0.0, index=sep_feat.index)) /
        (sep_feat.get("sbp_mean", pd.Series(120.0, index=sep_feat.index)) + 1e-6)
    )
    sep_feat["temp_hr_product"] = (
        sep_feat.get("temp_celsius_mean", pd.Series(37.0, index=sep_feat.index)) *
        sep_feat.get("hr_mean", pd.Series(70.0, index=sep_feat.index))
    )

    f_cols = [c for c in sep_feat.columns
              if c not in ("id", "subject_id", "sepsis_label")]
    feat_rows: List[tuple] = []
    for i, (_, row) in enumerate(sep_feat.iterrows()):
        feat_rows.append((
            int(row["id"]),
            int(row["subject_id"]),
            int(row.get("sepsis_label", 0)),
            sep_feat[f_cols].iloc[i].to_json(),
        ))

    insert_sep_features(feat_rows)
    logger.info("[SEP] %d feature rows stored", len(feat_rows))
    log_module("sepsis", "features", "done", "", len(feat_rows))

    # ── Stage 4: Train & predict ──────────────────────────────────────────────
    log_module("sepsis", "train", "started")
    feat_load = fetch_sep_features()
    X = feat_load["feature_json"].apply(json.loads).apply(pd.Series).fillna(0.0)
    y = feat_load["sepsis_label"].astype(int)

    X_tr, X_te, y_tr, y_te, sid_tr, sid_te = train_test_split(
        X, y, feat_load["subject_id"],
        test_size=0.2, stratify=y, random_state=42,
    )
    scaler = StandardScaler()
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42,
    )
    model.fit(scaler.fit_transform(X_tr), y_tr)
    y_prob = model.predict_proba(scaler.transform(X_te))[:, 1]
    y_pred = (y_prob > 0.4).astype(int)
    auc = roc_auc_score(y_te, y_prob)
    report = classification_report(
        y_te, y_pred,
        target_names=["No Sepsis", "Sepsis"],
        output_dict=True,
    )
    accuracy = report["accuracy"]
    logger.info("[SEP] AUC-ROC: %.4f | Accuracy: %.4f", auc, accuracy)
    logger.info("\n%s", classification_report(
        y_te, y_pred, target_names=["No Sepsis", "Sepsis"]
    ))

    insert_sep_results(accuracy, auc, report)
    pred_rows = [
        (int(sid), str(int(yt)), str(int(yp)), float(ypr))
        for sid, yt, yp, ypr in zip(sid_te, y_te, y_pred, y_prob)
    ]
    insert_sep_predictions(pred_rows)

    # Save vitals time-series for 5 sample patients
    sample_patients = df.sample(min(5, len(df)), random_state=1)
    for _, row in sample_patients.iterrows():
        insert_sep_vitals_plot(
            subject_id=int(row["subject_id"]),
            hr=[float(row.get(f"hr_mean", 70))] * 24,
            spo2=[float(row.get("spo2_mean", 97))] * 24,
            bp=[float(row.get("sbp_mean", 120))] * 24,
            temp=[float(row.get("temp_celsius_mean", 37))] * 24,
            rr=[float(row.get("respiratory_rate_mean", 16))] * 24,
        )

    log_module("sepsis", "train", "done",
               f"auc={auc:.4f} acc={accuracy:.4f}", len(pred_rows))
    logger.info("[SEP] Module complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vital Signs ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/pipeline.py                  # run all three modules
  python pipeline/pipeline.py --module apnea   # run apnea only
  python pipeline/pipeline.py --fresh          # delete DB and start clean
        """,
    )
    parser.add_argument(
        "--module",
        choices=["arrhythmia", "apnea", "sepsis"],
        default=None,
        help="Run a single module (default: run all three)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing database before running",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing CSV data files (default: DATA_DIR env or '.')",
    )
    parser.add_argument(
        "--beats",
        type=int,
        default=BEATS_PER_TYPE,
        help=f"Max beats per arrhythmia class (default: {BEATS_PER_TYPE})",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=N_MIMIC_RECORDS,
        help=f"Number of MIMIC-IV records to load (default: {N_MIMIC_RECORDS})",
    )
    return parser.parse_args()


def main() -> None:
    """Pipeline entry point."""
    args = _parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir

    if args.fresh and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info("[DB] Existing database deleted.")

    init_db()

    if args.module == "arrhythmia" or args.module is None:
        run_arrhythmia_module(beats_per_type=args.beats)

    if args.module == "apnea" or args.module is None:
        run_apnea_module(n_records=args.records)

    if args.module == "sepsis" or args.module is None:
        run_sepsis_module()

    logger.info("[DONE] Pipeline complete. DB: %s", os.path.abspath(DB_PATH))


if __name__ == "__main__":
    main()