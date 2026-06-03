"""
db/database.py
==============
SQLite3 schema initialisation and all CRUD helpers for the
Vital Signs ML Pipeline.

Three fully isolated module schemas:
  - arrhythmia_*   : beat-level ECG classification
  - apnea_*        : MIMIC-IV multi-signal apnea detection
  - sepsis_*       : ICU sepsis early warning

Shared:
  - pipeline_log   : stage execution audit trail

Public API
----------
init_db()                     create all tables
log_module(module, stage, status, message, rows)
                              insert a pipeline log row
_j(arr)  / _uj(s)             numpy → JSON str / JSON str → numpy

Arrhythmia helpers:
  insert_arr_raw, insert_arr_preprocessed, insert_arr_features,
  insert_arr_ecg_plot, insert_arr_results, insert_arr_predictions,
  fetch_arr_features, fetch_arr_predictions

Apnea helpers:
  insert_apnea_raw, insert_apnea_preprocessed, insert_apnea_features,
  insert_apnea_segment, insert_apnea_ecg_plot, insert_apnea_results,
  fetch_apnea_segments, fetch_apnea_ecg_plot

Sepsis helpers:
  insert_sep_raw, insert_sep_preprocessed, insert_sep_features,
  insert_sep_vitals_plot, insert_sep_results, insert_sep_predictions,
  fetch_sep_features, fetch_sep_predictions
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH: str = str(Path(__file__).resolve().parent.parent.parent / "vitals_pipeline.db")

# ── JSON helpers ──────────────────────────────────────────────────────────────

def _j(arr) -> str:
    """Serialize a numpy array or Python list to a compact JSON string."""
    if arr is None:
        return "[]"
    if isinstance(arr, np.ndarray):
        return json.dumps(arr.tolist())
    return json.dumps(list(arr))


def _uj(s: Optional[str]) -> np.ndarray:
    """Deserialize a JSON string back to a numpy array."""
    if not s:
        return np.array([])
    return np.array(json.loads(s))


# ── Connection factory ────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """Return a WAL-mode connection with Row factory."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


# ── Schema DDL ────────────────────────────────────────────────────────────────

_DDL = """
-- ═══════════════════════════════════════
--  ARRHYTHMIA MODULE
-- ═══════════════════════════════════════
CREATE TABLE IF NOT EXISTS arrhythmia_raw (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    record      TEXT,
    beat_type   TEXT    NOT NULL,
    raw_json    TEXT    NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arrhythmia_preprocessed (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id          INTEGER NOT NULL REFERENCES arrhythmia_raw(id),
    beat_type       TEXT    NOT NULL,
    rr_ratio        REAL,
    rr_diff         REAL,
    rr_symmetry     REAL,
    qrs_amplitude   REAL,
    qrs_diff_leads  REAL,
    st_diff_leads   REAL,
    p_absent        INTEGER,
    qtc_approx      REAL,
    processed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arrhythmia_features (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    preprocessed_id   INTEGER NOT NULL REFERENCES arrhythmia_preprocessed(id),
    beat_type         TEXT    NOT NULL,
    feature_json      TEXT    NOT NULL,
    extracted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arrhythmia_ecg_plot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    record          TEXT,
    beat_type       TEXT    NOT NULL,
    ecg_json        TEXT    NOT NULL,
    r_peaks_json    TEXT,
    p_peaks_json    TEXT,
    q_peaks_json    TEXT,
    s_peaks_json    TEXT,
    t_peaks_json    TEXT,
    fs              INTEGER NOT NULL DEFAULT 360,
    stored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arrhythmia_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    accuracy    REAL,
    report_json TEXT,
    trained_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arrhythmia_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    beat_ref        TEXT,
    true_label      TEXT,
    predicted_label TEXT,
    confidence      REAL,
    predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_arr_raw_type
    ON arrhythmia_raw(beat_type);
CREATE INDEX IF NOT EXISTS idx_arr_pred_labels
    ON arrhythmia_predictions(true_label, predicted_label);

-- ═══════════════════════════════════════
--  APNEA MODULE
-- ═══════════════════════════════════════
CREATE TABLE IF NOT EXISTS apnea_raw (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record      TEXT    NOT NULL,
    segment_idx INTEGER NOT NULL,
    ecg_json    TEXT,
    ppg_json    TEXT,
    resp_json   TEXT,
    abp_json    TEXT,
    fs          INTEGER NOT NULL DEFAULT 320,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS apnea_preprocessed (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id              INTEGER NOT NULL REFERENCES apnea_raw(id),
    ecg_clean_json      TEXT,
    r_peaks_json        TEXT,
    spo2_smoothed_json  TEXT,
    resp_smoothed_json  TEXT,
    rr_mean             REAL,
    rr_std              REAL,
    peak_count          INTEGER,
    rr_median           REAL,
    processed_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS apnea_features (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    preprocessed_id   INTEGER NOT NULL REFERENCES apnea_preprocessed(id),
    feature_json      TEXT    NOT NULL,
    extracted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS apnea_segments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    record                  TEXT    NOT NULL,
    segment_idx             INTEGER NOT NULL,
    -- HRV features
    rr_mean                 REAL,
    rr_std                  REAL,
    rmssd                   REAL,
    pnn50                   REAL,
    lf_hf_ratio             REAL,
    mean_hr                 REAL,
    hr_range                REAL,
    -- SpO2 features
    spo2_mean               REAL,
    spo2_min                REAL,
    spo2_delta_index        REAL,
    odi                     REAL,
    t90                     REAL,
    spo2_approx_entropy     REAL,
    -- Resp features
    resp_amplitude_mean     REAL,
    resp_amplitude_std      REAL,
    flatline_duration_s     REAL,
    resp_rate_bpm           REAL,
    resp_rate_variability   REAL,
    -- ABP features
    map_mean                REAL,
    map_std                 REAL,
    sbp_max                 REAL,
    dbp_min                 REAL,
    pulse_pressure          REAL,
    map_variability         REAL,
    -- Cross-signal features
    resp_spo2_lag_s         REAL,
    ptt_ms                  REAL,
    ecg_resp_coherence      REAL,
    -- Label info
    true_label              INTEGER NOT NULL DEFAULT 0,
    label_confidence        TEXT    NOT NULL DEFAULT 'normal',
    resp_flag               INTEGER NOT NULL DEFAULT 0,
    spo2_flag               INTEGER NOT NULL DEFAULT 0,
    hrv_flag                INTEGER NOT NULL DEFAULT 0,
    abp_flag                INTEGER NOT NULL DEFAULT 0,
    signals_positive        INTEGER NOT NULL DEFAULT 0,
    ingested_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS apnea_ecg_plot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    record          TEXT    NOT NULL,
    segment_idx     INTEGER NOT NULL,
    ecg_json        TEXT,
    r_peaks_json    TEXT,
    spo2_json       TEXT,
    resp_json       TEXT,
    abp_json        TEXT,
    fs              INTEGER NOT NULL DEFAULT 320,
    true_label      INTEGER NOT NULL DEFAULT 0,
    label_confidence TEXT   NOT NULL DEFAULT 'normal',
    stored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS apnea_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    auc_roc     REAL,
    report_json TEXT,
    trained_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_apnea_seg_record
    ON apnea_segments(record);
CREATE INDEX IF NOT EXISTS idx_apnea_seg_label
    ON apnea_segments(true_label);

-- ═══════════════════════════════════════
--  SEPSIS MODULE
-- ═══════════════════════════════════════
CREATE TABLE IF NOT EXISTS sepsis_raw (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  INTEGER NOT NULL,
    raw_json    TEXT    NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sepsis_preprocessed (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id          INTEGER NOT NULL REFERENCES sepsis_raw(id),
    subject_id      INTEGER NOT NULL,
    pulse_pressure  REAL,
    shock_index     REAL,
    spo2_rr_ratio   REAL,
    lactate_high    INTEGER,
    map_low         INTEGER,
    processed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sepsis_features (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    preprocessed_id       INTEGER NOT NULL REFERENCES sepsis_preprocessed(id),
    subject_id            INTEGER NOT NULL,
    sepsis_label          INTEGER NOT NULL,
    feature_json          TEXT    NOT NULL,
    extracted_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sepsis_vitals_plot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id      INTEGER NOT NULL,
    hr_series_json  TEXT,
    spo2_series_json TEXT,
    bp_series_json  TEXT,
    temp_series_json TEXT,
    rr_series_json  TEXT,
    stored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sepsis_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    accuracy    REAL,
    auc_roc     REAL,
    report_json TEXT,
    trained_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sepsis_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id      INTEGER,
    true_label      TEXT,
    predicted_label TEXT,
    confidence      REAL,
    predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sep_pred_conf
    ON sepsis_predictions(confidence);

-- ═══════════════════════════════════════
--  SHARED
-- ═══════════════════════════════════════
CREATE TABLE IF NOT EXISTS pipeline_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    module       TEXT    NOT NULL,
    stage        TEXT    NOT NULL,
    status       TEXT    NOT NULL,
    message      TEXT,
    rows_written INTEGER NOT NULL DEFAULT 0,
    ts           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create all tables and indexes if they do not already exist."""
    global DB_PATH
    DB_PATH = db_path
    with _conn() as con:
        con.executescript(_DDL)
    logger.info("[DB] Schema initialised at %s", db_path)


# ── Shared log helper ─────────────────────────────────────────────────────────

def log_module(
    module: str,
    stage: str,
    status: str,
    message: str = "",
    rows: int = 0,
) -> None:
    """Insert one row into pipeline_log."""
    with _conn() as con:
        con.execute(
            "INSERT INTO pipeline_log(module,stage,status,message,rows_written)"
            " VALUES(?,?,?,?,?)",
            (module, stage, status, message, rows),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ARRHYTHMIA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def insert_arr_raw(rows: List[tuple]) -> None:
    """Bulk-insert raw arrhythmia beats.

    Each tuple: (source, record, beat_type, raw_json)
    """
    with _conn() as con:
        con.executemany(
            "INSERT INTO arrhythmia_raw(source,record,beat_type,raw_json)"
            " VALUES(?,?,?,?)",
            rows,
        )


def insert_arr_preprocessed(rows: List[tuple]) -> None:
    """Bulk-insert preprocessed arrhythmia rows.

    Each tuple: (raw_id, beat_type, rr_ratio, rr_diff, rr_symmetry,
                 qrs_amplitude, qrs_diff_leads, st_diff_leads,
                 p_absent, qtc_approx)
    """
    with _conn() as con:
        con.executemany(
            "INSERT INTO arrhythmia_preprocessed"
            "(raw_id,beat_type,rr_ratio,rr_diff,rr_symmetry,"
            " qrs_amplitude,qrs_diff_leads,st_diff_leads,p_absent,qtc_approx)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            rows,
        )


def insert_arr_features(rows: List[tuple]) -> None:
    """Bulk-insert arrhythmia feature rows.

    Each tuple: (preprocessed_id, beat_type, feature_json)
    """
    chunk = 500
    with _conn() as con:
        for i in range(0, len(rows), chunk):
            con.executemany(
                "INSERT INTO arrhythmia_features"
                "(preprocessed_id,beat_type,feature_json)"
                " VALUES(?,?,?)",
                rows[i : i + chunk],
            )


def insert_arr_ecg_plot(
    record: str,
    beat_type: str,
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    p_peaks: np.ndarray,
    q_peaks: np.ndarray,
    s_peaks: np.ndarray,
    t_peaks: np.ndarray,
    fs: int,
) -> None:
    """Store one annotated ECG beat segment for the frontend."""
    with _conn() as con:
        con.execute(
            "INSERT INTO arrhythmia_ecg_plot"
            "(record,beat_type,ecg_json,r_peaks_json,p_peaks_json,"
            " q_peaks_json,s_peaks_json,t_peaks_json,fs)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                record, beat_type,
                _j(ecg), _j(r_peaks), _j(p_peaks),
                _j(q_peaks), _j(s_peaks), _j(t_peaks), fs,
            ),
        )


def insert_arr_results(accuracy: float, report: Dict[str, Any]) -> None:
    """Store arrhythmia model evaluation results."""
    with _conn() as con:
        con.execute(
            "INSERT INTO arrhythmia_results(accuracy,report_json)"
            " VALUES(?,?)",
            (accuracy, json.dumps(report)),
        )


def insert_arr_predictions(rows: List[tuple]) -> None:
    """Bulk-insert arrhythmia predictions.

    Each tuple: (beat_ref, true_label, predicted_label, confidence)
    """
    with _conn() as con:
        con.executemany(
            "INSERT INTO arrhythmia_predictions"
            "(beat_ref,true_label,predicted_label,confidence)"
            " VALUES(?,?,?,?)",
            rows,
        )


def fetch_arr_features() -> "pd.DataFrame":
    """Return all arrhythmia feature rows as a DataFrame."""
    import pandas as pd
    with _conn() as con:
        return pd.read_sql(
            "SELECT id, beat_type, feature_json FROM arrhythmia_features",
            con,
        )


def fetch_arr_predictions(limit: int = 200) -> List[Dict]:
    """Return a random sample of arrhythmia predictions."""
    with _conn() as con:
        rows = con.execute(
            "SELECT beat_ref, true_label, predicted_label, confidence"
            " FROM arrhythmia_predictions ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# APNEA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def insert_apnea_raw(
    record: str,
    segment_idx: int,
    ecg: np.ndarray,
    ppg: np.ndarray,
    resp: np.ndarray,
    abp: np.ndarray,
    fs: int,
) -> int:
    """Insert one raw apnea segment. Returns the new row id."""
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO apnea_raw"
            "(record,segment_idx,ecg_json,ppg_json,resp_json,abp_json,fs)"
            " VALUES(?,?,?,?,?,?,?)",
            (record, segment_idx, _j(ecg), _j(ppg), _j(resp), _j(abp), fs),
        )
        return cur.lastrowid


def insert_apnea_preprocessed(
    raw_id: int,
    ecg_clean: np.ndarray,
    r_peaks: np.ndarray,
    spo2_smoothed: np.ndarray,
    resp_smoothed: np.ndarray,
    rr_mean: float,
    rr_std: float,
    peak_count: int,
    rr_median: float,
) -> int:
    """Insert preprocessed apnea data. Returns the new row id."""
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO apnea_preprocessed"
            "(raw_id,ecg_clean_json,r_peaks_json,spo2_smoothed_json,"
            " resp_smoothed_json,rr_mean,rr_std,peak_count,rr_median)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                raw_id, _j(ecg_clean), _j(r_peaks),
                _j(spo2_smoothed), _j(resp_smoothed),
                rr_mean, rr_std, peak_count, rr_median,
            ),
        )
        return cur.lastrowid


def insert_apnea_features(preprocessed_id: int, feature_json: str) -> None:
    """Insert one apnea feature row."""
    with _conn() as con:
        con.execute(
            "INSERT INTO apnea_features(preprocessed_id,feature_json)"
            " VALUES(?,?)",
            (preprocessed_id, feature_json),
        )


def insert_apnea_segment(seg: Dict[str, Any]) -> None:
    """Insert one fully-labelled apnea segment row."""
    cols = [
        "record", "segment_idx",
        "rr_mean", "rr_std", "rmssd", "pnn50", "lf_hf_ratio",
        "mean_hr", "hr_range",
        "spo2_mean", "spo2_min", "spo2_delta_index", "odi", "t90",
        "spo2_approx_entropy",
        "resp_amplitude_mean", "resp_amplitude_std",
        "flatline_duration_s", "resp_rate_bpm", "resp_rate_variability",
        "map_mean", "map_std", "sbp_max", "dbp_min",
        "pulse_pressure", "map_variability",
        "resp_spo2_lag_s", "ptt_ms", "ecg_resp_coherence",
        "true_label", "label_confidence",
        "resp_flag", "spo2_flag", "hrv_flag", "abp_flag", "signals_positive",
    ]
    values = tuple(seg.get(c) for c in cols)
    placeholders = ",".join(["?"] * len(cols))
    sql = (
        f"INSERT INTO apnea_segments({','.join(cols)})"
        f" VALUES({placeholders})"
    )
    with _conn() as con:
        con.execute(sql, values)


def insert_apnea_ecg_plot(
    record: str,
    segment_idx: int,
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    spo2: np.ndarray,
    resp: np.ndarray,
    abp: np.ndarray,
    fs: int,
    true_label: int,
    label_confidence: str,
) -> None:
    """Store one apnea plot segment for the frontend."""
    with _conn() as con:
        con.execute(
            "INSERT INTO apnea_ecg_plot"
            "(record,segment_idx,ecg_json,r_peaks_json,spo2_json,"
            " resp_json,abp_json,fs,true_label,label_confidence)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                record, segment_idx,
                _j(ecg), _j(r_peaks), _j(spo2),
                _j(resp), _j(abp), fs,
                true_label, label_confidence,
            ),
        )


def insert_apnea_results(auc_roc: float, report: Dict[str, Any]) -> None:
    """Store apnea model results."""
    with _conn() as con:
        con.execute(
            "INSERT INTO apnea_results(auc_roc,report_json) VALUES(?,?)",
            (auc_roc, json.dumps(report)),
        )


def fetch_apnea_segments(
    record: Optional[str] = None,
    limit: int = 500,
) -> List[Dict]:
    """Return apnea segment rows, optionally filtered by record."""
    with _conn() as con:
        if record:
            rows = con.execute(
                "SELECT * FROM apnea_segments WHERE record=? LIMIT ?",
                (record, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM apnea_segments LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def fetch_apnea_ecg_plot(
    record: Optional[str] = None,
    segment_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Return one apnea ECG plot row."""
    with _conn() as con:
        if record and segment_idx is not None:
            row = con.execute(
                "SELECT * FROM apnea_ecg_plot"
                " WHERE record=? AND segment_idx=? LIMIT 1",
                (record, segment_idx),
            ).fetchone()
        elif record:
            row = con.execute(
                "SELECT * FROM apnea_ecg_plot WHERE record=? LIMIT 1",
                (record,),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT * FROM apnea_ecg_plot ORDER BY RANDOM() LIMIT 1",
            ).fetchone()
    return dict(row) if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# SEPSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def insert_sep_raw(rows: List[tuple]) -> None:
    """Bulk-insert raw sepsis rows.

    Each tuple: (subject_id, raw_json)
    """
    chunk = 1000
    with _conn() as con:
        for i in range(0, len(rows), chunk):
            con.executemany(
                "INSERT INTO sepsis_raw(subject_id,raw_json) VALUES(?,?)",
                rows[i : i + chunk],
            )


def insert_sep_preprocessed(rows: List[tuple]) -> None:
    """Bulk-insert preprocessed sepsis rows.

    Each tuple: (raw_id, subject_id, pulse_pressure, shock_index,
                 spo2_rr_ratio, lactate_high, map_low)
    """
    with _conn() as con:
        con.executemany(
            "INSERT INTO sepsis_preprocessed"
            "(raw_id,subject_id,pulse_pressure,shock_index,"
            " spo2_rr_ratio,lactate_high,map_low)"
            " VALUES(?,?,?,?,?,?,?)",
            rows,
        )


def insert_sep_features(rows: List[tuple]) -> None:
    """Bulk-insert sepsis feature rows.

    Each tuple: (preprocessed_id, subject_id, sepsis_label, feature_json)
    """
    chunk = 1000
    with _conn() as con:
        for i in range(0, len(rows), chunk):
            con.executemany(
                "INSERT INTO sepsis_features"
                "(preprocessed_id,subject_id,sepsis_label,feature_json)"
                " VALUES(?,?,?,?)",
                rows[i : i + chunk],
            )


def insert_sep_vitals_plot(
    subject_id: int,
    hr: list,
    spo2: list,
    bp: list,
    temp: list,
    rr: list,
) -> None:
    """Store vital time-series for one patient."""
    with _conn() as con:
        con.execute(
            "INSERT INTO sepsis_vitals_plot"
            "(subject_id,hr_series_json,spo2_series_json,"
            " bp_series_json,temp_series_json,rr_series_json)"
            " VALUES(?,?,?,?,?,?)",
            (
                subject_id,
                json.dumps(hr), json.dumps(spo2),
                json.dumps(bp), json.dumps(temp), json.dumps(rr),
            ),
        )


def insert_sep_results(
    accuracy: float,
    auc_roc: float,
    report: Dict[str, Any],
) -> None:
    """Store sepsis model results."""
    with _conn() as con:
        con.execute(
            "INSERT INTO sepsis_results(accuracy,auc_roc,report_json)"
            " VALUES(?,?,?)",
            (accuracy, auc_roc, json.dumps(report)),
        )


def insert_sep_predictions(rows: List[tuple]) -> None:
    """Bulk-insert sepsis predictions.

    Each tuple: (subject_id, true_label, predicted_label, confidence)
    """
    with _conn() as con:
        con.executemany(
            "INSERT INTO sepsis_predictions"
            "(subject_id,true_label,predicted_label,confidence)"
            " VALUES(?,?,?,?)",
            rows,
        )


def fetch_sep_features() -> "pd.DataFrame":
    """Return all sepsis feature rows as a DataFrame."""
    import pandas as pd
    with _conn() as con:
        return pd.read_sql(
            "SELECT id, subject_id, sepsis_label, feature_json"
            " FROM sepsis_features",
            con,
        )


def fetch_sep_predictions(limit: int = 200) -> List[Dict]:
    """Return top-confidence sepsis predictions."""
    with _conn() as con:
        rows = con.execute(
            "SELECT subject_id, true_label, predicted_label, confidence"
            " FROM sepsis_predictions ORDER BY confidence DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]