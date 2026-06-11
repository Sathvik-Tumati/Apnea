"""
db/database.py
==============
SQLite3 schema initialisation and all CRUD helpers for the
Vital Signs ML Pipeline (Apnea module only).

Schema:
  - apnea_raw
  - apnea_preprocessed
  - apnea_features
  - apnea_segments
  - apnea_ecg_plot
  - apnea_results
  - pipeline_log
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH: str = str(Path(__file__).resolve().parent.parent.parent / "vitals_pipeline.db")

def _j(arr) -> str:
    """Serialize a numpy array or Python list to a compact JSON string."""
    if arr is None:
        return "[]"
    if isinstance(arr, np.ndarray):
        return json.dumps(arr.tolist())
    return json.dumps(list(arr))

def _conn() -> sqlite3.Connection:
    """Return a WAL-mode connection with Row factory."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


_DDL = """
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
    fs          INTEGER NOT NULL DEFAULT 125,
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
    -- Resp features (EDR or GT)
    resp_amplitude_mean     REAL,
    resp_amplitude_std      REAL,
    flatline_duration_s     REAL,
    resp_rate_bpm           REAL,
    resp_rate_variability   REAL,
    -- ABP features (Low Reliability)
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
    -- FIX: tracks whether label came from GT Resp channel or EDR fallback.
    -- 'mimic_resp' = trustworthy label; 'edr' = circular fallback.
    -- Only 'mimic_resp' segments are used for LSTM training.
    label_source            TEXT    NOT NULL DEFAULT 'edr',
    run_id                  TEXT,
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
    fs              INTEGER NOT NULL DEFAULT 125,
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
CREATE INDEX IF NOT EXISTS idx_apnea_seg_label_source
    ON apnea_segments(label_source);

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
    global DB_PATH
    DB_PATH = db_path
    with _conn() as con:
        con.executescript(_DDL)
    logger.info("[DB] Schema initialised at %s", db_path)


def log_module(
    module: str,
    stage: str,
    status: str,
    message: str = "",
    rows: int = 0,
) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO pipeline_log(module,stage,status,message,rows_written)"
            " VALUES(?,?,?,?,?)",
            (module, stage, status, message, rows),
        )


def insert_apnea_raw(
    record: str,
    segment_idx: int,
    ecg: np.ndarray,
    ppg: np.ndarray,
    resp: np.ndarray,
    abp: np.ndarray,
    fs: int,
) -> int:
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
    with _conn() as con:
        con.execute(
            "INSERT INTO apnea_features(preprocessed_id,feature_json)"
            " VALUES(?,?)",
            (preprocessed_id, feature_json),
        )

def insert_apnea_segment(seg: Dict[str, Any]) -> None:
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
        "label_source",
        "run_id",
    ]
    values = tuple(seg.get(c) if c != "label_source" else seg.get(c, "edr") for c in cols)
    placeholders = ",".join(["?"] * len(cols))
    # Plain INSERT — run_id isolation means duplicates across runs are fine
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
    with _conn() as con:
        con.execute(
            "INSERT INTO apnea_results(auc_roc,report_json) VALUES(?,?)",
            (auc_roc, json.dumps(report)),
        )

def fetch_apnea_segments(
    record: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 5000,
) -> List[Dict]:
    with _conn() as con:
        conditions = []
        params = []
        if record:
            conditions.append("record=?")
            params.append(record)
        if run_id:
            conditions.append("run_id=?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        rows = con.execute(
            f"SELECT * FROM apnea_segments {where} LIMIT ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]

def fetch_apnea_ecg_plot(
    record: Optional[str] = None,
    segment_idx: Optional[int] = None,
) -> Optional[Dict]:
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