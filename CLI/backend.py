"""
backend.py
==========
FastAPI REST backend for the Vital Signs ML Dashboard.

Start with:
    ./start_backend.sh

All endpoints read from vitals_pipeline.db.
Run pipeline/pipeline.py first to populate the database.

Endpoint groups
---------------
/apnea/*        Apnea segment results, signal plots, flag breakdown
/summary        Combined counts across all modules
/pipeline_log   Stage execution audit trail
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

DB_PATH: str = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "vitals_pipeline.db"))

app = FastAPI(
    title="Vital Signs Dashboard API",
    description="REST API for Apnea ML pipeline results.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _db() -> sqlite3.Connection:
    """Open a read-only Row-factory connection, raise 503 if DB missing."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=503,
            detail="Database not found. Run pipeline/pipeline.py first.",
        )
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _q(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SELECT and return list of dicts."""
    con = _db()
    try:
        rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def _one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Execute a SELECT and return the first row as dict, or None."""
    rows = _q(sql, params)
    return rows[0] if rows else None


def _parse_report(rows: List[Dict]) -> List[Dict]:
    """Deserialise report_json fields in a list of result rows."""
    for r in rows:
        if r.get("report_json"):
            r["report"] = json.loads(r["report_json"])
            del r["report_json"]
    return rows

@app.exception_handler(Exception)
async def _global_handler(request, exc):
    """Return a generic 500 — never expose stack traces to clients."""
    logger.exception("Unhandled error on %s", request.url)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# SHARED ENDPOINTS


@app.get("/summary", tags=["Shared"])
def summary() -> Dict[str, Any]:
    """Combined record counts."""
    def _count(table: str) -> int:
        row = _one(f"SELECT COUNT(*) AS n FROM {table}")
        return row["n"] if row else 0

    return {
        "apnea_segments": _count("apnea_segments"),
        "pipeline_stages_run": _count("pipeline_log"),
    }


@app.get("/pipeline_log", tags=["Shared"])
def pipeline_log() -> List[Dict]:
    """All pipeline stage log entries ordered by timestamp."""
    return _q("SELECT * FROM pipeline_log ORDER BY ts")


@app.get("/pipeline_log/latest", tags=["Shared"])
def pipeline_log_latest() -> Dict:
    """Most recently completed pipeline stage."""
    return _one("SELECT * FROM pipeline_log ORDER BY ts DESC LIMIT 1") or {}


# APNEA ENDPOINTS


@app.get("/apnea/summary", tags=["Apnea"])
def apnea_summary() -> Dict[str, Any]:
    """Segment and result counts for the apnea module."""
    total = _one("SELECT COUNT(*) AS n FROM apnea_segments")
    apnea = _one(
        "SELECT COUNT(*) AS n FROM apnea_segments WHERE true_label=1"
    )
    normal = _one(
        "SELECT COUNT(*) AS n FROM apnea_segments WHERE true_label=0"
    )
    results = _one(
        "SELECT auc_roc FROM apnea_results ORDER BY trained_at DESC LIMIT 1"
    )
    return {
        "total_segments": total["n"] if total else 0,
        "apnea_segments": apnea["n"] if apnea else 0,
        "normal_segments": normal["n"] if normal else 0,
        "auc_roc": results["auc_roc"] if results else None,
    }


@app.get("/apnea/records", tags=["Apnea"])
def apnea_records() -> List[Dict]:
    """Return list of distinct patient record IDs with segment counts."""
    return _q("""
        SELECT record, COUNT(*) AS segment_count,
               SUM(true_label) AS apnea_count
        FROM apnea_segments
        GROUP BY record
        ORDER BY segment_count DESC
    """)


@app.get("/apnea/segments", tags=["Apnea"])
def apnea_segments(
    record: Optional[str] = Query(None),
    limit: int = Query(300, ge=1, le=5000),
) -> List[Dict]:
    """Return apnea segment feature rows, optionally filtered by record."""
    if record:
        return _q(
            "SELECT * FROM apnea_segments WHERE record=? LIMIT ?",
            (record, limit),
        )
    return _q("SELECT * FROM apnea_segments LIMIT ?", (limit,))


@app.get("/apnea/model_results", tags=["Apnea"])
def apnea_model_results() -> List[Dict]:
    """Apnea LSTM AUC-ROC and per-class classification report."""
    rows = _q(
        "SELECT auc_roc, report_json, trained_at FROM apnea_results ORDER BY trained_at DESC"
    )
    return _parse_report(rows)


@app.get("/apnea/ecg_plot", tags=["Apnea"])
def apnea_ecg_plot(
    record: Optional[str] = Query(None),
    segment_idx: Optional[int] = Query(None),
) -> Dict:
    """
    Return one apnea plot segment with ECG, SpO2, Resp, and ABP arrays.

    Arrays are pre-decoded from JSON into lists for direct frontend use.
    """
    if record and segment_idx is not None:
        row = _one(
            "SELECT * FROM apnea_ecg_plot"
            " WHERE record=? AND segment_idx=? LIMIT 1",
            (record, segment_idx),
        )
    elif record:
        row = _one(
            "SELECT * FROM apnea_ecg_plot WHERE record=? LIMIT 1",
            (record,),
        )
    else:
        row = _one(
            "SELECT * FROM apnea_ecg_plot ORDER BY RANDOM() LIMIT 1"
        )

    if not row:
        return {}

    for field in ("ecg_json", "r_peaks_json", "spo2_json",
                  "resp_json", "abp_json"):
        if row.get(field):
            row[field.replace("_json", "")] = json.loads(row.pop(field))

    return row


@app.get("/apnea/spo2_plot", tags=["Apnea"])
def apnea_spo2_plot(
    record: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> List[Dict]:
    """
    Return SpO2 stats per segment for a record — used by the SpO2 chart page.
    """
    if record:
        return _q(
            "SELECT segment_idx, spo2_mean, spo2_min, spo2_delta_index,"
            " t90, odi, true_label, label_confidence"
            " FROM apnea_segments WHERE record=? ORDER BY segment_idx LIMIT ?",
            (record, limit),
        )
    return _q(
        "SELECT segment_idx, spo2_mean, spo2_min, spo2_delta_index,"
        " t90, odi, true_label, label_confidence"
        " FROM apnea_segments ORDER BY segment_idx LIMIT ?",
        (limit,),
    )


@app.get("/apnea/resp_plot", tags=["Apnea"])
def apnea_resp_plot(
    record: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> List[Dict]:
    """
    Return Resp signal stats per segment — used by the Resp chart page.
    """
    if record:
        return _q(
            "SELECT segment_idx, resp_amplitude_mean, resp_amplitude_std,"
            " flatline_duration_s, resp_rate_bpm, resp_rate_variability,"
            " true_label, label_confidence"
            " FROM apnea_segments WHERE record=? ORDER BY segment_idx LIMIT ?",
            (record, limit),
        )
    return _q(
        "SELECT segment_idx, resp_amplitude_mean, resp_amplitude_std,"
        " flatline_duration_s, resp_rate_bpm, resp_rate_variability,"
        " true_label, label_confidence"
        " FROM apnea_segments ORDER BY segment_idx LIMIT ?",
        (limit,),
    )


@app.get("/apnea/label_distribution", tags=["Apnea"])
def apnea_label_distribution(
    record: Optional[str] = Query(None),
) -> List[Dict]:
    """Counts per label_confidence category, optionally filtered by record."""
    if record:
        return _q("""
            SELECT label_confidence, COUNT(*) AS count
            FROM apnea_segments
            WHERE record=?
            GROUP BY label_confidence
            ORDER BY count DESC
        """, (record,))
    return _q("""
        SELECT label_confidence, COUNT(*) AS count
        FROM apnea_segments
        GROUP BY label_confidence
        ORDER BY count DESC
    """)


@app.get("/apnea/signal_flags", tags=["Apnea"])
def apnea_signal_flags(
    record: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
) -> List[Dict]:
    """
    Return per-segment signal flag breakdown, optionally filtered by record.
    Shows which of the 3 signals (Resp, SpO2, HRV) fired for each segment.
    """
    if record:
        return _q("""
            SELECT record, segment_idx,
                   resp_flag, spo2_flag, hrv_flag, abp_flag,
                   signals_positive, true_label, label_confidence
            FROM apnea_segments
            WHERE record=?
            ORDER BY signals_positive DESC, segment_idx
            LIMIT ?
        """, (record, limit))
    return _q("""
        SELECT record, segment_idx,
               resp_flag, spo2_flag, hrv_flag, abp_flag,
               signals_positive, true_label, label_confidence
        FROM apnea_segments
        ORDER BY signals_positive DESC, segment_idx
        LIMIT ?
    """, (limit,))


@app.get("/apnea/feature_importance", tags=["Apnea"])
def apnea_feature_importance() -> List[Dict]:
    """
    Return aggregate feature importance proxy from apnea segments.

    Uses variance of each numeric feature across apnea vs normal segments
    as a proxy for discriminative power (actual model importance requires
    the serialised model — this provides a lightweight DB-only approximation).
    """
    all_segs = _q("SELECT * FROM apnea_segments LIMIT 2000")
    if not all_segs:
        return []

    import statistics

    numeric_cols = [
        "rr_mean", "rr_std", "rmssd", "pnn50", "lf_hf_ratio",
        "mean_hr", "hr_range",
        "spo2_mean", "spo2_min", "spo2_delta_index", "odi", "t90",
        "resp_amplitude_mean", "resp_amplitude_std",
        "flatline_duration_s", "resp_rate_bpm",
        "map_mean", "map_std", "sbp_max", "pulse_pressure",
        "resp_spo2_lag_s", "ptt_ms", "ecg_resp_coherence",
    ]

    apnea = [s for s in all_segs if s["true_label"] == 1]
    normal = [s for s in all_segs if s["true_label"] == 0]

    result = []
    for col in numeric_cols:
        ap_vals = [s[col] for s in apnea if s.get(col) is not None]
        no_vals = [s[col] for s in normal if s.get(col) is not None]
        if not ap_vals or not no_vals:
            continue
        diff = abs(
            statistics.mean(ap_vals) - statistics.mean(no_vals)
        ) / (statistics.stdev(ap_vals + no_vals) + 1e-6)
        result.append({"feature": col, "importance": round(diff, 4)})

    return sorted(result, key=lambda x: x["importance"], reverse=True)