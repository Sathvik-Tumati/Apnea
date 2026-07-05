from typing import Any, Dict, List, Optional, Tuple
import datetime
import json
import logging
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from pipeline.db.database import fetch_apnea_segments, insert_apnea_results, log_module
from pipeline.modules.config import *
from pipeline.modules.config import (
    _SPO2_IDXS, _HAS_SPO2_IDX, _ABP_IDXS, _CROSS_IDXS, _HAS_ABP_IDX
)
from pipeline.modules.ingest_mimic import ingest_mimic_records
from pipeline.modules.ingest_slpdb import ingest_slpdb_records

_NumpyEncoder = NumpyEncoder
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  MODALITY DROPOUT  (training augmentation)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_modality_dropout_sequences(
    X_seq: np.ndarray,
    source_labels: np.ndarray,
    drop_spo2_prob: float = 0.30,
    drop_abp_prob:  float = 0.30,
) -> np.ndarray:
    """
    Apply modality dropout to sequence tensor (N, T, F).
    Only applied to MIMIC segments (source_label == 'mimic').
    """
    X_aug   = X_seq.copy()
    is_mimic = source_labels == 0   # 0=mimic, 1=slpdb
    mimic_idx = np.where(is_mimic)[0]
    if len(mimic_idx) == 0:
        return X_aug

    # Per-sample dropout decision
    drop_spo2 = np.random.rand(len(mimic_idx)) < drop_spo2_prob
    drop_abp  = np.random.rand(len(mimic_idx)) < drop_abp_prob

    for k, mi in enumerate(mimic_idx):
        if drop_spo2[k]:
            X_aug[mi, :, _SPO2_IDXS]   = 0.0
            X_aug[mi, :, _HAS_SPO2_IDX] = 0.0
        if drop_abp[k]:
            X_aug[mi, :, _ABP_IDXS]    = 0.0
            X_aug[mi, :, _CROSS_IDXS]  = 0.0
            X_aug[mi, :, _HAS_ABP_IDX] = 0.0

    return X_aug


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENCE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def _build_sequences(
    X: np.ndarray, y: np.ndarray, timesteps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.array([X[i: i + timesteps] for i in range(len(X) - timesteps)])
    ys = np.array([y[i + timesteps]    for i in range(len(y) - timesteps)])
    return xs, ys



# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_combined_dataset(
    mimic_df: pd.DataFrame,
    slpdb_rows: List[Dict],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Merge MIMIC and SLPDB segments, fit/apply scaler, return:
      X_all  (N, n_features)
      y_all  (N,)
      source (N,) — 0=mimic, 1=slpdb
      scaler
    """
    # MIMIC rows
    mimic_X = mimic_df[APNEA_FEATURE_COLS].fillna(0.0).values.astype(float)
    mimic_y = mimic_df["true_label"].values.astype(float)
    mimic_src = np.zeros(len(mimic_X), dtype=int)

    # SLPDB rows
    if slpdb_rows:
        slpdb_df = pd.DataFrame(slpdb_rows)
        for c in APNEA_FEATURE_COLS:
            if c not in slpdb_df.columns:
                slpdb_df[c] = 0.0
        slpdb_X   = slpdb_df[APNEA_FEATURE_COLS].fillna(0.0).values.astype(float)
        slpdb_y   = slpdb_df["true_label"].values.astype(float)
        slpdb_src = np.ones(len(slpdb_X), dtype=int)

        X_all   = np.vstack([mimic_X,   slpdb_X])
        y_all   = np.concatenate([mimic_y,   slpdb_y])
        src_all = np.concatenate([mimic_src, slpdb_src])
    else:
        X_all   = mimic_X
        y_all   = mimic_y
        src_all = mimic_src

    # Fit scaler on ALL data so both domains are normalised consistently
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_all)
    X_scaled = scaler.transform(X_all)

    logger.info(
        "[DATASET] Total: %d segments | MIMIC: %d | SLPDB: %d | "
        "apnea: %d (%.0f%%)",
        len(y_all),
        int(mimic_src.sum() == 0) * len(mimic_src),
        len(slpdb_rows) if slpdb_rows else 0,
        int(y_all.sum()),
        y_all.mean() * 100,
    )
    return X_scaled, y_all, src_all, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_apnea_module(
    n_records:      int  = N_MIMIC_RECORDS,
    save_model:     bool = False,
    use_slpdb:      bool = True,
    skip_slpdb:     bool = False,   # alias accepted from pipeline.py --no-slpdb
    fresh:          bool = False,   # if True, wipes DB before run
    slpdb_records:  Optional[List[str]] = None,
    slpdb_max_segs: int = 80,
) -> None:
    # Resolve slpdb flag (skip_slpdb takes priority)
    if skip_slpdb:
        use_slpdb = False

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 65)
    logger.info("  APNEA MODULE  run_id=%s  slpdb=%s", run_id, use_slpdb)
    logger.info("=" * 65)


    # ── STAGE 1: Ingest MIMIC ────────────────────────────────────────────────
    mimic_rows = ingest_mimic_records(n_records=n_records, run_id=run_id)
    total_segs = len(mimic_rows)

    if total_segs == 0:
        log_module("apnea", "ingest", "failed", "No MIMIC records fetched", 0)
        return

    # ── STAGE 2: Ingest SLPDB ────────────────────────────────────────────────
    slpdb_rows: List[Dict] = []
    if use_slpdb:
        records_to_use = slpdb_records or SLPDB_RECORDS
        logger.info("[SLPDB] Ingesting %d records ...", len(records_to_use))
        slpdb_rows = ingest_slpdb_records(records_to_use, run_id, slpdb_max_segs)
        logger.info("[SLPDB] Total SLPDB segments ingested: %d", len(slpdb_rows))
        
        if len(slpdb_rows) == 0:
            logger.warning("[SLPDB] No SLPDB segments loaded — training will use MIMIC only")

    # ── STAGE 3: Prepare Dataset for Training ─────────────────────────────────
    log_module("apnea", "train", "started")

    # Build MIMIC DataFrame from DB (most up-to-date version)
    segs = fetch_apnea_segments(run_id=run_id)
    if len(segs) == 0:
        logger.warning("[APNEA] No MIMIC segments for run_id=%s", run_id)
        log_module("apnea", "train", "skipped", "No segments", 0)
        return

    seg_df = pd.DataFrame(segs)

    # Filter: GT-labelled MIMIC segments only
    if "label_source" in seg_df.columns:
        gt_mask = seg_df["label_source"] == "mimic_resp"
        dropped = int((~gt_mask).sum())
        seg_df = seg_df[gt_mask].reset_index(drop=True)
        logger.info(
            "[MIMIC] Label filter: %d / %d segments have GT Resp labels (dropped %d EDR-only)",
            len(seg_df), len(segs), dropped,
        )

    # Quality gate
    if "mean_hr" in seg_df.columns:
        q_mask  = seg_df["mean_hr"] > 0
        dropped = int((~q_mask).sum())
        if dropped:
            logger.info("[MIMIC] Quality filter: dropped %d segments with mean_hr=0", dropped)
        seg_df = seg_df[q_mask].reset_index(drop=True)

    if "true_label" not in seg_df.columns or len(seg_df) == 0:
        logger.error("[APNEA] No valid MIMIC segments after filtering")
        log_module("apnea", "train", "skipped", "No valid segments", 0)
        return

    # Add modality flag columns if missing (backward compat)
    for col, default in [("has_spo2", 1), ("has_abp", 1), ("has_resp_gt", 1)]:
        if col not in seg_df.columns:
            seg_df[col] = default

    # Validate MIMIC label distribution
    n_apnea  = int((seg_df["true_label"] == 1).sum())
    n_normal = int((seg_df["true_label"] == 0).sum())
    apnea_pct = n_apnea / max(len(seg_df), 1)
    logger.info(
        "[MIMIC] Training set: %d segments — %d apnea / %d normal (%.0f%%)",
        len(seg_df), n_apnea, n_normal, apnea_pct * 100,
    )
    if apnea_pct > 0.70:
        logger.error("[MIMIC] %.0f%% apnea — labelling likely broken", apnea_pct * 100)
        log_module("apnea", "train", "skipped", "Implausible label distribution", 0)
        return
    if n_apnea == 0 or n_normal == 0:
        logger.error("[MIMIC] Only one class — cannot train")
        log_module("apnea", "train", "skipped", "Single class", 0)
        return
    if len(seg_df) < 20:
        logger.warning("[MIMIC] Only %d segments — need ≥20", len(seg_df))
        log_module("apnea", "train", "skipped", "Not enough segments", 0)
        return

    # Build combined dataset (MIMIC + SLPDB), fit scaler on all data
    X_scaled, y_all, src_all, scaler = _build_combined_dataset(seg_df, slpdb_rows)

    if len(X_scaled) <= TIMESTEPS:
        logger.warning("[APNEA] Not enough segments for sequence model")
        return

    log_module("apnea", "train", "done", f"data_ready=True", total_segs + len(slpdb_rows))
    logger.info("[APNEA] Dataset prepared: %d combined segments. XGBoost training runs via pipeline.py.",
                len(X_scaled))
    logger.info("[APNEA] Module complete.")
