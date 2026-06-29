from typing import Any, Dict, List, Optional, Tuple
import datetime
import json
import logging
import pickle
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from pipeline.db.database import fetch_apnea_segments, insert_apnea_results, log_module
from pipeline.modules.config import *
from pipeline.modules.config import (
    _SPO2_IDXS, _HAS_SPO2_IDX, _ABP_IDXS, _CROSS_IDXS, _HAS_ABP_IDX
)
from pipeline.modules.model import _build_model, _focal_loss
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

    # ── STAGE 3: Train modality-aware BiLSTM ─────────────────────────────────
    log_module("apnea", "train", "started")
    if not HAS_TF:
        logger.error("[APNEA] TensorFlow not installed")
        log_module("apnea", "train", "failed", "tensorflow not installed", 0)
        return

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

    # Build sequences on combined array
    X_seq, y_seq = _build_sequences(X_scaled, y_all, TIMESTEPS)
    _, src_seq = _build_sequences(X_scaled, src_all, TIMESTEPS)

    # Original approach: split sequences, stratify by label
    X_tr_val, X_te, y_tr_val, y_te, src_tr_val, src_te = train_test_split(
        X_seq, y_seq, src_seq,
        test_size=0.20, stratify=y_seq, random_state=42,
    )
    X_tr, X_val, y_tr, y_val, src_tr, src_val = train_test_split(
        X_tr_val, y_tr_val, src_tr_val,
        test_size=0.125,  # Original working value
        stratify=y_tr_val,
        random_state=42,
    )

    n_mimic_tr  = int((src_tr == 0).sum())
    n_slpdb_tr  = int((src_tr == 1).sum())
    n_mimic_val = int((src_val == 0).sum())
    n_slpdb_val = int((src_val == 1).sum())
    n_mimic_te  = int((src_te == 0).sum())
    n_slpdb_te  = int((src_te == 1).sum())

    logger.info(
        "[TRAIN] Train: %d (MIMIC=%d, SLPDB=%d) | "
        "Val: %d (MIMIC=%d, SLPDB=%d) | "
        "Test: %d (MIMIC=%d, SLPDB=%d)",
        len(y_tr), n_mimic_tr, n_slpdb_tr,
        len(y_val), n_mimic_val, n_slpdb_val,
        len(y_te), n_mimic_te, n_slpdb_te,
    )

    # Calculate class weights
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    class_weight = {0: 1.0, 1: neg / (pos + 1e-6)}
    logger.info("[TRAIN] Class weights: normal=%.3f, apnea=%.3f", 
                class_weight[0], class_weight[1])

    # Apply modality dropout
    X_tr_aug = _apply_modality_dropout_sequences(
        X_tr, src_tr,
        drop_spo2_prob=0.30,
        drop_abp_prob=0.30,
    )

    # Build and compile model
    model = _build_model(n_features=len(APNEA_FEATURE_COLS), timesteps=TIMESTEPS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=_focal_loss(gamma=2.0, alpha=0.75),
        metrics=["AUC"],
    )
    model.summary(print_fn=logger.info)

    # Callbacks with ModelCheckpoint
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "apnea_best.keras",
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=6,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=3,
            mode="max",
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        X_tr_aug, y_tr,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks,
    )

    # Evaluation
    y_prob = model.predict(X_te, verbose=0).flatten()

    if len(np.unique(y_te)) > 1:
        thresholds = np.arange(0.25, 0.75, 0.01)
        
        # Global threshold
        f1s = [f1_score(y_te, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
        best_thresh = float(thresholds[np.argmax(f1s)])
        auc = roc_auc_score(y_te, y_prob)
        
        # Domain-specific thresholds
        mimic_mask = src_te == 0
        mimic_thresh = best_thresh
        slpdb_thresh = best_thresh
        
        if mimic_mask.sum() >= 5 and len(np.unique(y_te[mimic_mask])) > 1:
            mimic_f1s = [f1_score(y_te[mimic_mask], (y_prob[mimic_mask] > t).astype(int),
                                  zero_division=0) for t in thresholds]
            mimic_thresh = float(thresholds[np.argmax(mimic_f1s)])
        
        slpdb_mask = src_te == 1
        if slpdb_mask.sum() >= 5 and len(np.unique(y_te[slpdb_mask])) > 1:
            slpdb_f1s = [f1_score(y_te[slpdb_mask], (y_prob[slpdb_mask] > t).astype(int),
                                  zero_division=0) for t in thresholds]
            slpdb_thresh = float(thresholds[np.argmax(slpdb_f1s)])
        
        logger.info("[EVAL] Domain thresholds: MIMIC=%.2f  SLPDB=%.2f  Global=%.2f", 
                    mimic_thresh, slpdb_thresh, best_thresh)
        
        # Compute validation AUC
        y_val_prob = model.predict(X_val, verbose=0).flatten()
        val_auc = roc_auc_score(y_val, y_val_prob) if len(np.unique(y_val)) > 1 else 0.0
        logger.info("[EVAL] Validation AUC=%.4f", val_auc)

        # Bootstrap CI for global AUC
        from sklearn.utils import resample as sk_resample
        rng = np.random.RandomState(42)
        boots = [
            roc_auc_score(y_te[idx], y_prob[idx])
            for _ in range(1000)
            for idx in [rng.randint(0, len(y_te), len(y_te))]
            if len(np.unique(y_te[idx])) > 1
        ]
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

        logger.info(
            "[EVAL] Overall AUC=%.4f (95%% CI %.3f–%.3f) threshold=%.2f F1=%.3f",
            auc, ci_lo, ci_hi, best_thresh, max(f1s),
        )

        # Per-source evaluation with domain-specific thresholds
        for src_val, src_name, src_thresh in [(0, "MIMIC", mimic_thresh), (1, "SLPDB", slpdb_thresh)]:
            mask = src_te == src_val
            if mask.sum() >= 5 and len(np.unique(y_te[mask])) > 1:
                src_auc = roc_auc_score(y_te[mask], y_prob[mask])
                src_f1 = f1_score(y_te[mask], (y_prob[mask] > src_thresh).astype(int), zero_division=0)
                src_sens = recall_score(y_te[mask], (y_prob[mask] > src_thresh).astype(int), zero_division=0)
                src_prec = precision_score(y_te[mask], (y_prob[mask] > src_thresh).astype(int), zero_division=0)
                logger.info(
                    "[EVAL] %-6s AUC=%.3f F1=%.3f Sensitivity=%.1f%% Precision=%.1f%% n=%d (thresh=%.2f)",
                    src_name, src_auc, src_f1, src_sens * 100, src_prec * 100, int(mask.sum()), src_thresh,
                )

        report = classification_report(
            y_te, (y_prob > best_thresh).astype(int),
            target_names=["Normal", "Apnea"], output_dict=True,
        )
        logger.info("\n%s", classification_report(
            y_te, (y_prob > best_thresh).astype(int),
            target_names=["Normal", "Apnea"],
        ))

        insert_apnea_results(auc, report)
        log_module("apnea", "train", "done", f"auc={auc:.4f}", total_segs + len(slpdb_rows))
        
        # Save thresholds alongside model
        if save_model and HAS_TF:
            thresholds_dict = {
                "global": best_thresh,
                "mimic": mimic_thresh,
                "slpdb": slpdb_thresh
            }
            with open("apnea_thresholds.json", "w") as f:
                json.dump(thresholds_dict, f, indent=2, cls=_NumpyEncoder)
            logger.info("[SAVE] Thresholds → apnea_thresholds.json")
            
    else:
        logger.warning("[APNEA] Only one class in test set — AUC not computed")
        log_module("apnea", "train", "skipped", "Single class in test", 0)

    # Save model
    if save_model and HAS_TF:
        model.save("apnea_model.keras")
        with open("apnea_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("apnea_feature_cols.json", "w") as f:
            json.dump(APNEA_FEATURE_COLS, f, indent=2, cls=_NumpyEncoder)
        logger.info("[SAVE] Model → apnea_model.keras")
        logger.info("[SAVE] Scaler → apnea_scaler.pkl")
        logger.info("[SAVE] Features → apnea_feature_cols.json")

    logger.info("[APNEA] Module complete.")


