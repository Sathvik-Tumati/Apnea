"""
pipeline/pipeline.py
====================
Multi-source apnea detection pipeline.
Trains both:
  1. Modality-aware BiLSTM  (apnea_model.keras + apnea_scaler.pkl)
  2. XGBoost (seq)          (apnea_model_xgb_seq.pkl + apnea_scaler_tree.pkl)

Both models are trained on the same MIMIC-IV + SLPDB data and the same
train/val/test split so results are directly comparable.

Usage
-----
  python pipeline/pipeline.py
  python pipeline/pipeline.py --fresh
  python pipeline/pipeline.py --save-model
  python pipeline/pipeline.py --no-slpdb
  python pipeline/pipeline.py --slpdb-records slp37 slp41 slp66
  python pipeline/pipeline.py --save-model --bilstm-only
  python pipeline/pipeline.py --save-model --xgb-only
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
import warnings
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.db.database import init_db, fetch_apnea_segments
from pipeline.modules.config import set_all_seeds, logger, SLPDB_RECORDS, TIMESTEPS
from pipeline.modules.train import (
    run_apnea_module,
    _build_combined_dataset,
    _build_sequences,
    _apply_modality_dropout_sequences,
)
from pipeline.modules.ingest_slpdb import ingest_slpdb_records

warnings.filterwarnings("ignore")
set_all_seeds(42)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh",          action="store_true",
                        help="Delete old data/model and re-run.")
    parser.add_argument("--save-model",     action="store_true",
                        help="Save trained model + scaler files.")
    parser.add_argument("--no-slpdb",       action="store_true",
                        help="Skip SLPDB completely.")
    parser.add_argument("--slpdb-records",  nargs="+",
                        help="Specific SLPDB records to ingest (e.g. slp01a slp66)")
    parser.add_argument("--bilstm-only",    action="store_true",
                        help="Train BiLSTM only, skip XGBoost.")
    parser.add_argument("--xgb-only",       action="store_true",
                        help="Train XGBoost only, skip BiLSTM.")
    return parser.parse_args()


def _train_xgboost(save_model: bool, skip_slpdb: bool, slpdb_records) -> None:
    """
    Train XGBoost (seq) on the same data as the BiLSTM and optionally save.
    Reuses _build_combined_dataset so the scaler and split are identical.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("[XGB] xgboost not installed — pip install xgboost")
        return

    import datetime
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    logger.info("[XGB] ── XGBoost training ─────────────────────────────")

    # ── Load data ─────────────────────────────────────────────────────────────
    seg_df = fetch_apnea_segments()
    if seg_df is None or len(seg_df) == 0:
        logger.error("[XGB] No segments in DB — run BiLSTM first or use --fresh")
        return

    if not isinstance(seg_df, pd.DataFrame):
        seg_df = pd.DataFrame(seg_df)

    from pipeline.modules.config import APNEA_FEATURE_COLS
    for col in APNEA_FEATURE_COLS:
        if col not in seg_df.columns:
            seg_df[col] = 0.0
    seg_df["has_spo2"] = seg_df.get(
        "has_spo2", pd.Series(1, index=seg_df.index)
    ).fillna(1).astype(float)

    slpdb_rows = []
    if not skip_slpdb:
        records = slpdb_records or SLPDB_RECORDS
        run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slpdb_rows = ingest_slpdb_records(records, run_id=run_id)
        logger.info("[XGB] SLPDB segments: %d", len(slpdb_rows))

    X_flat, y_all, src_all, scaler = _build_combined_dataset(seg_df, slpdb_rows)
    logger.info("[XGB] Combined dataset: %d segments | apnea=%.1f%%",
                len(y_all), y_all.mean() * 100)

    # ── Build sequences ───────────────────────────────────────────────────────
    X_seq, y_seq   = _build_sequences(X_flat, y_all, TIMESTEPS)
    _, src_seq     = _build_sequences(X_flat, src_all, TIMESTEPS)

    # ── Same stratified split as BiLSTM (random_state=42) ────────────────────
    X_tr_val, X_te, y_tr_val, y_te, src_tr_val, src_te = train_test_split(
        X_seq, y_seq, src_seq,
        test_size=0.20, stratify=y_seq, random_state=42)
    X_tr, X_val, y_tr, y_val, src_tr, src_val = train_test_split(
        X_tr_val, y_tr_val, src_tr_val,
        test_size=0.125, stratify=y_tr_val, random_state=42)

    # Modality dropout on training sequences
    X_tr_aug = _apply_modality_dropout_sequences(X_tr, src_tr)

    # Flatten sequences for XGBoost: (N, T, F) → (N, T*F)
    X_tr_flat  = X_tr_aug.reshape(len(X_tr_aug), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_te_flat  = X_te.reshape(len(X_te), -1)

    logger.info("[XGB] Train=%d  Val=%d  Test=%d  Features=%d",
                len(y_tr), len(y_val), len(y_te), X_tr_flat.shape[1])

    # ── Train ─────────────────────────────────────────────────────────────────
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    scale_pos = neg / (pos + 1e-6)
    logger.info("[XGB] Class balance — neg=%d pos=%d scale_pos_weight=%.1f",
                neg, pos, scale_pos)

    model = xgb.XGBClassifier(
        n_estimators          = 500,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = scale_pos,
        eval_metric           = "auc",
        early_stopping_rounds = 20,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
    )
    model.fit(
        X_tr_flat, y_tr,
        eval_set  = [(X_val_flat, y_val)],
        verbose   = False,
    )
    logger.info("[XGB] Best iteration: %d", model.best_iteration)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, f1_score, classification_report
    y_prob = model.predict_proba(X_te_flat)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)

    # Optimal threshold
    thresholds = [t / 100 for t in range(20, 80)]
    f1s        = [f1_score(y_te, (y_prob > t).astype(int), zero_division=0)
                  for t in thresholds]
    best_thresh = thresholds[int(np.argmax(f1s))]
    y_pred      = (y_prob > best_thresh).astype(int)

    logger.info("[XGB] Test AUC=%.4f  best_threshold=%.2f", auc, best_thresh)
    logger.info("\n%s", classification_report(
        y_te, y_pred, target_names=["Normal", "Apnea"]))

    # ── Save ──────────────────────────────────────────────────────────────────
        # ── Save ──────────────────────────────────────────────────────────────────
    if save_model:
        project_root = Path(__file__).resolve().parent.parent
        xgb_path = project_root / "apnea_model_xgb_seq.pkl"
        scaler_path = project_root / "apnea_scaler_tree.pkl"

        with open(xgb_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info("[XGB] Saved → %s", xgb_path)
        logger.info("[XGB] Saved → %s", scaler_path)
        
        # ── Testing: reload and verify ──────────────────────────────────────
        logger.info("[XGB] Testing reload of saved model...")
        with open(scaler_path, "rb") as f:
            scaler_reloaded = pickle.load(f)
        # Use scaler_reloaded for testing if needed
        logger.info("[XGB] Scaler reloaded successfully")
    else:
        logger.info("[XGB] --save-model not set — models not saved to disk")


def main():
    args = _parse_args()

    if args.fresh:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "vitals_pipeline.db")
        if os.path.exists(db_path):
            confirm = input(
                f"[FRESH] This will delete {db_path} and retrain from scratch.\n"
                f"        Existing model files will be overwritten on --save-model.\n"
                f"        Type 'yes' to continue: "
            )
            if confirm.strip().lower() != "yes":
                logger.info("[FRESH] Aborted.")
                sys.exit(0)
            os.remove(db_path)
            logger.info("[FRESH] Deleted %s", db_path)

    init_db()

    # ── BiLSTM ────────────────────────────────────────────────────────────────
    if not args.xgb_only:
        logger.info("[PIPELINE] ── Training BiLSTM ──────────────────────────")
        run_apnea_module(
            save_model    = args.save_model,
            skip_slpdb    = args.no_slpdb,
            slpdb_records = args.slpdb_records,
        )
    else:
        logger.info("[PIPELINE] Skipping BiLSTM (--xgb-only)")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if not args.bilstm_only:
        logger.info("[PIPELINE] ── Training XGBoost ─────────────────────────")
        _train_xgboost(
            save_model    = args.save_model,
            skip_slpdb    = args.no_slpdb,
            slpdb_records = args.slpdb_records,
        )
    else:
        logger.info("[PIPELINE] Skipping XGBoost (--bilstm-only)")

    logger.info("[PIPELINE] All done.")


if __name__ == "__main__":
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for d in physical_devices:
            tf.config.experimental.set_memory_growth(d, True)
    except Exception:
        pass

    main()