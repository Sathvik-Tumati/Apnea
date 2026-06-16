"""
pipeline/modules/evaluate.py
============================
Standalone evaluation helpers: threshold sweeping, per-source metrics,
bootstrap AUC CI.  Called by train.run_apnea_module() after training.
"""

import json
import logging
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, classification_report,
)
from pipeline.db.database import insert_apnea_results
from pipeline.modules.config import NumpyEncoder

logger = logging.getLogger(__name__)


def sweep_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                    lo: float = 0.25, hi: float = 0.75, step: float = 0.01):
    """Return (best_threshold, best_f1) over [lo, hi]."""
    thresholds = np.arange(lo, hi, step)
    f1s = [f1_score(y_true, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     n: int = 1000, seed: int = 42):
    """Return (lo, hi) 95% CI for AUC via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    boots = [
        roc_auc_score(y_true[idx], y_prob[idx])
        for _ in range(n)
        for idx in [rng.randint(0, len(y_true), len(y_true))]
        if len(np.unique(y_true[idx])) > 1
    ]
    return tuple(np.percentile(boots, [2.5, 97.5]).tolist()) if boots else (0.0, 0.0)


def evaluate_model(
    model,
    X_te: np.ndarray,
    y_te: np.ndarray,
    src_te: np.ndarray,
    total_segs: int,
) -> dict:
    """
    Run full evaluation suite on the test set:
      - Global + per-source threshold sweep
      - Bootstrap AUC CI
      - Classification report
      - DB persistence
    Returns a dict with all metrics.
    """
    if len(np.unique(y_te)) < 2:
        logger.warning("[EVAL] Only one class in test set — AUC not computed")
        return {}

    y_prob = model.predict(X_te, verbose=0).flatten()

    # Global threshold
    best_thresh, best_f1 = sweep_threshold(y_te, y_prob)
    auc = roc_auc_score(y_te, y_prob)
    ci_lo, ci_hi = bootstrap_auc_ci(y_te, y_prob)

    logger.info(
        "[EVAL] Overall AUC=%.4f (95%% CI %.3f–%.3f)  threshold=%.2f  F1=%.3f",
        auc, ci_lo, ci_hi, best_thresh, best_f1,
    )

    # Per-source thresholds
    source_names  = {0: "MIMIC", 1: "SLPDB"}
    src_thresholds: dict = {}
    for src_id, src_name in source_names.items():
        mask = src_te == src_id
        if mask.sum() < 5 or len(np.unique(y_te[mask])) < 2:
            src_thresholds[src_name] = best_thresh
            continue
        t, f1 = sweep_threshold(y_te[mask], y_prob[mask])
        src_thresholds[src_name] = t
        src_auc  = roc_auc_score(y_te[mask], y_prob[mask])
        src_sens = recall_score(y_te[mask], (y_prob[mask] > t).astype(int), zero_division=0)
        src_prec = precision_score(y_te[mask], (y_prob[mask] > t).astype(int), zero_division=0)
        logger.info(
            "[EVAL] %-6s AUC=%.3f  F1=%.3f  Sens=%.1f%%  Prec=%.1f%%  n=%d  (thresh=%.2f)",
            src_name, src_auc, f1, src_sens * 100, src_prec * 100, int(mask.sum()), t,
        )

    report = classification_report(
        y_te, (y_prob > best_thresh).astype(int),
        target_names=["Normal", "Apnea"], output_dict=True,
    )
    logger.info(
        "\n%s",
        classification_report(y_te, (y_prob > best_thresh).astype(int),
                               target_names=["Normal", "Apnea"]),
    )

    insert_apnea_results(auc, report)

    return {
        "auc":          auc,
        "ci_lo":        ci_lo,
        "ci_hi":        ci_hi,
        "best_thresh":  best_thresh,
        "best_f1":      best_f1,
        "src_thresholds": src_thresholds,
        "report":       report,
    }
