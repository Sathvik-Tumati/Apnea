"""
evaluate_on_slpdb.py
====================
Evaluates the trained modality-aware apnea model against the
MIT-BIH Polysomnographic Database (SLPDB) on PhysioNet.

All signal processing and feature extraction is imported from pipeline.py
to ensure zero code duplication.  The model receives correct modality flags
(has_spo2=0, has_abp=0, has_resp_gt=0) so it knows it is operating in
ECG-only mode — the same path used during joint MIMIC+SLPDB training.

Usage
-----
  python evaluate_on_slpdb.py
  python evaluate_on_slpdb.py --records slp37 slp41 slp66
  python evaluate_on_slpdb.py --max-segments 60 --threshold 0.45
  python evaluate_on_slpdb.py --workers 6 --no-cache
  python evaluate_on_slpdb.py --model path/to/model.keras --scaler path/to/scaler.pkl
"""

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from concurrent import futures
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Import everything from pipeline.py ────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import (
        # Signal processing
        _bandpass,
        _detect_r_peaks,
        _compute_edr,
        # Feature extraction (ECG-only path)
        _extract_features_slpdb,
        # SLPDB data loading
        _load_slpdb_record,
        _get_slpdb_segment_labels,
        # Constants
        FS_ECG,
        FS_RESP,
        SEGMENT_LEN_S,
        TIMESTEPS,
        APNEA_FEATURE_COLS,
        SAMPLES_PER_SEG,
        SLPDB_RECORDS,
    )
    logger_prefix = "[pipeline.py imported OK]"
except ImportError as e:
    raise ImportError(
        f"Could not import from pipeline.py: {e}\n"
        "Ensure pipeline.py is in the same directory."
    ) from e

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

try:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    )
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(logger_prefix)


# ── Sequence builder (mirrors pipeline training) ──────────────────────────────

def _build_sequences(X: np.ndarray, timesteps: int) -> np.ndarray:
    return np.array([X[i: i + timesteps] for i in range(len(X) - timesteps)])


# ── Per-record processing ─────────────────────────────────────────────────────

def _process_record(
    record_name:  str,
    model,
    scaler,
    threshold:    float,
    max_segments: int,
) -> List[Dict]:
    ecg, annotation, fs_orig = _load_slpdb_record(record_name)
    if ecg is None:
        return []

    n_segs = min(len(ecg) // SAMPLES_PER_SEG, max_segments)
    if n_segs == 0:
        logger.warning("%s: not enough samples for one segment", record_name)
        return []

    gt_labels = _get_slpdb_segment_labels(annotation, n_segs, fs_orig)
    logger.info(
        "[%s] %d segments | GT apnea=%d normal=%d",
        record_name, n_segs, int(gt_labels.sum()), n_segs - int(gt_labels.sum()),
    )

    # Extract features for all segments
    feat_rows = []
    for i in range(n_segs):
        s   = i * SAMPLES_PER_SEG
        seg = ecg[s: s + SAMPLES_PER_SEG]
        f   = _extract_features_slpdb(seg)
        f["segment_idx"] = i
        f["gt_label"]    = int(gt_labels[i])
        feat_rows.append(f)

    feat_df = pd.DataFrame(feat_rows)

    # Ensure all model feature columns present
    for c in APNEA_FEATURE_COLS:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    # Scale and build sequences
    X_raw    = feat_df[APNEA_FEATURE_COLS].fillna(0.0).values.astype(float)
    X_scaled = scaler.transform(X_raw)
    X_seq    = _build_sequences(X_scaled, TIMESTEPS)

    prob_col = np.full(n_segs, np.nan)
    pred_col = np.full(n_segs, np.nan)

    if len(X_seq) > 0:
        y_prob = model.predict(X_seq, verbose=0, batch_size=64).flatten()
        for j, yp in enumerate(y_prob):
            si = j + TIMESTEPS
            if si < n_segs:
                prob_col[si] = float(yp)
                pred_col[si] = int(yp > threshold)
    else:
        logger.warning(
            "%s: only %d segments, need >%d for LSTM predictions",
            record_name, n_segs, TIMESTEPS,
        )

    results = []
    for i, row in feat_df.iterrows():
        results.append({
            "record":        record_name,
            "segment_idx":   int(row["segment_idx"]),
            "gt_label":      int(row["gt_label"]),
            "apnea_prob":    float(prob_col[i]) if not np.isnan(prob_col[i]) else np.nan,
            "apnea_pred":    int(pred_col[i])   if not np.isnan(pred_col[i]) else np.nan,
            "resp_rate_bpm": row.get("resp_rate_bpm",    0.0),
            "mean_hr":       row.get("mean_hr",          0.0),
            "rmssd":         row.get("rmssd",            0.0),
            "flatline_s":    row.get("flatline_duration_s", 0.0),
            "has_spo2":      row.get("has_spo2",         0),
            "has_abp":       row.get("has_abp",          0),
        })

    n_scored = int(np.sum(~np.isnan(prob_col)))
    logger.info("[%s] Scored: %d / %d segments", record_name, n_scored, n_segs)
    return results


# ── Evaluation metrics ────────────────────────────────────────────────────────

def _evaluate(df: pd.DataFrame, threshold: float, out_dir: str) -> Dict:
    scored = df[df["apnea_pred"].notna()].copy()
    scored["apnea_pred"] = scored["apnea_pred"].astype(int)

    y_true = scored["gt_label"].values
    y_pred = scored["apnea_pred"].values
    y_prob = scored["apnea_prob"].values

    n_total  = len(df)
    n_scored = len(scored)
    n_pos    = int(y_true.sum())
    n_neg    = n_scored - n_pos

    logger.info("\n" + "=" * 62)
    logger.info("  SLPDB EVALUATION RESULTS")
    logger.info("=" * 62)
    logger.info("  Records    : %d", df["record"].nunique())
    logger.info("  Segments   : %d total | %d scored | %d skipped",
                n_total, n_scored, n_total - n_scored)
    logger.info("  GT apnea   : %d | GT normal : %d", n_pos, n_neg)
    logger.info("  Threshold  : %.2f", threshold)

    if not HAS_SKL:
        logger.error("scikit-learn not installed — pip install scikit-learn")
        return {}

    acc  = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / max(tn + fp, 1)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    metrics = {
        "n_records":   df["record"].nunique(),
        "n_total":     n_total,
        "n_scored":    n_scored,
        "n_apnea_gt":  n_pos,
        "n_normal_gt": n_neg,
        "threshold":   threshold,
        "accuracy":    round(acc,  4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision":   round(prec, 4),
        "f1_score":    round(f1,   4),
        "auc_roc":     round(auc,  4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    # Per-record breakdown
    rec_rows = []
    for rec, grp in scored.groupby("record"):
        yt  = grp["gt_label"].values
        yp  = grp["apnea_pred"].values
        ypr = grp["apnea_prob"].values
        r_tp = int(((yp == 1) & (yt == 1)).sum())
        r_tn = int(((yp == 0) & (yt == 0)).sum())
        r_fp = int(((yp == 1) & (yt == 0)).sum())
        r_fn = int(((yp == 0) & (yt == 1)).sum())
        try:
            r_auc = roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else float("nan")
        except Exception:
            r_auc = float("nan")
        dur_h = len(grp) * SEGMENT_LEN_S / 3600.0
        ahi   = r_tp / max(dur_h, 1e-6)
        rec_rows.append({
            "record":      rec,
            "n_segments":  len(grp),
            "gt_apnea":    int(yt.sum()),
            "pred_apnea":  int(yp.sum()),
            "tp": r_tp, "tn": r_tn, "fp": r_fp, "fn": r_fn,
            "sensitivity": round(r_tp / max(r_tp + r_fn, 1), 3),
            "specificity": round(r_tn / max(r_tn + r_fp, 1), 3),
            "auc_roc":     round(r_auc, 3) if not np.isnan(r_auc) else None,
            "ahi_proxy":   round(ahi, 1),
        })

    rec_df = pd.DataFrame(rec_rows).sort_values("record")
    metrics["per_record"] = rec_df.to_dict("records")

    logger.info("\n  ── Segment-level Classification ──────────────────────")
    logger.info("  Accuracy    : %.1f%%", acc  * 100)
    logger.info("  Sensitivity : %.1f%%", sens * 100)
    logger.info("  Specificity : %.1f%%", spec * 100)
    logger.info("  Precision   : %.1f%%", prec * 100)
    logger.info("  F1 score    : %.3f",   f1)
    logger.info("  AUC-ROC     : %.3f",   auc)
    logger.info("\n  ── Confusion Matrix ──────────────────────────────────")
    logger.info("                  Pred Normal   Pred Apnea")
    logger.info("  GT Normal   :     %6d        %6d", tn, fp)
    logger.info("  GT Apnea    :     %6d        %6d", fn, tp)
    logger.info("\n  ── Per-Record Breakdown ──────────────────────────────")
    logger.info("  %-10s  %5s  %6s  %6s  %5s  %5s  %5s",
                "Record", "Segs", "GTPos", "PrPos", "Sens", "Spec", "AHI")
    for r in rec_rows:
        logger.info("  %-10s  %5d  %6d  %6d  %4.0f%%  %4.0f%%  %5.1f",
                    r["record"], r["n_segments"], r["gt_apnea"], r["pred_apnea"],
                    r["sensitivity"] * 100, r["specificity"] * 100, r["ahi_proxy"])

    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_roc(y_true, y_prob, out_dir):
    if not HAS_MPL: return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2171b5", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — SLPDB Apnea Detection"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150); plt.close(fig)
    logger.info("[PLOT] ROC curve saved")


def _plot_prob_distribution(df, out_dir):
    if not HAS_MPL: return
    sc  = df[df["apnea_prob"].notna()]
    pos = sc[sc["gt_label"] == 1]["apnea_prob"]
    neg = sc[sc["gt_label"] == 0]["apnea_prob"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(neg, bins=30, alpha=0.6, color="#2171b5", label=f"Normal (n={len(neg)})", density=True)
    ax.hist(pos, bins=30, alpha=0.6, color="#d7301f", label=f"Apnea  (n={len(pos)})", density=True)
    ax.set_xlabel("Apnea Probability"); ax.set_ylabel("Density")
    ax.set_title("Model Output Distribution — GT Normal vs Apnea")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prob_distribution.png"), dpi=150); plt.close(fig)
    logger.info("[PLOT] Probability distribution saved")


def _plot_confusion_matrix(tp, tn, fp, fn, out_dir):
    if not HAS_MPL: return
    cm  = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Normal", "Pred Apnea"])
    ax.set_yticklabels(["GT Normal",   "GT Apnea"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix — SLPDB"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)
    logger.info("[PLOT] Confusion matrix saved")


def _plot_per_record(rec_df, out_dir):
    if not HAS_MPL or rec_df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(rec_df)); w = 0.35
    axes[0].bar([i - w/2 for i in x], rec_df["sensitivity"]*100, width=w,
                color="#2171b5", label="Sensitivity")
    axes[0].bar([i + w/2 for i in x], rec_df["specificity"]*100, width=w,
                color="#41ab5d", label="Specificity")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(rec_df["record"].tolist(), rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("%"); axes[0].set_title("Per-Record Sensitivity / Specificity")
    axes[0].legend(); axes[0].axhline(80, color="#d7301f", lw=1, linestyle="--")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, rec_df["ahi_proxy"], color="#fd8d3c", edgecolor="white")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(rec_df["record"].tolist(), rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("AHI proxy (events/hr)"); axes[1].set_title("Per-Record AHI Proxy")
    for thr, col, lbl in [(5,"#41ab5d","Normal"),(15,"#fd8d3c","Mild"),(30,"#d7301f","Moderate")]:
        axes[1].axhline(thr, color=col, lw=1.2, linestyle="--", label=lbl)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_record_metrics.png"), dpi=150); plt.close(fig)
    logger.info("[PLOT] Per-record metrics saved")


def _plot_timeline(df, record_name, out_dir):
    if not HAS_MPL: return
    rec = df[df["record"] == record_name].sort_values("segment_idx")
    if rec.empty or rec["apnea_prob"].isna().all(): return
    t   = rec["segment_idx"].values * SEGMENT_LEN_S / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].fill_between(t, rec["gt_label"].values, step="post", alpha=0.7,
                         color="#d7301f", label="GT Apnea")
    axes[0].set_ylabel("GT Label"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].plot(t, rec["apnea_prob"].values, color="#2171b5", lw=1.2, label="Apnea prob")
    axes[1].fill_between(t, rec["apnea_pred"].fillna(0).values, alpha=0.3,
                         color="#fd8d3c", step="post", label="Prediction")
    axes[1].set_ylabel("Probability"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[2].plot(t, rec["resp_rate_bpm"].values, color="#41ab5d", lw=1.2, label="Resp rate (bpm)")
    axes[2].set_ylabel("Resp Rate bpm"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Time (min)")
    fig.suptitle(f"SLPDB {record_name} — GT vs Predicted Apnea", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"timeline_{record_name}.png"), dpi=150)
    plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(metrics: Dict, out_dir: str) -> None:
    lines = [
        "=" * 62,
        "  SLPDB APNEA MODEL EVALUATION REPORT",
        "  (XGBoost — ECG-only inference path)",
        "=" * 62,
        "",
        f"  Records evaluated  : {metrics['n_records']}",
        f"  Total segments     : {metrics['n_total']}",
        f"  Scored segments    : {metrics['n_scored']}  "
        f"({metrics['n_total'] - metrics['n_scored']} skipped — LSTM warmup)",
        f"  GT apnea segments  : {metrics['n_apnea_gt']}",
        f"  GT normal segments : {metrics['n_normal_gt']}",
        f"  Decision threshold : {metrics['threshold']}",
        "",
        "  ── Segment-level Classification ─────────────────────────",
        f"  Accuracy    : {metrics['accuracy']*100:.1f}%",
        f"  Sensitivity : {metrics['sensitivity']*100:.1f}%  (apnea recall)",
        f"  Specificity : {metrics['specificity']*100:.1f}%  (normal recall)",
        f"  Precision   : {metrics['precision']*100:.1f}%",
        f"  F1 Score    : {metrics['f1_score']:.3f}",
        f"  AUC-ROC     : {metrics['auc_roc']:.3f}",
        "",
        "  ── Confusion Matrix ──────────────────────────────────────",
        f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}",
        "",
        "  ── Per-Record Breakdown ──────────────────────────────────",
        f"  {'Record':<10}  {'Segs':>5}  {'GTPos':>6}  {'PrPos':>6}  "
        f"{'Sens':>5}  {'Spec':>5}  {'AHI':>5}",
    ]
    for r in metrics.get("per_record", []):
        lines.append(
            f"  {r['record']:<10}  {r['n_segments']:>5}  {r['gt_apnea']:>6}  "
            f"{r['pred_apnea']:>6}  {r['sensitivity']*100:>4.0f}%  "
            f"{r['specificity']*100:>4.0f}%  {r['ahi_proxy']:>5.1f}"
        )
    lines += [
        "",
        "  ── Interpretation ───────────────────────────────────────",
        "  Sensitivity ≥ 80% : Good apnea recall",
        "  Specificity ≥ 80% : Low false alarm rate",
        "  AUC ≥ 0.80        : Clinically useful discrimination",
        "  NOTE: Research prototype. Not for clinical use.",
        "=" * 62,
    ]
    report = "\n".join(lines)
    logger.info("\n%s", report)
    path = os.path.join(out_dir, "slpdb_evaluation_report.txt")
    with open(path, "w") as f:
        f.write(report + "\n")
    logger.info("[REPORT] Saved → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(
    model_path:   str        = "apnea_model_xgb_seq.pkl",

    scaler_path:  str        = "apnea_scaler.pkl",
    out_dir:      str        = "slpdb_eval_output",
    records:      List[str]  = None,
    max_segments: int        = 80,
    threshold:    float      = 0.45,
    workers:      int        = 6,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if not HAS_WFDB:
        logger.error("wfdb not installed — pip install wfdb"); return
    if not HAS_SKL:
        logger.error("scikit-learn not installed"); return

    for path, label in [(model_path, "Model"), (scaler_path, "Scaler")]:
        if not os.path.exists(path):
            logger.error("%s not found at '%s'", label, path); return

    logger.info("Loading model from %s ...", model_path)
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model input shape: %s", model.input_shape)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc); return

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded.  n_features=%d", scaler.n_features_in_)

    # Validate feature count matches
    if scaler.n_features_in_ != len(APNEA_FEATURE_COLS):
        logger.error(
            "Feature count mismatch: scaler expects %d, pipeline has %d.\n"
            "Re-run pipeline.py --save-model to regenerate.",
            scaler.n_features_in_, len(APNEA_FEATURE_COLS),
        )
        return

    if records is None:
        records = SLPDB_RECORDS

    logger.info(
        "Evaluating %d records | workers=%d | max_segs=%d | threshold=%.2f",
        len(records), workers, max_segments, threshold,
    )

    all_results = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {
            pool.submit(_process_record, rec, model, scaler, threshold, max_segments): rec
            for rec in records
        }
        for fut in futures.as_completed(fut_map):
            rec = fut_map[fut]
            try:
                res = fut.result()
                all_results.extend(res)
                logger.info("[DONE] %s: %d segments", rec, len(res))
            except Exception as exc:
                logger.error("[FAIL] %s: %s", rec, exc)

    if not all_results:
        logger.error("No segments processed — check record names and connectivity")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "slpdb_segment_results.csv"), index=False)
    logger.info("Results saved → %s/slpdb_segment_results.csv", out_dir)

    # Print probability distribution diagnostic
    logger.info("\n── Probability distribution diagnostic ──────────────")
    logger.info("\n%s", df["apnea_prob"].describe().to_string())
    if "gt_label" in df.columns:
        logger.info("\nBy GT label:\n%s",
                    df.groupby("gt_label")["apnea_prob"].describe().to_string())

    metrics = _evaluate(df, threshold, out_dir)
    if not metrics:
        return

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "per_record"}, f, indent=2)

    rec_df = pd.DataFrame(metrics.get("per_record", []))
    rec_df.to_csv(os.path.join(out_dir, "per_record_metrics.csv"), index=False)

    if HAS_MPL:
        scored = df[df["apnea_prob"].notna()]
        y_true = scored["gt_label"].values
        y_prob = scored["apnea_prob"].values
        if len(np.unique(y_true)) > 1:
            _plot_roc(y_true, y_prob, out_dir)
        _plot_prob_distribution(df, out_dir)
        _plot_confusion_matrix(metrics["tp"], metrics["tn"],
                               metrics["fp"], metrics["fn"], out_dir)
        if not rec_df.empty:
            _plot_per_record(rec_df, out_dir)
        for rec in df["record"].unique():
            _plot_timeline(df, rec, out_dir)

    _write_report(metrics, out_dir)
    logger.info("\n[EVAL] Done! Outputs → %s/", out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate modality-aware apnea model on SLPDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python evaluate_on_slpdb.py
  python evaluate_on_slpdb.py --records slp37 slp41 slp66 slp48 slp67x
  python evaluate_on_slpdb.py --max-segments 60 --threshold 0.45
  python evaluate_on_slpdb.py --workers 8
  python evaluate_on_slpdb.py --model custom_model.keras --scaler custom_scaler.pkl
        """
    )
    p.add_argument("--model",        default="apnea_model_xgb_seq.pkl")

    p.add_argument("--scaler",       default="apnea_scaler.pkl")
    p.add_argument("--out-dir",      default="slpdb_eval_output")
    p.add_argument("--records",      nargs="+", default=None)
    p.add_argument("--max-segments", type=int,   default=80)
    p.add_argument("--threshold",    type=float, default=0.45)
    p.add_argument("--workers",      type=int,   default=6)
    return p.parse_args()


def main():
    args = _parse_args()
    run_evaluation(
        model_path   = args.model,
        scaler_path  = args.scaler,
        out_dir      = args.out_dir,
        records      = args.records,
        max_segments = args.max_segments,
        threshold    = args.threshold,
        workers      = args.workers,
    )


if __name__ == "__main__":
    main()