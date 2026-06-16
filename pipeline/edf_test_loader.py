"""
edf_test_loader.py
==================
Load converted EDF data (CSV or JSON from edf_to_pipeline.py) and run
them through the pipeline's feature extraction + BiLSTM inference.

This is a DROP-IN TEST HARNESS — it does NOT need MIMIC or SLPDB access.
Point it at your converted files and it will:
  1. Load segments from CSV or JSON
  2. Run _extract_features_slpdb() on each segment (ECG-only path)
     OR _extract_apnea_features() if pleth/resp/abp are present
  3. Build sequences and run inference if a saved model exists
  4. Print per-segment predictions vs true labels

Usage
-----
  # Run feature extraction only (no model needed)
  python edf_test_loader.py --data ./converted/ --mode features

  # Run inference with a saved model
  python edf_test_loader.py --data ./converted/ --mode infer \\
      --model apnea_model.keras \\
      --scaler apnea_scaler.pkl \\
      --features apnea_feature_cols.json

  # Load from JSON instead of CSV
  python edf_test_loader.py --data ./converted/recording.json --mode features
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Add pipeline root to path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pipeline imports (adjust path if needed)
try:
    from pipeline.modules.features import (
        _extract_features_slpdb,
        _extract_apnea_features,
        _bandpass, _detect_r_peaks, _compute_edr,
    )
    from pipeline.modules.config import (
        APNEA_FEATURE_COLS,
        FS_ECG, FS_PPG, FS_RESP, SEGMENT_LEN_S as SEG_LEN
    )
    from pipeline.modules.model import GatherFlags
    HAS_PIPELINE = True
except ImportError:
    try:
        # Try importing from current directory
        from pipeline.modules.features import (
            _extract_features_slpdb,
            _extract_apnea_features,
            _bandpass, _detect_r_peaks, _compute_edr,
        )
        from pipeline.modules.config import (
            APNEA_FEATURE_COLS,
            FS_ECG, FS_PPG, FS_RESP, SEGMENT_LEN_S as SEG_LEN
        )
        from pipeline.modules.model import GatherFlags
        HAS_PIPELINE = True
    except ImportError:
        print("WARNING: Could not import pipeline. Feature extraction will be skipped.")
        print("  Make sure pipeline/pipeline.py is accessible from this script's directory.")
        HAS_PIPELINE = False
        FS_ECG = 125
        FS_PPG = 120
        FS_RESP = 4
        SEG_LEN = 30
        APNEA_FEATURE_COLS = []


# ══════════════════════════════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    segments = data.get("segments", [])
    meta = data.get("meta", {})
    print(f"  JSON: {os.path.basename(path)} | {len(segments)} segments | "
          f"source={meta.get('source_file', 'unknown')}")
    return segments


def load_csv_dir(directory: str) -> List[Dict]:
    """
    Load segments from the CSV files written by edf_to_pipeline.py.
    Expects: <stem>_ecg.csv, <stem>_pleth.csv, <stem>_resp.csv, <stem>_abp.csv
    """
    directory = Path(directory)
    stems = set()
    # Support both *_ecg.csv (full) and *_ecg_sleep.csv (sleep-filtered)
    for suffix in ("*_ecg_sleep.csv", "*_ecg.csv"):
        for f in directory.glob(suffix):
            ecg_suffix = "_ecg_sleep" if suffix == "*_ecg_sleep.csv" else "_ecg"
            stems.add((f.stem.replace(ecg_suffix, ""), "_sleep" if "_sleep" in suffix else ""))
        if stems:
            break  # prefer sleep-filtered if both exist

    if not stems:
        print(f"  No *_ecg.csv or *_ecg_sleep.csv files found in {directory}")
        return []

    all_segments = []
    for stem, variant in sorted(stems):
        print(f"  Loading stem: {stem}{variant}")
        ecg_df   = pd.read_csv(directory / f"{stem}_ecg{variant}.csv")
        pleth_df = _try_load(directory / f"{stem}_pleth{variant}.csv")
        resp_df  = _try_load(directory / f"{stem}_resp{variant}.csv")
        abp_df   = _try_load(directory / f"{stem}_abp{variant}.csv")

        signal_cols = [c for c in ecg_df.columns if c.startswith("t")]

        for _, row in ecg_df.iterrows():
            idx   = int(row["segment_idx"])
            label = int(row["true_label"])
            ecg   = row[signal_cols].values.astype(np.float32)

            seg = {
                "segment_idx": idx,
                "true_label":  label,
                "ecg":         ecg.tolist(),
                "pleth":       _get_signal(pleth_df, idx),
                "resp":        _get_signal(resp_df,  idx),
                "abp":         _get_signal(abp_df,   idx),
                "has_pleth":   pleth_df is not None,
                "has_resp":    resp_df  is not None,
                "has_abp":     abp_df   is not None,
            }
            all_segments.append(seg)

    print(f"  Total segments loaded: {len(all_segments)}")
    return all_segments


def _try_load(path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _get_signal(df: Optional[pd.DataFrame], idx: int) -> List[float]:
    if df is None:
        return []
    rows = df[df["segment_idx"] == idx]
    if rows.empty:
        return []
    signal_cols = [c for c in df.columns if c.startswith("t")]
    return rows.iloc[0][signal_cols].values.astype(float).tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_from_segment(seg: Dict) -> Optional[Dict]:
    """
    Route to the correct feature extractor based on available channels.
    Mirrors the pipeline's logic: GT-Resp path → full feature set, else ECG-only.
    """
    if not HAS_PIPELINE:
        return None

    ecg   = np.array(seg["ecg"],   dtype=np.float32)
    pleth = np.array(seg.get("pleth", []), dtype=np.float32)
    resp  = np.array(seg.get("resp",  []), dtype=np.float32)
    abp   = np.array(seg.get("abp",  []), dtype=np.float32)

    has_pleth = seg.get("has_pleth", False) and len(pleth) > 0 and np.any(pleth != 0)
    has_resp  = seg.get("has_resp",  False) and len(resp)  > 0 and np.any(resp  != 0)
    has_abp   = seg.get("has_abp",   False) and len(abp)   > 0 and np.any(abp   != 0)

    try:
        if has_pleth or has_resp or has_abp:
            # Full feature extraction path (MIMIC-style)
            r_peaks = _detect_r_peaks(ecg, FS_ECG)
            edr     = _compute_edr(ecg, r_peaks, FS_ECG, FS_RESP)

            # Pad/trim to expected lengths
            spp = FS_PPG  * SEG_LEN
            spr = FS_RESP * SEG_LEN
            spa = FS_ECG  * SEG_LEN

            pleth_in = _pad_or_trim(pleth, spp, 97.0) if has_pleth else np.full(spp, 97.0)
            abp_in   = _pad_or_trim(abp,   spa, 80.0) if has_abp   else np.zeros(spa)
            resp_gt  = _pad_or_trim(resp,  spr, 0.0)  if has_resp  else None

            baseline = {
                "baseline_spo2":    float(np.mean(pleth_in)) if has_pleth else 97.0,
                "baseline_rmssd":   35.0,
                "baseline_rr_ms":   833.0,
                "baseline_map_std": 5.0,
                "baseline_sbp":     120.0,
            }

            feats = _extract_apnea_features(
                ecg, pleth_in, edr, abp_in, r_peaks, baseline,
                resp_gt=resp_gt, has_abp_signal=has_abp,
            )
        else:
            # ECG-only path (SLPDB-style)
            feats = _extract_features_slpdb(ecg)

        feats["segment_idx"] = seg["segment_idx"]
        feats["true_label"]  = seg["true_label"]
        return feats

    except Exception as e:
        print(f"  WARNING: Feature extraction failed for segment {seg['segment_idx']}: {e}")
        return None


def _pad_or_trim(arr: np.ndarray, target_len: int, fill: float = 0.0) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), constant_values=fill)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    features_df: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    feature_cols_path: str,
    threshold: float = 0.5,
    timesteps: int = 10,
):
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed — cannot run inference")
        return

    print(f"\nLoading model: {model_path}")
    # GatherFlags is imported at module level from pipeline.pipeline, which
    # triggers @keras.saving.register_keras_serializable — so load_model
    # can locate it with no custom_objects needed.
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(feature_cols_path) as f:
        feat_cols = json.load(f)

    # Align columns
    for col in feat_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feat_cols].fillna(0.0).values.astype(float)
    y = features_df["true_label"].values.astype(int)

    X_scaled = scaler.transform(X)

    if len(X_scaled) <= timesteps:
        print(f"WARNING: Need > {timesteps} segments for sequence model; got {len(X_scaled)}")
        return

    X_seq = np.array([X_scaled[i: i + timesteps] for i in range(len(X_scaled) - timesteps)])
    y_seq = y[timesteps:]

    print(f"Running inference on {len(X_seq)} sequences ...")
    y_prob = model.predict(X_seq, verbose=0).flatten()
    y_pred = (y_prob > threshold).astype(int)

    # Per-segment results
    results = pd.DataFrame({
        "segment_idx": features_df["segment_idx"].values[timesteps:],
        "true_label":  y_seq,
        "pred_prob":   y_prob.round(4),
        "pred_label":  y_pred,
        "correct":     (y_pred == y_seq).astype(int),
    })

    print("\n── Per-segment predictions (first 30) ──────────────────────────")
    print(results.head(30).to_string(index=False))

    # Summary metrics
    from sklearn.metrics import classification_report, roc_auc_score
    print("\n── Classification Report ────────────────────────────────────────")
    print(classification_report(y_seq, y_pred, target_names=["Normal", "Apnea"]))

    if len(np.unique(y_seq)) > 1:
        auc = roc_auc_score(y_seq, y_prob)
        print(f"AUC: {auc:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Load converted EDF data and test the apnea pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",     "-d", required=True,
                        help="Converted JSON file or directory of CSVs")
    parser.add_argument("--mode",     choices=["features", "infer"], default="features",
                        help="features: extract only; infer: features + BiLSTM inference")
    parser.add_argument("--model",    default="apnea_model.keras",  help="Saved .keras model")
    parser.add_argument("--scaler",   default="apnea_scaler.pkl",   help="Saved scaler pickle")
    parser.add_argument("--features", default="apnea_feature_cols.json", help="Feature cols JSON")
    parser.add_argument("--threshold", type=float, default=0.5,     help="Classification threshold")
    parser.add_argument("--out-csv",  default=None,
                        help="Save feature matrix to this CSV path")
    args = parser.parse_args()

    # ── Load segments ─────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if data_path.suffix.lower() == ".json":
        segments = load_json(str(data_path))
    elif data_path.is_dir():
        # Check for JSON files first, then CSVs
        json_files = list(data_path.glob("*.json"))
        if json_files:
            segments = []
            for jf in json_files:
                segments.extend(load_json(str(jf)))
        else:
            segments = load_csv_dir(str(data_path))
    else:
        print(f"ERROR: --data must be a .json file or a directory")
        sys.exit(1)

    if not segments:
        print("No segments loaded.")
        sys.exit(1)

    # ── Feature extraction ────────────────────────────────────────────────────
    if not HAS_PIPELINE:
        print("\nSkipping feature extraction (pipeline not importable).")
        print("Segment labels summary:")
        labels = [s["true_label"] for s in segments]
        print(f"  Total: {len(labels)} | Apnea: {sum(labels)} | Normal: {len(labels)-sum(labels)}")
        return

    print(f"\nExtracting features from {len(segments)} segments ...")
    feat_rows = []
    for i, seg in enumerate(segments):
        if i % 50 == 0:
            print(f"  {i}/{len(segments)} ...")
        feats = extract_features_from_segment(seg)
        if feats:
            feat_rows.append(feats)

    if not feat_rows:
        print("No features extracted.")
        sys.exit(1)

    features_df = pd.DataFrame(feat_rows)
    print(f"\n── Feature Summary ─────────────────────────────────────────────")
    print(f"  Segments with features : {len(features_df)}")
    print(f"  Feature columns        : {len([c for c in features_df.columns if c in APNEA_FEATURE_COLS])}")
    n_apnea  = int((features_df["true_label"] == 1).sum())
    n_normal = int((features_df["true_label"] == 0).sum())
    print(f"  Label distribution     : {n_apnea} apnea / {n_normal} normal "
          f"({100*n_apnea/max(len(features_df),1):.0f}%)")

    # ECG feature preview
    ecg_feats = ["rr_mean", "rr_std", "rmssd", "mean_hr", "resp_rate_bpm"]
    present   = [c for c in ecg_feats if c in features_df.columns]
    if present:
        print("\n── ECG Feature Preview (mean by class) ─────────────────────────")
        print(features_df.groupby("true_label")[present].mean().round(2).to_string())

    if args.out_csv:
        features_df.to_csv(args.out_csv, index=False)
        print(f"\nFeature matrix saved → {args.out_csv}")

    # ── Inference ─────────────────────────────────────────────────────────────
    if args.mode == "infer":
        if not os.path.exists(args.model):
            print(f"\nERROR: Model not found at {args.model}")
            print("  Train the pipeline first with --save-model, then re-run with --mode infer")
            sys.exit(1)
        run_inference(
            features_df, args.model, args.scaler, args.features, args.threshold
        )


if __name__ == "__main__":
    main()