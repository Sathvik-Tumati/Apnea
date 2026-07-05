"""
spo2_split_ahi.py
==================
Splits a validated admission into contiguous windows based on has_spo2
availability, and reports AHI (raw predicted + physiologically-validated)
separately for each window. This prevents SpO2 sensor dropout periods
from silently deflating the validated AHI for the whole recording.

Usage
-----
    python spo2_split_ahi.py --validated infer_output/<ADM>/<ADM>_validated.csv

Expects the CSV produced by apnea_validator.py's save_validated_csv(),
i.e. it must already contain: has_spo2, apnea_pred, validation_verdict,
segment_idx, start_time_s (or timestamp).
"""

import argparse
import pandas as pd
import numpy as np

SEGMENT_LEN_S = 30


def _contiguous_spo2_windows(df: pd.DataFrame):
    """
    Yield (has_spo2_flag, start_idx, end_idx) tuples for contiguous runs
    of has_spo2 == 0 or == 1, ordered by segment_idx.
    """
    df = df.sort_values("segment_idx").reset_index(drop=True)
    flags = df["has_spo2"].astype(float).astype(int).values
    n = len(flags)
    if n == 0:
        return
    start = 0
    for i in range(1, n + 1):
        if i == n or flags[i] != flags[start]:
            yield flags[start], start, i - 1
            start = i


def _window_ahi(df: pd.DataFrame, lo: int, hi: int):
    """Compute raw + validated AHI for df rows[lo:hi] inclusive (positional)."""
    window = df.iloc[lo:hi + 1]
    n_seg = len(window)
    dur_min = n_seg * SEGMENT_LEN_S / 60.0
    if dur_min <= 0:
        return dict(n_segments=n_seg, duration_min=0, n_predicted=0,
                    raw_ahi=0.0, n_confirmed_probable=0, validated_ahi=0.0)

    n_predicted = int((window["apnea_pred"].astype(float) == 1.0).sum())
    raw_ahi = n_predicted / max(dur_min / 60.0, 1e-6)

    verdicts = window.get("validation_verdict", pd.Series([""] * n_seg))
    n_conf_prob = int(verdicts.isin(["CONFIRMED", "PROBABLE"]).sum())
    validated_ahi = n_conf_prob / max(dur_min / 60.0, 1e-6)

    return dict(
        n_segments=n_seg,
        duration_min=round(dur_min, 1),
        n_predicted=n_predicted,
        raw_ahi=round(raw_ahi, 2),
        n_confirmed_probable=n_conf_prob,
        validated_ahi=round(validated_ahi, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", required=True,
                     help="validated CSV output from apnea_validator.py")
    args = ap.parse_args()

    df = pd.read_csv(args.validated, low_memory=False)
    if "has_spo2" not in df.columns:
        raise SystemExit("CSV missing has_spo2 column")

    df = df.sort_values("segment_idx").reset_index(drop=True)

    print("=" * 70)
    print("SpO2-AWARE AHI BREAKDOWN")
    print("=" * 70)

    overall = _window_ahi(df, 0, len(df) - 1)
    print(f"Whole recording : {overall}")
    print("-" * 70)

    for flag, lo, hi in _contiguous_spo2_windows(df):
        stats = _window_ahi(df, lo, hi)
        label = "HAS_SPO2" if flag == 1 else "NO_SPO2 (dropout)"
        seg_range = f"seg[{df.loc[lo, 'segment_idx']}-{df.loc[hi, 'segment_idx']}]"
        t0 = df.loc[lo, "timestamp"] if "timestamp" in df.columns else ""
        t1 = df.loc[hi, "timestamp"] if "timestamp" in df.columns else ""
        print(f"{label:20s} {seg_range:18s} {t0} -> {t1}")
        for k, v in stats.items():
            print(f"    {k:22s}: {v}")
        print()

    # Aggregate: validated AHI restricted to has_spo2==1 segments only
    spo2_ok = df[df["has_spo2"].astype(float) == 1]
    if len(spo2_ok) > 0:
        stats = _window_ahi(spo2_ok.reset_index(drop=True), 0, len(spo2_ok) - 1)
        print("=" * 70)
        print(f"AGGREGATE (has_spo2==1 segments only, non-contiguous): {stats}")
        print("=" * 70)


if __name__ == "__main__":
    main()