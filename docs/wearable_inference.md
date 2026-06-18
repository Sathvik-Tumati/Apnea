# Wearable Device Inference Guide

The pipeline supports two inference paths:

1. **MongoDB Path** (primary): Pulls ECG directly from the hospital database, applies ECG-only sleep filtering, and runs inference automatically. See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).
2. **EDF/Local Path** (offline): Converts local `.edf` files from consumer wearables and runs inference using the full multi-signal toolchain.

This document covers the **EDF/Local Path**.

---

## The Challenge

Consumer wearables (Apple Watch, Garmin, chest straps) differ from clinical monitors in several ways:

| Aspect | Clinical Monitor (MIMIC/SLPDB) | Consumer Wearable |
|---|---|---|
| Signals | ECG + SpO2 + Resp + ABP | ECG + maybe PPG |
| ECG channels | 12-lead or 3-lead | 1 lead (often noisy) |
| Duration | Hours (ICU stay) | Full day (24h+) |
| Sample rate | Known exactly | Varies (250, 512, 1000 Hz) |
| Timestamps | Precise to millisecond | Often approximated |

The pipeline handles all of this automatically:
- Auto-detects sample rate from spectral analysis
- Resamples to 125 Hz standard
- Uses EDR when respiratory channels are absent
- Sets `has_spo2 = 0`, `has_abp = 0`, `has_resp_gt = 0` for ECG-only recordings

---

## Step 1: Convert EDF to Segment CSV

```bash
python pipeline/edf_to_pipeline.py \
    --input path/to/recording.edf \
    --mode csv \
    --out-dir pipeline/converted/
```

This:
- Reads the EDF file, identifies the ECG channel by name heuristics
- Detects and resamples to 125 Hz
- Splits into 30-second non-overlapping windows
- Saves as `pipeline/converted/recording_ecg.csv`

---

## Step 2: Sleep Filtering (Recommended)

For EDF files, use the multi-signal `sleep_filter.py` which can use activity, posture, HR, and skin temperature if available:

```bash
python pipeline/sleep_filter.py \
    --detect --filter \
    --input path/to/recording.edf \
    --csvs pipeline/converted/ \
    --out-dir pipeline/converted/sleep_only/
```

If only ECG is available (most wearables), it falls back to HR-based sleep scoring — similar in principle to `ecg_sleep_filter.py` in the automation path.

---

## Step 3: Run Inference

```bash
# Full BiLSTM inference
python pipeline/edf_test_loader.py \
    --data pipeline/converted/sleep_only/ \
    --mode infer \
    --model apnea_model.keras \
    --scaler apnea_scaler.pkl \
    --features apnea_feature_cols.json

# Feature extraction only (no model)
python pipeline/edf_test_loader.py \
    --data pipeline/converted/sleep_only/ \
    --mode features \
    --out-csv features.csv
```

The script will:
- Detect that SpO2/ABP/GT_Resp are missing, set modality flags to 0
- Use the dual-engine EDR algorithm for respiratory features
- Run the Modality-Aware BiLSTM with the appropriate flags
- Output a per-segment probability sequence and clinical summary

---

## Output

```
============================================================
  APNEA INFERENCE SUMMARY
============================================================
  Duration (wall)   : 300.5 min
  Total segments    : 601
  Flagged (skipped) : 0
  Scored by model   : 591
  Apnea detected    : 47  (7.9%)
  Mean apnea prob   : 0.18
  AHI proxy         : 9.4 /hr  → Mild

  AHI: <5 Normal | 5-15 Mild | 15-30 Moderate | >30 Severe
  NOTE: Research prototype. Not for clinical use.
============================================================
```

---

## Important Notes

- **Not for clinical use.** AHI proxy values are research estimates, not validated medical diagnoses.
- **Model was trained on two sources:** MIMIC-IV (ICU, high apnea rate) and SLPDB (sleep lab, diverse severity). It performs best on ECG signals similar to these — standard 125 Hz, filtered, no major motion artifacts.
- **neurokit2 strongly recommended.** Install it for accurate R-peak detection: `pip install neurokit2`. Without it, scipy fallback is used and HRV features are less accurate.
