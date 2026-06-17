# Complete Pipeline Guide

This document walks through every step needed to run the apnea detection pipeline —
from environment setup and training, to wearable EDF inference and live MongoDB inference.

> [!IMPORTANT]
> **Always run commands from the `project2/` root**, not from inside `pipeline/`. Import paths depend on this.

---

## 1. Prerequisites

### Python Environment
The pipeline requires **Python 3.9** and the packages below. Use the provided `venv` or create a fresh one:

```bash
# If you have the original venv from training, symlink it back:
ln -s /path/to/project2_git_ignored_files/pipeline/venv ./venv

# Or create a new one:
python3.9 -m venv venv
source venv/bin/activate
pip install tensorflow keras numpy pandas scipy scikit-learn \
            wfdb pyedflib neurokit2 pymongo joblib
```

### Environment Variables (for MongoDB inference only)
```bash
export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/"
export MONGO_DB="your_database_name"
# Optional overrides:
export MODEL_PATH="apnea_model.keras"
export SCALER_PATH="apnea_scaler.pkl"
export THRESHOLD="0.60"
```

---

## 2. Train the Model

This downloads MIMIC-IV and SLPDB automatically, runs feature extraction, trains the BiLSTM, and saves the model artefacts.

```bash
source venv/bin/activate

# First run — downloads data, trains, saves model files
python pipeline/pipeline.py --save-model

# Force a clean re-run (deletes DB + re-downloads everything)
python pipeline/pipeline.py --fresh --save-model

# Train on MIMIC only (skip SLPDB)
python pipeline/pipeline.py --save-model --no-slpdb

# Use only specific SLPDB records (faster for testing)
python pipeline/pipeline.py --save-model --slpdb-records slp37 slp41 slp66
```

**Output files written to `project2/`:**

| File | Contents |
|---|---|
| `apnea_model.keras` | Full trained BiLSTM model |
| `apnea_best.keras` | Best checkpoint (highest val AUC) |
| `apnea_scaler.pkl` | Fitted StandardScaler |
| `apnea_feature_cols.json` | Ordered list of the 30 feature columns |
| `apnea_thresholds.json` | Optimal thresholds `{global, mimic, slpdb}` |

---

## 3. Wearable EDF Inference (Local Files)

Use this path when you have `.edf` recordings from a consumer wearable (Apple Watch, Garmin, chest strap, etc.).

### Step 3a — Convert EDF → Segments CSV
```bash
python pipeline/edf_to_pipeline.py \
    --input path/to/recording.edf \
    --mode csv \
    --out-dir pipeline/converted/
```

### Step 3b — Filter to Sleep Windows (Recommended)
```bash
python pipeline/sleep_filter.py \
    --detect --filter \
    --input path/to/recording.edf \
    --csvs pipeline/converted/ \
    --out-dir pipeline/converted/sleep_only/
```

### Step 3c — Run Inference
```bash
# Feature extraction only (no model needed)
python pipeline/edf_test_loader.py \
    --data pipeline/converted/sleep_only/ \
    --mode features

# Full BiLSTM inference
python pipeline/edf_test_loader.py \
    --data pipeline/converted/sleep_only/ \
    --mode infer \
    --model apnea_model.keras \
    --scaler apnea_scaler.pkl \
    --features apnea_feature_cols.json

# Save the feature matrix for inspection
python pipeline/edf_test_loader.py \
    --data pipeline/converted/sleep_only/ \
    --mode infer \
    --model apnea_model.keras \
    --scaler apnea_scaler.pkl \
    --out-csv features.csv
```

---

## 4. MongoDB Live Inference

Use this path when ECG data lives in a MongoDB database (e.g., streaming from a hospital monitor or wearable backend).

> [!IMPORTANT]
> Make sure `MONGO_URI` and `MONGO_DB` are set in your environment before running.

### Single Admission
```bash
python automation/mongo_infer.py \
    --admission ADM1819906487 \
    --model apnea_model.keras \
    --scaler apnea_scaler.pkl
```

### All Admissions from the Last 24 Hours
```bash
python automation/mongo_infer.py --since 24h
```

### Date Range
```bash
python automation/mongo_infer.py \
    --from 2026-06-01 \
    --to 2026-06-15
```

### Dry Run (CSV extraction only, skip inference)
```bash
python automation/mongo_infer.py \
    --admission ADM1819906487 \
    --dry-run
```

### Write Results Back to MongoDB
```bash
python automation/mongo_infer.py \
    --since 24h \
    --write-mongo
```

**Output directory structure (under `infer_output/`):**
```
infer_output/
└── ADM1819906487/
    ├── ADM1819906487_segments.csv   ← raw segments
    └── infer_summary.csv            ← per-admission AHI / severity
```

---

## 5. Configuration Reference

### `pipeline/modules/config.py` Key Constants

| Constant | Default | Description |
|---|---|---|
| `FS_ECG` | 125 Hz | ECG target sampling rate |
| `FS_PPG` | 120 Hz | SpO2 / Pleth sampling rate |
| `FS_RESP` | 4 Hz | Respiratory signal rate |
| `SEGMENT_LEN_S` | 30 s | Window length |
| `TIMESTEPS` | 10 | Sequences fed to BiLSTM |
| `N_MIMIC_RECORDS` | 60 | MIMIC-IV records to download |

### `automation/mongo_infer.py` Key Constants

| Constant | Default | Description |
|---|---|---|
| `MIN_SEGMENTS` | 11 | Minimum segments before inference (LSTM needs ≥11 for first prediction) |
| `MIN_DURATION_MINUTES` | 30 | Skip recordings shorter than this |
| `COMPLETION_GAP_HOURS` | 2 | Hours without new packets → recording considered done |
| `THRESHOLD` | 0.60 | Classification probability threshold |

---

## 6. Directory Reference

```
project2/
├── pipeline/
│   ├── modules/            ← Core ML modules (config, features, model, train, evaluate)
│   ├── db/                 ← SQLite persistence (database.py)
│   ├── utils/              ← Diagnostic scripts
│   ├── pipeline.py         ← Main training orchestrator  ← RUN THIS TO TRAIN
│   ├── edf_to_pipeline.py  ← EDF converter
│   ├── edf_test_loader.py  ← Wearable inference harness
│   ├── infer.py            ← Core inference engine (used by mongo_infer.py)
│   └── sleep_filter.py     ← Sleep window detector
├── automation/
│   └── mongo_infer.py      ← MongoDB → ECG → Inference  ← RUN THIS FOR LIVE DATA
├── docs/
│   ├── architecture.md
│   ├── feature_engineering.md
│   └── wearable_inference.md
├── apnea_model.keras        ← Saved model (after --save-model)
├── apnea_scaler.pkl
├── apnea_feature_cols.json
└── apnea_thresholds.json
```

---

## 7. Common Issues

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: tensorflow` | Wrong venv active | `source venv/bin/activate` |
| `TypeError: Could not locate class 'GatherFlags'` | `pipeline.modules.model` not imported before `load_model` | Run from project root, not from inside `pipeline/` |
| `wfdb not installed` | Missing dependency | `pip install wfdb` |
| `No MIMIC records fetched` | Network / PhysioNet timeout | Retry; PhysioNet is rate-limited |
| `MongoDB connection failed` | Wrong URI or env vars not set | Check `MONGO_URI` / `MONGO_DB` |
| `Only one class in test set` | Too few SLPDB records / all-normal MIMIC run | Add more records or use `--slpdb-records` with diverse records |
