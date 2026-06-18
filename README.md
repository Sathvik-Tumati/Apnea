# Apnea Detection Pipeline

[![Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Sathvik-Tumati/Apnea)

A multi-source, modality-aware sleep apnea detection system built on a Bidirectional LSTM (BiLSTM). The pipeline trains on two clinical datasets (MIMIC-IV and SLPDB), extracts physiological features from ECG and SpO2 signals, and runs fully automated inference on live patient recordings pulled from MongoDB — with results written to Supabase.

---

## 📚 Documentation

For deep dives into the technical architecture and logic, see the `docs/` directory:
- [**Complete Pipeline Guide**](docs/PIPELINE_GUIDE.md): Step-by-step commands for training, EDF inference, and live MongoDB inference.
- [**Modality-Aware BiLSTM Architecture**](docs/architecture.md): How we handle missing signals using Modality Flags and Modality Dropout.
- [**Feature Engineering**](docs/feature_engineering.md): Details on all 30 features extracted per segment.
- [**Sleep Detection**](docs/sleep_detection.md): How `ecg_sleep_filter.py` isolates sleep-only epochs before inference.

---

## Repository Layout

```
project2/
├── pipeline/
│   ├── modules/                  # Core ML modules
│   │   ├── config.py             # Feature columns, constants, index mapping
│   │   ├── features.py           # Signal processing and feature extraction
│   │   ├── ingest_mimic.py       # MIMIC-IV streaming and pseudo-labelling
│   │   ├── ingest_slpdb.py       # SLPDB streaming and annotation parsing
│   │   ├── model.py              # Modality-Aware BiLSTM + GatherFlags layer
│   │   └── train.py              # Dataset building and training loop
│   ├── pipeline.py               # Training orchestrator  ← RUN TO TRAIN
│   ├── infer.py                  # Core inference engine (called by mongo_infer.py)
│   ├── sleep_filter.py           # EDF-based sleep filter (offline / EDF path only)
│   └── db/database.py            # SQLite persistence for training runs
├── automation/
│   ├── mongo_infer.py            # MongoDB → ECG+SpO2 → Sleep filter → Inference → Supabase
│   └── ecg_sleep_filter.py       # ECG-only sleep detection (IST time gate + HRV scoring)
├── docs/                         # Technical documentation
├── apnea_model.keras             # Saved trained model
├── apnea_best.keras              # Best checkpoint (highest val AUC during training)
├── apnea_scaler.pkl              # Fitted StandardScaler (30 features)
├── apnea_feature_cols.json       # Ordered list of the 30 feature column names
└── apnea_thresholds.json         # Optimal thresholds {global, mimic, slpdb}
```

> **Important:** Always run commands from `project2/` (the project root), not from inside `pipeline/` or `automation/`. Import paths depend on this.

---

## Quick Start — Live MongoDB Inference

This is the primary use-case. Make sure your `.env` file is set up (see [Environment Variables](#environment-variables)), then run:

```bash
source venv/bin/activate

# Single admission — full pipeline (sleep filter + inference + Supabase write)
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase

# All admissions from the last 24 hours
python automation/mongo_infer.py --since 24h --write-supabase

# Dry run — extract and sleep-filter the CSV, skip inference
python automation/mongo_infer.py --admission ADM1819906487 --dry-run

# Skip sleep filtering (e.g. short recording, ICU patient)
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase --no-sleep-filter

# Re-process an admission that already has a Supabase result
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase --reprocess
```

**What happens on each run:**

```
1. SSH tunnel → MongoDB
2. Pull ECG (30812 docs → 3,851,500 samples for a ~8h recording)
3. Pull SpO2 (1 Hz device-computed values)
4. ECG sleep detection (IST time gate + HR/SDNN scoring) → isolate sleep epochs
5. Build segment CSV (30 s windows, SpO2 aligned by timestamp)
6. Run BiLSTM inference (feature extraction → scale → model → AHI proxy)
7. Upsert results → Supabase apnea_results table
```

---

## Train the Model

```bash
source venv/bin/activate

# First run — downloads MIMIC-IV + SLPDB, trains BiLSTM, saves model artefacts
python pipeline/pipeline.py --fresh --save-model

# Subsequent runs reuse cached data (much faster)
python pipeline/pipeline.py --save-model

# Train on specific SLPDB records only (fast iteration)
python pipeline/pipeline.py --save-model --slpdb-records slp37 slp41 slp66
```

**Saved artefacts (in `project2/`):**

| File | Contents |
|---|---|
| `apnea_model.keras` | Full trained BiLSTM model |
| `apnea_best.keras` | Best val AUC checkpoint |
| `apnea_scaler.pkl` | Fitted StandardScaler (30 features) |
| `apnea_feature_cols.json` | Ordered feature column list |
| `apnea_thresholds.json` | `{"global": 0.51, "mimic": 0.52, "slpdb": 0.57}` |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# MongoDB
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=your_database_name

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Model paths (defaults shown — override if needed)
MODEL_PATH=apnea_model.keras
SCALER_PATH=apnea_scaler.pkl
THRESHOLD=0.45
OUTPUT_DIR=infer_output/
```

The `.env` file is automatically loaded by `mongo_infer.py` when it starts (via `python-dotenv`).

---

## Supabase Schema

Run this once in the Supabase SQL editor to create the results table:

```sql
CREATE TABLE IF NOT EXISTS apnea_results (
    admission_id      TEXT PRIMARY KEY,
    facility_id       TEXT,
    processed_at      TIMESTAMPTZ,
    ahi_proxy         FLOAT,
    severity          TEXT,
    apnea_pct         FLOAT,
    total_segments    INT,
    scored_segments   INT,
    n_apnea           INT,
    duration_min      FLOAT,
    model_threshold   FLOAT
);
```

---

## Output Files

After each run, `infer_output/<admissionId>/` contains:

| File | Contents |
|---|---|
| `ADM*_segments.csv` | Sleep-filtered, feature-enriched segment CSV |
| `ADM*_sleep_windows.csv` | Sleep detection results (is_sleep, window_id, HR, SDNN) |
| `infer_results_ADM*.csv` | Per-segment model predictions and features |
| `infer_summary.csv` | One row per admission: AHI, severity, duration, etc. |
| `infer_summary.txt` | Human-readable clinical summary |

---

## Dependencies

```bash
# Core ML
tensorflow >= 2.x
keras
numpy
pandas
scipy
scikit-learn
joblib

# Signal processing
neurokit2     # R-peak detection (optional but strongly recommended)

# Data sources (training only)
wfdb          # PhysioNet / MIMIC / SLPDB access

# Database & connectivity
pymongo       # MongoDB client
supabase      # Supabase Python client
python-dotenv # .env loading
```

Install:
```bash
source venv/bin/activate
pip install tensorflow keras numpy pandas scipy scikit-learn joblib \
            neurokit2 wfdb pymongo supabase python-dotenv
```

---

> **Research Prototype.** This system is not validated for clinical use. AHI values are proxies derived from ECG alone and should not replace polysomnography.
