# Apnea Detection Pipeline

[![Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Sathvik-Tumati/Apnea)

A multi-source, modality-aware sleep apnea detection system that trains **two complementary models** in a single pipeline run:
- **Modality-Aware BiLSTM** — deep sequence model with ECG/SpO2/ABP modality flags
- **XGBoost (seq)** — gradient-boosted trees operating on the same flattened sequence features

Both models train on MIMIC-IV and SLPDB with the same train/val/test split, so results are directly comparable. At inference time, both models run on each admission and their outputs are compared — disagreements are flagged for clinical review.

---

## 📚 Documentation

For deep dives into the technical architecture and logic, see the `docs/` directory:
- [**Complete Pipeline Guide**](docs/PIPELINE_GUIDE.md): Step-by-step commands for training, EDF inference, and live MongoDB inference.
- [**Model Architecture**](docs/architecture.md): BiLSTM and XGBoost model details, modality flags, and modality dropout.
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
├── apnea_validator.py            # Post-inference physiological plausibility validator ← NEW
├── apnea_model.keras             # BiLSTM saved model
├── apnea_best.keras              # BiLSTM best checkpoint (highest val AUC)
├── apnea_scaler.pkl              # BiLSTM fitted StandardScaler (30 features)
├── apnea_feature_cols.json       # Ordered list of the 30 feature column names
├── apnea_thresholds.json         # BiLSTM optimal thresholds {global, mimic, slpdb}
├── apnea_model_xgb_seq.pkl       # XGBoost sequence model (primary inference model)
├── apnea_model_xgb_flat.pkl      # XGBoost flat model (per-segment, no sequence context)
├── apnea_model_lgbm_seq.pkl      # LightGBM sequence model
├── apnea_model_lgbm_flat.pkl     # LightGBM flat model
└── apnea_scaler_tree.pkl         # Shared StandardScaler for XGBoost/LightGBM
```

> **Important:** Always run commands from `project2/` (the project root), not from inside `pipeline/` or `automation/`. Import paths depend on this.

---

## Quick Start — Live MongoDB Inference

This is the primary use-case. Make sure your `.env` file is set up (see [Environment Variables](#environment-variables)), then run:

```bash
source venv/bin/activate

# Single admission — full pipeline (sleep filter + dual-model inference + Supabase write)
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase

# Use XGBoost as primary, BiLSTM as consensus
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase \
    --model apnea_model_xgb_seq.pkl \
    --scaler apnea_scaler_tree.pkl \
    --threshold 0.55 \
    --model-bilstm apnea_model.keras \
    --scaler-bilstm apnea_scaler.pkl

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
6. Primary model inference (XGBoost) → AHI proxy
7. Secondary model inference (BiLSTM, optional) → consensus check
8. If both models disagree → flagged for clinical review
9. Upsert results → Supabase apnea_results table
```

---

## Post-Inference Validation

After inference, run the physiological plausibility validator to verify model predictions against cardiorespiratory physiology:

```bash
# Validate predictions for one admission (reads the infer_results CSV)
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv

# Verbose mode — shows per-event score breakdown
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --verbose

# Save validated CSV (adds validation_score, verdict, spo2_drop_pct columns)
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --out   infer_output/ADM1819906487/validated_results.csv

# Update Supabase with validation results
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --write-supabase
```

**Scoring criteria (total 0–100 points):**

| Criterion | Weight | What it checks |
|---|---|---|
| HR/RR ratio pattern | 40 pts | RR interval increases + HR spike at termination |
| SpO2 desaturation | 35 pts | ≥3% drop within ±2 segments of event (accounts for circulation lag) |
| Respiratory suppression | 15 pts | EDR resp amplitude drop + elevated rate variability |
| HR dip-surge pattern | 10 pts | Classic bradycardia during apnea → tachycardia at termination |

**Verdicts:**
- `✓ CONFIRMED` (≥60) — strong physiological support
- `~ PROBABLE` (40–59) — partial support, review recommended  
- `? UNCERTAIN` (20–39) — weak support, may be artifact
- `✗ UNCONFIRMED` (<20) — no physiological support, likely false positive

The **validated AHI** (CONFIRMED + PROBABLE events only) is a more conservative estimate than the raw model AHI.

---

## Train the Model

```bash
source venv/bin/activate

# Default: trains BOTH BiLSTM and XGBoost on MIMIC-IV + SLPDB
# Note: --fresh prompts for confirmation before deleting the DB
python pipeline/pipeline.py --fresh --save-model

# Subsequent runs reuse cached data (no prompt, much faster)
python pipeline/pipeline.py --save-model

# Train only one model
python pipeline/pipeline.py --save-model --bilstm-only
python pipeline/pipeline.py --save-model --xgb-only

# Train on specific SLPDB records only (fast iteration)
python pipeline/pipeline.py --save-model --slpdb-records slp37 slp41 slp66
```

> **Note:** When running `--xgb-only`, XGBoost reads segments already in `vitals_pipeline.db` from a prior BiLSTM run. If the DB is empty, run BiLSTM first or use `--fresh --save-model` (no `--xgb-only`).

**Saved artefacts (in `project2/`):**

| File | Model | Contents |
|---|---|---|
| `apnea_model.keras` | BiLSTM | Full trained BiLSTM model |
| `apnea_best.keras` | BiLSTM | Best val AUC checkpoint |
| `apnea_scaler.pkl` | BiLSTM | Fitted StandardScaler (30 features) |
| `apnea_feature_cols.json` | BiLSTM | Ordered feature column list |
| `apnea_thresholds.json` | BiLSTM | `{"global": 0.51, "mimic": 0.52, "slpdb": 0.57}` |
| `apnea_model_xgb_seq.pkl` | XGBoost | Trained XGBClassifier — sequences flattened to T×F |
| `apnea_model_xgb_flat.pkl` | XGBoost | Per-segment XGBoost (no sequence context) |
| `apnea_model_lgbm_seq.pkl` | LightGBM | LightGBM on same T×F flattened sequences |
| `apnea_model_lgbm_flat.pkl` | LightGBM | Per-segment LightGBM |
| `apnea_scaler_tree.pkl` | XGB/LGBM | Shared StandardScaler for all tree models |

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

# Primary model (XGBoost seq — used as --model in mongo_infer.py)
MODEL_PATH=apnea_model_xgb_seq.pkl
SCALER_PATH=apnea_scaler_tree.pkl
THRESHOLD=0.55

# Secondary BiLSTM model (optional — enables consensus mode)
BILSTM_MODEL_PATH=apnea_model.keras
BILSTM_SCALER_PATH=apnea_scaler.pkl

# Output
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
    apnea_label       TEXT,        -- 'No Apnea' | 'Mild' | 'Moderate' | 'Severe'
    has_apnea         BOOLEAN,     -- true if ahi_proxy >= 5
    apnea_pct         FLOAT,
    total_segments    INT,
    scored_segments   INT,
    n_apnea           INT,
    duration_min      FLOAT,
    model_threshold   FLOAT,
    -- Consensus fields (populated when BiLSTM model is provided)
    ahi_bilstm        FLOAT,
    severity_bilstm   TEXT,
    models_agree      BOOLEAN,
    needs_review      BOOLEAN      -- true when XGBoost and BiLSTM disagree
);

-- Run this ALTER after creating the table to add validation columns:
ALTER TABLE apnea_results
    ADD COLUMN IF NOT EXISTS validated_ahi             FLOAT,
    ADD COLUMN IF NOT EXISTS validation_confirmed      INT,
    ADD COLUMN IF NOT EXISTS validation_probable       INT,
    ADD COLUMN IF NOT EXISTS validation_uncertain      INT,
    ADD COLUMN IF NOT EXISTS validation_unconfirmed    INT,
    ADD COLUMN IF NOT EXISTS validation_mean_score     FLOAT,
    ADD COLUMN IF NOT EXISTS physiologically_supported BOOLEAN;
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
xgboost        # Required for XGBoost model training and inference

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
pip install tensorflow keras numpy pandas scipy scikit-learn joblib xgboost \
            neurokit2 wfdb pymongo supabase python-dotenv
```

---

> **Research Prototype.** This system is not validated for clinical use. AHI values are proxies derived from ECG alone and should not replace polysomnography.
