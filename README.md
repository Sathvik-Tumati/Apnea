# Apnea Detection Pipeline

[![Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Sathvik-Tumati/Apnea)

A multi-source, modality-aware sleep apnea detection system that trains an **XGBoost (seq)** model — gradient-boosted trees operating on flattened sequence features.

The model trains on MIMIC-IV and SLPDB with a stratified train/val/test split. At inference time, the model runs on each admission and its output is flagged for clinical review if the physiological validator disagrees.

---

## 📚 Documentation

For deep dives into the technical architecture and logic, see the `docs/` directory:
- [**Complete Pipeline Guide**](docs/PIPELINE_GUIDE.md): Step-by-step commands for training, EDF inference, and live MongoDB inference.
- [**Model Architecture**](docs/architecture.md): XGBoost model details, modality flags, and modality dropout.
- [**Feature Engineering**](docs/feature_engineering.md): Details on all 30 features extracted per segment.
- [**Sleep Detection**](docs/sleep_detection.md): How `ecg_sleep_filter.py` isolates sleep-only epochs before inference.
- [**Apnea Validator**](docs/apnea_validator.md): Full technical reference for `apnea_validator.py` — scoring logic, constants, output columns, dual-mechanism primary criterion, and calibration guide.

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
│   │   └── train.py              # Dataset building and training loop
│   ├── pipeline.py               # Training orchestrator  ← RUN TO TRAIN
│   ├── infer.py                  # Core inference engine (called by mongo_infer.py)
│   ├── sleep_filter.py           # EDF-based sleep filter (offline / EDF path only)
│   └── db/database.py            # SQLite persistence for training runs
├── automation/
│   ├── mongo_infer.py            # MongoDB → ECG+SpO2 → Sleep filter → Inference → Supabase
│   └── ecg_sleep_filter.py       # ECG-only sleep detection (IST time gate + HRV scoring)
├── docs/                         # Technical documentation
├── apnea_validator.py            # Post-inference physiological plausibility validator
├── spo2_split_ahi.py             # Diagnostic: per-window SpO2-aware AHI breakdown
├── apnea_feature_cols.json       # Ordered list of the 30 feature column names
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
7. Upsert results → Supabase apnea_results table
```

---

## Post-Inference Validation

After inference, run the physiological plausibility validator to cross-check every model prediction against the patient's actual cardiorespiratory physiology. This is the "second opinion" layer that separates false positives from real apnea events.

```bash
# Validate predictions (reads the infer_results CSV, outputs a validated CSV)
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv

# Verbose mode — prints a full per-event score breakdown to the log
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --verbose

# Specify a custom output path for the validated CSV
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --out   infer_output/ADM1819906487/ADM1819906487_validated.csv

# Write validation results back to Supabase
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --write-supabase
```

**Scoring criteria (total 0–100 points):**

| Criterion | Weight | What it checks |
|---|---|---|
| **Primary — Combined** (max of two mechanisms, see below) | 40 pts | Whichever of the two mechanisms scores higher is used |
| SpO2 desaturation | 35 pts | ≥3% drop within ±2 segments of event (checks both `spo2_min` and `spo2_mean`) |
| Respiratory suppression | 15 pts | EDR resp amplitude drop + elevated rate variability |
| HR dip-surge (fallback) | 10 pts | Classic bradycardia during apnea → tachycardia at termination |

**Primary criterion — dual mechanism (Criterion 1):**

The primary criterion computes **both** of the following mechanisms for every event and uses the **higher score**:

- **HR/RespRate ratio**: During apnea, respiratory rate falls toward zero while HR stays elevated (sympathetic arousal). The ratio HR÷RespRate rises sharply. A ≥15% ratio increase scores partial credit; ≥80% increase scores full credit. Requires ≥2 valid pre-event baseline segments — returns `NA` if insufficient data.

- **HR drop/surge**: Classic vagal bradycardia during apnea (HR ≥3 bpm below pre-event baseline) followed by sympathetic tachycardia at termination (HR ≥4 bpm surge in the following segment). Always computed when baseline and event HR exist.

The `method_used` field in the output CSV records which mechanism won (`"ratio"`, `"hr_drop"`, `"both"`, or `"neither"`). Both computed values are shown in `--verbose` output regardless of which one won, so you can audit the scoring.

**Handling missing data:**
- **Dynamic scoring**: If SpO2 is structurally unavailable (`has_spo2=0` for both baseline and event windows), its 35 points are redistributed proportionally across the other three criteria so the maximum achievable score stays at 100 — not 65.
- **Cluster surge bonus**: When back-to-back apnea events form a dense cluster, the heart has no time to recover between events. The validator searches for a single HR recovery surge at the **first clean segment after the cluster ends**. If found, every event in the cluster gets +8 points (capped at 59 — the bonus alone can never push a verdict to CONFIRMED).

**Data completeness flags:**

Events scoring below 60 are tagged with a `data_completeness` flag that tells reviewers *why* they failed:

| Flag | Meaning |
|---|---|
| `complete` | All data was available — the event genuinely lacks physiological support |
| `insufficient_spo2` | SpO2 sensor was unavailable; SpO2 check could not be performed |
| `insufficient_trailing` | Recording ended before the post-cluster recovery window; surge search incomplete |
| `insufficient_both` | Both SpO2 and trailing window were unavailable |

**AHI range reporting (temporary):**

Because events flagged as data-incomplete may be real events the validator simply couldn't check, AHI is reported as a **range**:
- **Lower bound** (`validated_ahi`): CONFIRMED + PROBABLE events only.
- **Upper bound** (`validated_ahi_upper`): Adds all data-incomplete UNCERTAIN/UNCONFIRMED events as if they were real.

If the upper bound crosses a severity threshold the lower bound doesn't, the system logs a flag recommending manual review before finalizing the severity classification.

**Verdicts:**
- `✓ CONFIRMED` (≥60) — strong physiological support
- `~ PROBABLE` (40–59) — partial support, manual review recommended
- `? UNCERTAIN` (20–39) — weak support, likely artifact
- `✗ UNCONFIRMED` (<20) — no physiological support, likely a false positive

**SpO2-aware AHI breakdown (diagnostic):**

When SpO2 sensor dropout and apnea events coincide, a single blended AHI can be misleading. Use `spo2_split_ahi.py` to split the recording into contiguous SpO2-available / dropout windows and report raw + validated AHI for each:

```bash
python spo2_split_ahi.py \
    --validated infer_output/ADM1819906487/ADM1819906487_validated.csv
```

This is a standalone diagnostic — it does not modify any files.

---

## Train the Model

```bash
source venv/bin/activate

# Default: trains XGBoost on MIMIC-IV + SLPDB
# Note: --fresh prompts for confirmation before deleting the DB
python pipeline/pipeline.py --fresh --save-model

# Subsequent runs reuse cached data (no prompt, much faster)
python pipeline/pipeline.py --save-model

# Train on specific SLPDB records only (fast iteration)
python pipeline/pipeline.py --save-model --slpdb-records slp37 slp41 slp66
```

**Saved artefacts (in `project2/`):**

| File | Model | Contents |
|---|---|---|
| `apnea_feature_cols.json` | XGBoost | Ordered feature column list |
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
    needs_review      BOOLEAN      -- true when physiological validator disagrees
);

-- Run this ALTER after creating the table to add validation columns:
ALTER TABLE apnea_results
    ADD COLUMN IF NOT EXISTS validated_ahi             FLOAT,
    ADD COLUMN IF NOT EXISTS validated_ahi_upper       FLOAT,   -- upper bound (includes data-incomplete events)
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
| `ADM*_segments.csv` | Sleep-filtered, feature-enriched segment CSV (includes `spo2Data[0..29]` raw array) |
| `ADM*_sleep_windows.csv` | Sleep detection results (is_sleep, window_id, HR, SDNN) |
| `infer_results_ADM*.csv` | Per-segment XGBoost predictions and features |
| `infer_summary.csv` | One row per admission: AHI, severity, duration, etc. |
| `infer_summary.txt` | Human-readable clinical summary |
| `ADM*_validated.csv` | Inference results with validation columns appended (see below) |

**Validated CSV columns** (added by `apnea_validator.py`):

| Column | Type | Description |
|---|---|---|
| `validation_score` | float | Final 0–100 score for this segment |
| `validation_verdict` | str | `CONFIRMED` / `PROBABLE` / `UNCERTAIN` / `UNCONFIRMED` |
| `method_used` | str | Which primary mechanism won: `ratio`, `hr_drop`, `both`, or `neither` |
| `ratio_score` | float | Score from the HR/RespRate ratio mechanism (0–40) |
| `ratio_baseline` | float | Pre-event HR/RespRate baseline ratio |
| `ratio_event` | float | HR/RespRate ratio at the event segment |
| `ratio_increase_pct` | float | % increase in ratio vs. baseline |
| `hr_drop_score` | float | Score from the HR drop/surge mechanism (0–40) |
| `hr_drop_bpm` | float | Actual HR drop in bpm (NaN if not computable) |
| `hr_dropped` | bool | True if HR fell ≥3 bpm below baseline |
| `hr_surged` | bool | True if HR rose ≥4 bpm in the following segment |
| `spo2_drop_pct` | float | SpO2 drop magnitude in % |
| `spo2_available` | bool | True if SpO2 data existed in the baseline or event window |
| `resp_suppressed` | bool | True if resp amplitude dropped ≥15% below baseline |
| `hr_dip_confirmed` | bool | True if fallback dip criterion confirmed |
| `hr_surge_confirmed` | bool | True if fallback surge criterion confirmed |
| `cluster_bonus_applied` | bool | True if a cluster-level recovery surge bonus was added |
| `data_completeness` | str | `complete` / `insufficient_spo2` / `insufficient_trailing` / `insufficient_both` |

---

## Dependencies

```bash
# Core ML
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
