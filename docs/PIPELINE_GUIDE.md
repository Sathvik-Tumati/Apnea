# Complete Pipeline Guide

This document walks through every step needed to run the apnea detection pipeline —
from environment setup and training, to live MongoDB inference with sleep filtering.

> [!IMPORTANT]
> **Always run commands from the `project2/` root**, not from inside `pipeline/` or `automation/`. Import paths depend on this.

---

## 1. Prerequisites

### Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow keras numpy pandas scipy scikit-learn joblib xgboost \
            neurokit2 wfdb pymongo supabase python-dotenv
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# MongoDB (required)
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=your_database_name

# Supabase (required for --write-supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Primary model (required for inference)
MODEL_PATH=apnea_model.keras
SCALER_PATH=apnea_scaler.pkl
THRESHOLD=0.45

# Secondary BiLSTM model (optional — enables consensus mode)
BILSTM_MODEL_PATH=apnea_model.keras
BILSTM_SCALER_PATH=apnea_scaler.pkl

# Output
OUTPUT_DIR=infer_output/
```

---

## 2. Train the Model

Downloads MIMIC-IV and SLPDB automatically, runs feature extraction, trains the BiLSTM, and saves model artefacts.

```bash
source venv/bin/activate

# First run — fresh DB, downloads everything, trains BOTH models
python pipeline/pipeline.py --fresh --save-model

# Subsequent runs — reuse cached MIMIC/SLPDB data
python pipeline/pipeline.py --save-model

# Train only one model (BiLSTM or XGBoost)
python pipeline/pipeline.py --save-model --bilstm-only
python pipeline/pipeline.py --save-model --xgb-only

# Fast iteration — specific SLPDB records only
python pipeline/pipeline.py --save-model --slpdb-records slp37 slp41 slp66 slp48

# Skip SLPDB entirely (MIMIC only — not recommended, lower sensitivity)
python pipeline/pipeline.py --save-model --no-slpdb
```

> [!NOTE]
> When running `--xgb-only`, XGBoost reads pre-built segments from `vitals_pipeline.db` populated by a prior BiLSTM run. If the DB is empty (e.g. after `--fresh`), run without `--xgb-only` first so BiLSTM populates the DB.

**Saved artefacts (written to `project2/`):**

| File | Model | Contents |
|---|---|---|
| `apnea_model.keras` | BiLSTM | Full trained BiLSTM model |
| `apnea_best.keras` | BiLSTM | Best checkpoint (highest val AUC) |
| `apnea_scaler.pkl` | BiLSTM | Fitted StandardScaler (30 features) |
| `apnea_feature_cols.json` | BiLSTM | Ordered list of the 30 feature columns |
| `apnea_thresholds.json` | BiLSTM | Optimal thresholds `{global, mimic, slpdb}` |
| `apnea_model_xgb_seq.pkl` | XGBoost | Trained XGBClassifier (sequences flattened T×F) |
| `apnea_scaler_tree.pkl` | XGBoost | Fitted StandardScaler for XGBoost input |

**Latest training results (June 2026):**

| Dataset | BiLSTM AUC | BiLSTM F1 | XGBoost AUC |
|---|---|---|---|
| **Overall** | **0.912** | 0.772 | — |
| MIMIC (ICU) | 0.917 | 0.714 | — |
| SLPDB (Sleep lab) | 0.906 | 0.784 | — |

---

## 3. Live MongoDB Inference (Primary Use-Case)

Use this path when ECG and SpO2 data lives in a MongoDB database.

### How It Works

Each run performs these steps automatically:
1. Opens SSH tunnel → MongoDB
2. Pulls ECG documents and assembles signal
3. Pulls SpO2 values (1 Hz, device-computed)
4. **ECG sleep detection** — scores each 30-second epoch using IST time gate + HR/SDNN heuristics, isolates sleep-only windows
5. Builds 30-second segment CSV (SpO2 aligned by timestamp)
6. **Primary inference (XGBoost)** — HRV extraction → feature scaling → model → AHI proxy
7. **Secondary inference (BiLSTM, optional)** — same CSV, separate model → consensus check
8. If both models disagree on normal/abnormal classification → `needs_review = true` in Supabase
9. Upserts merged results to Supabase `apnea_results` table

### Commands

```bash
# Single admission — full pipeline (XGBoost only by default)
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase

# With BiLSTM consensus (both models run, disagreements flagged)
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase \
    --model-bilstm apnea_model.keras --scaler-bilstm apnea_scaler.pkl

# Or set BILSTM_MODEL_PATH + BILSTM_SCALER_PATH in .env and omit the flags

# All admissions from the last 24 hours
python automation/mongo_infer.py --since 24h --write-supabase

# Date range
python automation/mongo_infer.py --from 2026-06-01 --to 2026-06-15 --write-supabase

# Dry run — extract and sleep-filter CSV only, no inference
python automation/mongo_infer.py --admission ADM1819906487 --dry-run

# Skip sleep filtering (e.g. short recording, ICU patient not sleeping overnight)
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase --no-sleep-filter

# Re-process an admission that already has a Supabase result
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase --reprocess
```

### Output Files

Under `infer_output/<admissionId>/`:

| File | Contents |
|---|---|
| `ADM*_segments.csv` | Sleep-filtered, SpO2-enriched segment CSV fed to the model |
| `ADM*_sleep_windows.csv` | Per-segment sleep scores (is_sleep, window_id, HR, SDNN, hour_ist) |
| `infer_results_ADM*.csv` | Per-segment predictions: apnea_prob, apnea_pred, features, quality flags |
| `infer_summary.csv` | One row: AHI, severity, duration_min, n_apnea, scored_segments, mean_prob |
| `infer_summary.txt` | Human-readable clinical summary block |

### Supabase Schema

Run once in the SQL editor:

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
    -- Consensus fields (populated when BiLSTM model path is provided)
    ahi_bilstm        FLOAT,
    severity_bilstm   TEXT,
    models_agree      BOOLEAN,
    needs_review      BOOLEAN      -- true when XGBoost and BiLSTM disagree
);
```

---

## 4. ECG Sleep Detection

The `automation/ecg_sleep_filter.py` module runs automatically inside `mongo_infer.py`. It can also be run standalone on any segment CSV produced by `--dry-run`:

```bash
# Inspect sleep windows for an already-extracted CSV
python automation/ecg_sleep_filter.py \
    --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
    --stats

# Save sleep windows CSV + diagnostic plot
python automation/ecg_sleep_filter.py \
    --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
    --out infer_output/ADM1819906487/sleep_windows.csv \
    --plot

# Disable IST time gate (useful for short test recordings)
python automation/ecg_sleep_filter.py \
    --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
    --no-time-gate --stats
```

See [sleep_detection.md](sleep_detection.md) for full algorithm details.

---

## 5. EDF / Local File Inference

Use this path for offline `.edf` recordings from consumer wearables.

```bash
# Convert EDF → segment CSV
python pipeline/edf_to_pipeline.py \
    --input path/to/recording.edf \
    --mode csv \
    --out-dir pipeline/converted/

# Sleep filter (EDF-based, multi-signal)
python pipeline/sleep_filter.py \
    --detect --filter \
    --input path/to/recording.edf \
    --csvs pipeline/converted/ \
    --out-dir pipeline/converted/sleep_only/

# Run inference
python pipeline/edf_test_loader.py \
    --data pipeline/converted/sleep_only/ \
    --mode infer \
    --model apnea_model.keras \
    --scaler apnea_scaler.pkl \
    --features apnea_feature_cols.json
```

---

## 6. Configuration Reference

### `pipeline/modules/config.py` — Core Constants

| Constant | Default | Description |
|---|---|---|
| `FS_ECG` | 125 Hz | ECG target sample rate |
| `FS_RESP` | 4 Hz | EDR / respiratory signal rate |
| `SEGMENT_LEN_S` | 30 s | Segment window length |
| `TIMESTEPS` | 10 | Sequence length for BiLSTM and XGBoost |
| `N_MIMIC_RECORDS` | 60 | MIMIC-IV records to download |

### `automation/mongo_infer.py` — Pipeline Constants

| Constant | Default | Description |
|---|---|---|
| `FS_ECG` | 125 Hz | Expected ECG sample rate |
| `FS_SPO2` | 1 Hz | Device-computed SpO2 rate |
| `SEGMENT_LEN_S` | 30 s | Segment window |
| `MIN_SEGMENTS` | 11 | Minimum segments for inference |
| `MIN_DURATION_MINUTES` | 30 | Skip recordings shorter than this |
| `COMPLETION_GAP_HOURS` | 2 | Hours without new packets → recording done |
| `BILSTM_MODEL_PATH` | `apnea_model.keras` | Secondary model path (env: `BILSTM_MODEL_PATH`) |
| `BILSTM_SCALER_PATH` | `apnea_scaler.pkl` | Secondary scaler path (env: `BILSTM_SCALER_PATH`) |

### `automation/ecg_sleep_filter.py` — Sleep Detection Constants

| Constant | Default | Description |
|---|---|---|
| `SLEEP_HOUR_START` | 21 (9pm IST) | Start of sleep time gate |
| `SLEEP_HOUR_END` | 9 (9am IST) | End of sleep time gate |
| `HR_PERCENTILE_THRESHOLD` | 45 | HR below p45 of nighttime = sleep candidate |
| `SDNN_PERCENTILE_THRESHOLD` | 55 | SDNN above p55 = sleep candidate |
| `MIN_SLEEP_EPOCHS` | 40 | ≥ 40 × 30s = 20 min minimum sleep bout |
| `MAX_WAKE_GAP_EPOCHS` | 6 | Bridge up to 3-min wake gaps within a bout |

---

## 7. Common Issues

| Error | Cause | Fix |
|---|---|---|
| `command not found: python` | System Python not activated | `source venv/bin/activate` |
| `Could not locate class 'GatherFlags'` | `pipeline.modules.model` not imported before `load_model` | Run from `project2/` root, not `pipeline/` |
| `NameError: HAS_WFDB / HAS_TF / _SPO2_IDXS` | Underscore variables not re-exported by `import *` | Already fixed — explicit imports added |
| `wfdb not installed` | Missing dependency | `pip install wfdb` |
| `xgboost not installed` | Missing dependency | `pip install xgboost` |
| `No MIMIC records fetched` | PhysioNet network timeout | Retry; PhysioNet is rate-limited |
| `[XGB] No segments in DB` | Running `--xgb-only` on empty DB | Run BiLSTM first (without `--xgb-only`) to populate `vitals_pipeline.db` |
| `MongoDB connection failed` | Wrong URI / env vars | Check `.env` — `MONGO_URI` and `MONGO_DB` |
| `[SLEEP] Only N sleep segments` | Short recording or daytime admission | Add `--no-sleep-filter` to use full recording |
| `[AHI] Timestamp-derived duration looks wrong` | Sleep filter trimmed timestamps; AHI falls back to segment count | Expected — AHI still computed correctly from segment count |
| `[INFER] Primary model failed` | Model or scaler file missing | Check `MODEL_PATH` / `SCALER_PATH` in `.env` |
| `needs_review = true` in Supabase | XGBoost and BiLSTM disagree on normal/abnormal | Expected behaviour — flag for clinical review |
| `Only one class in test set` | All-normal or all-apnea training split | Add more diverse SLPDB records |
