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
pip install tensorflow keras numpy pandas scipy scikit-learn joblib \
            neurokit2 wfdb pymongo supabase python-dotenv
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# MongoDB (required for live inference)
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGO_DB=your_database_name

# Supabase (required for --write-supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Model paths (defaults — override only if non-standard)
MODEL_PATH=apnea_model.keras
SCALER_PATH=apnea_scaler.pkl
THRESHOLD=0.45
OUTPUT_DIR=infer_output/
```

---

## 2. Train the Model

Downloads MIMIC-IV and SLPDB automatically, runs feature extraction, trains the BiLSTM, and saves model artefacts.

```bash
source venv/bin/activate

# First run — fresh DB, downloads everything, trains
python pipeline/pipeline.py --fresh --save-model

# Subsequent runs — reuse cached MIMIC/SLPDB data
python pipeline/pipeline.py --save-model

# Fast iteration — specific SLPDB records only
python pipeline/pipeline.py --save-model --slpdb-records slp37 slp41 slp66 slp48

# Skip SLPDB entirely (MIMIC only, not recommended — lower sensitivity)
python pipeline/pipeline.py --save-model --no-slpdb
```

**Saved artefacts (written to `project2/`):**

| File | Contents |
|---|---|
| `apnea_model.keras` | Full trained BiLSTM model |
| `apnea_best.keras` | Best checkpoint (highest val AUC) |
| `apnea_scaler.pkl` | Fitted StandardScaler (30 features) |
| `apnea_feature_cols.json` | Ordered list of the 30 feature columns |
| `apnea_thresholds.json` | Optimal thresholds `{global, mimic, slpdb}` |

**Latest training results (June 2026):**

| Dataset | AUC | F1 | Sensitivity | Precision |
|---|---|---|---|---|
| **Overall** | **0.912** | 0.772 | — | — |
| MIMIC (ICU) | 0.917 | 0.714 | 71.4% | 71.4% |
| SLPDB (Sleep lab) | 0.906 | 0.784 | 92.3% | 68.1% |

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
6. Runs BiLSTM inference (HRV extraction → EDR → feature scaling → model)
7. Computes AHI proxy from scored segment count / sleep duration
8. Upserts results to Supabase `apnea_results` table

### Commands

```bash
# Single admission — full pipeline
python automation/mongo_infer.py --admission ADM1819906487 --write-supabase

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
    apnea_pct         FLOAT,
    total_segments    INT,
    scored_segments   INT,
    n_apnea           INT,
    duration_min      FLOAT,
    model_threshold   FLOAT
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
| `TIMESTEPS` | 10 | Sequence length for BiLSTM |
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
| `No MIMIC records fetched` | PhysioNet network timeout | Retry; PhysioNet is rate-limited |
| `MongoDB connection failed` | Wrong URI / env vars | Check `.env` — `MONGO_URI` and `MONGO_DB` |
| `[SLEEP] Only N sleep segments` | Short recording or daytime admission | Add `--no-sleep-filter` to use full recording |
| `[AHI] Timestamp-derived duration looks wrong` | Sleep filter trimmed timestamps; AHI falls back to segment count | Expected — AHI still computed from segment count |
| `Only one class in test set` | All-normal or all-apnea training split | Add more diverse SLPDB records |
