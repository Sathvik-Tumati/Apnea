# Apnea Detection Pipeline

[![Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Sathvik-Tumati/Apnea)

A multi-source, modality-aware sleep apnea detection system built on a Bidirectional LSTM (BiLSTM). The pipeline trains on two clinical datasets (MIMIC-IV and SLPDB), extracts physiological features from ECG and auxiliary signals, and runs inference on converted wearable EDF recordings.

---

## 📚 Documentation

For deep dives into the technical architecture and logic, see the `docs/` directory:
- [**Modality-Aware BiLSTM Architecture**](docs/architecture.md): How we handle missing signals (like SpO2 or ABP) using Modality Flags and Modality Dropout.
- [**Feature Engineering**](docs/feature_engineering.md): Details on the 30 features extracted per segment, including the dual-engine ECG-Derived Respiration (EDR).
- [**Wearable Inference Guide**](docs/wearable_inference.md): A complete guide on converting `.edf` files, applying sleep filters, and running inference.

---

## Repository Layout

```
project2/
├── pipeline/
│   ├── modules/                # Core modular components
│   │   ├── config.py           # Constants, thresholds, and logging
│   │   ├── features.py         # Signal processing and feature extraction
│   │   ├── ingest_mimic.py     # MIMIC-IV streaming and pseudo-labelling
│   │   ├── ingest_slpdb.py     # SLPDB streaming and annotation parsing
│   │   ├── model.py            # Modality-Aware BiLSTM architecture
│   │   ├── train.py            # Dataset building and training loop
│   │   └── evaluate.py         # Metric calculation and threshold sweeping
│   ├── pipeline.py             # Orchestrator script
│   ├── edf_to_pipeline.py      # Converts raw EDF wearable files → CSV/JSON
│   ├── edf_test_loader.py      # Inference test harness
│   ├── sleep_filter.py         # Detects sleep windows from wearable signals
│   └── utils/                  # Evaluation and diagnostics utilities
├── docs/                       # Technical documentation
├── apnea_model.keras           # Saved trained model
├── apnea_scaler.pkl            # Fitted StandardScaler
├── apnea_feature_cols.json     # Ordered list of feature column names
├── apnea_best.keras            # Best checkpoint saved during training
└── apnea_thresholds.json       # Optimal classification thresholds per data source
```

> **Important:** Always run commands from `project2/` (the root), not from inside `pipeline/`. The import paths and file-save locations depend on this.

---

## Quick Start

```bash
cd /path/to/project2

# 1. Convert your EDF recordings to CSVs
python pipeline/edf_to_pipeline.py --input pipeline/my_recording.edf --mode csv

# 2. Filter to sleep-only segments
python pipeline/sleep_filter.py --detect --filter \
  --input pipeline/my_recording.edf \
  --csvs pipeline/converted/ \
  --out-dir pipeline/converted/sleep_only/

# 3. Train the model (downloads MIMIC + SLPDB automatically)
source pipeline/venv/bin/activate
python pipeline/pipeline.py --save-model --fresh

# 4. Run inference on your wearable data
python pipeline/edf_test_loader.py \
  --data pipeline/converted/sleep_only/ \
  --mode infer \
  --model apnea_model.keras \
  --scaler apnea_scaler.pkl
```

---

## Pipeline Tools

### 1. `edf_to_pipeline.py` — EDF Converter
Converts raw EDF/BDF wearable recordings into 30-second segments stored as CSV or JSON files. Identifies channels by name heuristics, resamples to standard rates (ECG: 125Hz, Resp: 4Hz), and extracts non-overlapping windows.

### 2. `sleep_filter.py` — Sleep Window Detector
Filters converted segment CSVs to keep only segments that fall within detected sleep windows, reducing noise from awake periods. Uses heart rate and movement heuristics.

### 3. `pipeline.py` — Core Orchestrator
Handles data ingestion from MIMIC-IV and SLPDB, builds combined datasets, applies modality dropout, and trains the BiLSTM model. After training, it saves model artifacts and optimal thresholds.

### 4. `edf_test_loader.py` — Inference Harness
A standalone script for running the trained model against converted wearable data. It loads 30-second segments, detects available channels, routes to the appropriate feature extractor, and generates clinical summaries.

---

## Caching & Database

Both data sources are cached locally after the first download:
- SLPDB: `~/.cache/slpdb/`
- MIMIC-IV: `~/.cache/mimic4wdb/`

All ingested segments, features, preprocessed signals, ECG plots, and results are persisted to a SQLite database (`vitals_pipeline.db`) via `CLI/db/database.py`.

---

## Dependencies

```
tensorflow >= 2.x
keras
numpy
pandas
scipy
scikit-learn
wfdb          # PhysioNet data access
pyedflib      # EDF file reading (or: mne)
neurokit2     # R-peak detection (optional but recommended)
```

Install into the virtual environment:

```bash
source venv/bin/activate
pip install tensorflow keras numpy pandas scipy scikit-learn wfdb pyedflib neurokit2
```
