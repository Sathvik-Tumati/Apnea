# Vital Signs ML Pipeline

A modular machine learning pipeline for clinical prediction across three independent tasks: **Arrhythmia beat classification**, **Sleep Apnea detection**, and **ICU Sepsis early warning**.

---

## Overview

| Module | Task | Data Source | Model | Key Metric |
|--------|------|-------------|-------|------------|
| Arrhythmia | 5-class beat classification (N, VEB, SVEB, F, Q) | MIT-BIH / INCART / SCD Holter CSVs | Random Forest | Accuracy |
| Apnea | Binary apnea detection per 30s segment | MIMIC-IV Waveform DB (PhysioNet, streamed) | Bidirectional LSTM | AUC-ROC |
| Sepsis | Binary sepsis prediction from ICU vitals | Synthetic ICU CSV | Random Forest | Accuracy + AUC-ROC |

---

## Project Structure

```
project/
├── pipeline/
│   └── pipeline.py          # All three module runners + argparse CLI
├── backend.py               # FastAPI server — all REST endpoints
├── db/
│   └── database.py          # SQLite3 schema init, helpers (_j, _uj, log_module)
├── vitals_pipeline.db        # SQLite3 database (auto-created on first run)
├── vitals_pipeline_colab.ipynb  # Standalone Colab evaluation notebook
├── requirements.txt
└── README.md
```

### Data files (place in project root or adjust paths in pipeline.py)
```
MIT-BIH Arrhythmia Database.csv
MIT-BIH Supraventricular Arrhythmia Database.csv
INCART 2-lead Arrhythmia Database.csv
Sudden Cardiac Death Holter Database.csv
sepsis_icu_synthetic.csv
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python pipeline/pipeline.py
```

### 3. Run a single module
```bash
python pipeline/pipeline.py --module arrhythmia
python pipeline/pipeline.py --module apnea
python pipeline/pipeline.py --module sepsis
```

### 4. Start the API server
```bash
uvicorn backend:app --reload --port 8000
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Endpoints — Quick Reference

### Arrhythmia
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/arrhythmia/summary` | Record/prediction counts |
| GET | `/arrhythmia/beat_distribution` | Beat type counts |
| GET | `/arrhythmia/predictions?limit=200` | Sample predictions |
| GET | `/arrhythmia/model_results` | Accuracy + classification report |
| GET | `/arrhythmia/ecg_plot?beat_type=N` | Annotated ECG segment |
| GET | `/arrhythmia/confusion_matrix` | 5×5 confusion matrix |

### Apnea
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/apnea/summary` | Segment/result counts |
| GET | `/apnea/segments?record=&limit=300` | Segment feature rows |
| GET | `/apnea/model_results` | AUC-ROC + report |
| GET | `/apnea/ecg_plot?record=&segment_idx=` | ECG + SpO2 + Resp for one segment |
| GET | `/apnea/spo2_plot?record=` | Full SpO2 series for a record |
| GET | `/apnea/resp_plot?record=` | Full Resp series |
| GET | `/apnea/label_distribution` | Apnea vs Normal counts |

### Sepsis
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sepsis/summary` | Patient/prediction counts + risk rate |
| GET | `/sepsis/predictions?limit=200` | Sample predictions with confidence |
| GET | `/sepsis/model_results` | Accuracy + AUC + report |
| GET | `/sepsis/risk_distribution` | Low/Med/High risk bucket counts |
| GET | `/sepsis/vitals_plot?subject_id=` | 24-hour vital time-series |
| GET | `/sepsis/feature_importance` | Top feature importances |

### Shared
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/summary` | Combined counts across all modules |
| GET | `/pipeline_log` | All pipeline stage logs |
| GET | `/pipeline_log/latest` | Most recent log entry |

---

## Database Schema

Each module uses fully isolated tables. See [Technical Documentation](#) for full schema.

**Arrhythmia:** `arrhythmia_raw`, `arrhythmia_preprocessed`, `arrhythmia_features`, `arrhythmia_results`, `arrhythmia_predictions`, `arrhythmia_ecg_plot`

**Apnea:** `apnea_raw`, `apnea_preprocessed`, `apnea_features`, `apnea_segments`, `apnea_results`, `apnea_ecg_plot`

**Sepsis:** `sepsis_raw`, `sepsis_preprocessed`, `sepsis_features`, `sepsis_results`, `sepsis_predictions`, `sepsis_vitals_plot`

**Shared:** `pipeline_log`

---

## Apnea Labelling — AASM Multi-Signal Composite Rule

Apnea labels are **not** derived from a single signal threshold. A 30-second segment is labelled `Apnea = 1` only if **at least 2 of 4 independent signal conditions** are simultaneously true:

| # | Signal | Condition |
|---|--------|-----------|
| 1 | Resp | Sustained amplitude suppression ≥ 10s (≥90% airflow reduction) |
| 2 | Pleth/SpO2 | ≥3% drop from subject baseline AND min < 94% |
| 3 | ECG/HRV | RMSSD surge >1.5× baseline OR bradycardia >1.2× baseline RR |
| 4 | ABP | MAP variability >1.5× baseline OR BP spike >baseline + 15 mmHg |

This is aligned with AASM polysomnography scoring philosophy and prevents single-signal noise from generating false positives.

---

## Colab Evaluation

Open `vitals_pipeline_colab.ipynb` in Google Colab to run the full pipeline without any local setup. Upload your CSV files when prompted. The notebook runs all three modules independently and prints accuracy, AUC-ROC, classification reports, and saves confusion matrix / feature importance plots.

---

## Requirements

See `requirements.txt`. Key dependencies:

- `scikit-learn` — Random Forest, metrics
- `tensorflow` — Bidirectional LSTM (Apnea)
- `wfdb` — PhysioNet waveform streaming (Apnea)
- `neurokit2` — ECG R-peak detection
- `fastapi` + `uvicorn` — REST API
- `pandas`, `numpy`, `scipy` — data processing

---

## License

For academic and research use. Data sources (MIT-BIH, MIMIC-IV) are subject to their respective PhysioNet data use agreements.
