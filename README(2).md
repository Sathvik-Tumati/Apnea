# Apnea Detection Pipeline — Full Documentation

> **Single-module ML pipeline for sleep apnea detection under real-world wearable constraints.**
> Data source: MIMIC-IV Waveform Database. Model: Bidirectional LSTM.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dependencies & Installation](#3-dependencies--installation)
4. [Configuration Reference](#4-configuration-reference)
5. [How to Run](#5-how-to-run)
6. [Pipeline Architecture — End to End](#6-pipeline-architecture--end-to-end)
7. [Signal Processing Internals](#7-signal-processing-internals)
   - 7.1 [Bandpass Filtering](#71-bandpass-filtering)
   - 7.2 [R-Peak Detection](#72-r-peak-detection)
   - 7.3 [ECG-Derived Respiration (EDR)](#73-ecg-derived-respiration-edr)
   - 7.4 [EDR v3 — Precision Engine](#74-edr-v3--precision-engine-compute_edr_fixedpy)
8. [Apnea Labelling Logic](#8-apnea-labelling-logic)
   - 8.1 [Signal Flags](#81-signal-flags)
   - 8.2 [3-Signal Composite Label](#82-3-signal-composite-label)
9. [Feature Engineering](#9-feature-engineering)
   - 9.1 [HRV Features](#91-hrv-features)
   - 9.2 [Respiratory Features (EDR)](#92-respiratory-features-edr)
   - 9.3 [SpO2 Features](#93-spo2-features)
   - 9.4 [ABP Features](#94-abp-features)
   - 9.5 [Cross-Signal Features](#95-cross-signal-features)
   - 9.6 [EDR Quality Metadata](#96-edr-quality-metadata)
10. [LSTM Model](#10-lstm-model)
11. [Database Schema Overview](#11-database-schema-overview)
12. [Logging](#12-logging)
13. [Wearable Device Constraints Simulated](#13-wearable-device-constraints-simulated)
14. [Evaluation — EDR Benchmarking](#14-evaluation--edr-benchmarking)
15. [Troubleshooting](#15-troubleshooting)
16. [Extending the Pipeline](#16-extending-the-pipeline)
17. [Glossary](#17-glossary)

---

## 1. Project Overview

This pipeline detects **obstructive sleep apnea (OSA)** events from waveform data using only signals realistically available on a wearable device:

- **ECG** at 125 Hz — the primary signal
- **PPG (Pleth/SpO₂)** at 120 Hz — oxygen saturation proxy, **intermittent** (simulating a real pulse oximeter that doesn't read continuously)
- **Respiration** derived from the ECG itself (EDR), not from a chest belt
- **ABP (arterial blood pressure)** at 125 Hz — present in MIMIC-IV, used as a low-weight reference signal

The key design constraint is **no dedicated respiratory sensor**. Breathing is inferred purely from how the ECG signal changes shape with the movement of the thorax — a technique called ECG-Derived Respiration (EDR).

Apnea labels are generated programmatically using a 3-signal composite rule (EDR + SpO₂ + HRV) rather than manual annotation, making the pipeline fully self-contained. A Bidirectional LSTM trained on 10-segment sequences then learns temporal patterns across the extracted features to classify each 30-second window.

---

## 2. Repository Structure

```
project/
│
├── pipeline/
│   ├── pipeline.py              ← Main pipeline (this file)
│   └── compute_edr_fixed.py     ← EDR v3 precision engine (drop-in upgrade)
│
├── CLI/
│   └── db/
│       └── database.py          ← SQLite DB helper (init, insert, fetch)
│
├── pipeline.log                 ← Runtime log (auto-created)
└── README.md                    ← This file
```

> `compute_edr_fixed.py` must sit **in the same directory** as `pipeline.py`. If it's missing, the pipeline falls back to the legacy EDR engine automatically and logs a warning.

---

## 3. Dependencies & Installation

### Required

```bash
pip install numpy pandas scipy scikit-learn tensorflow wfdb neurokit2
```

| Package | Purpose |
|---|---|
| `numpy` | Array math throughout |
| `pandas` | Rolling windows, NaN handling |
| `scipy` | Butterworth filter, Welch PSD, peak detection, resampling |
| `scikit-learn` | StandardScaler, train/test split, classification report |
| `tensorflow` | Bidirectional LSTM model |
| `wfdb` | Streaming MIMIC-IV waveform records from PhysioNet |
| `neurokit2` | High-accuracy R-peak detection (optional but strongly recommended) |

### Optional but recommended

```bash
pip install neurokit2   # Better R-peak detection — significant accuracy gain
```

### Graceful degradation

The pipeline is designed to run even when optional packages are missing:

| Missing package | Consequence |
|---|---|
| `neurokit2` | Falls back to `scipy.signal.find_peaks` for R-peak detection |
| `tensorflow` | Data ingestion and feature extraction still run; LSTM training is skipped |
| `wfdb` | Pipeline cannot fetch MIMIC-IV data and exits early |
| `compute_edr_fixed.py` | Falls back to legacy `_compute_edr` engine |

---

## 4. Configuration Reference

All constants are defined at the top of `pipeline.py` and can be modified directly or overridden via environment variables where noted.

| Constant | Value | Description |
|---|---|---|
| `DATA_DIR` | `../archive2/` | Local data directory (overridable via `DATA_DIR` env var) |
| `MIMIC_URL` | PhysioNet URL | Base URL for streaming MIMIC-IV records |
| `FS_MIMIC` | 320 Hz | Native sampling rate of MIMIC-IV waveforms |
| `FS_ECG` | 125 Hz | Target ECG sampling rate after resampling |
| `FS_PPG` | 120 Hz | Target PPG sampling rate after resampling |
| `FS_RESP` | 4 Hz | Output sampling rate of the EDR signal |
| `SEGMENT_LEN_S` | 30 seconds | Length of each analysis window |
| `N_MIMIC_RECORDS` | 20 | Number of MIMIC-IV records to fetch per run |

> Changing `FS_ECG`, `FS_RESP`, or `SEGMENT_LEN_S` will affect every downstream function. These three constants are tightly coupled — update all affected calls if you change them.

---

## 5. How to Run

### Standard run

```bash
python pipeline/pipeline.py
```

This will:
1. Fetch up to 20 MIMIC-IV records from PhysioNet (requires internet access)
2. Process each record into 30-second segments (up to 10 per record)
3. Store raw signals, preprocessed data, and features in the SQLite database
4. Train a Bidirectional LSTM on the accumulated segments
5. Print AUC-ROC and a classification report to the console and `pipeline.log`

### Fresh run (wipe existing database first)

```bash
python pipeline/pipeline.py --fresh
```

Deletes the existing `.db`, `-wal`, and `-shm` files before starting. Use this when you want a clean slate — for example, after changing feature definitions or resampling rates.

### Programmatic use

```python
from pipeline.pipeline import run_apnea_module
run_apnea_module(n_records=5)   # run on just 5 records
```

---

## 6. Pipeline Architecture — End to End

The pipeline runs in a single sequential pass. Here is the complete flow:

```
MIMIC-IV (PhysioNet)
        │
        ▼
┌───────────────────┐
│  _load_mimic_     │  Streams RECORDS index, filters out layout files,
│  records()        │  returns list of valid waveform paths
└────────┬──────────┘
         │  for each record
         ▼
┌───────────────────┐
│  wfdb.rdrecord()  │  Loads raw multi-channel waveform
│  + Resample       │  ECG → 125 Hz, PPG → 120 Hz, ABP → 125 Hz
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Baseline         │  Scan first N windows, identify clean segments,
│  Computation      │  compute per-patient baseline (SpO₂, RMSSD, MAP, etc.)
└────────┬──────────┘
         │  for each 30-second segment
         ▼
┌────────────────────────────────────────────────────────────┐
│                    Per-Segment Processing                  │
│                                                            │
│  1. _bandpass()         ECG 0.5–40 Hz                      │
│  2. SpO₂ simulation     intermittent reading               │
│  3. _detect_r_peaks()   neurokit2 or scipy fallback        │
│  4. compute_edr_v3()    EDR signal + precision BPM + SNR   │
│     (or _compute_edr()  legacy fallback)                   │
│                                                            │
│  Stage 1: insert_apnea_raw()         raw signals → DB      │
│                                                            │
│  Stage 2: insert_apnea_preprocessed()                      │
│           smoothed SpO₂, smoothed resp, RR stats → DB      │
│                                                            │
│  Stage 3: _extract_apnea_features()  28 features + label   │
│           insert_apnea_features()    → DB                  │
│           insert_apnea_segment()     flat row → DB         │
│           insert_apnea_ecg_plot()    plot data → DB        │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│  LSTM Training    │  Fetch all segments from DB
│  (Stage 4)        │  Scale → sequence (10 steps) → BiLSTM
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Results          │  AUC-ROC, classification report
│  insert_apnea_    │  → DB + pipeline.log
│  results()        │
└───────────────────┘
```

---

## 7. Signal Processing Internals

### 7.1 Bandpass Filtering

**Function:** `_bandpass(signal, fs, lo=0.5, hi=40.0, order=3)`

Applied to the ECG before all downstream processing. Uses a zero-phase Butterworth filter (`scipy.signal.filtfilt`) which introduces no phase distortion — important because R-peak timing must remain accurate.

- Default passband: **0.5–40 Hz** — removes baseline wander (below 0.5 Hz) and high-frequency noise (above 40 Hz) while preserving the QRS complex.
- The upper cutoff is automatically clamped to `nyquist - 0.1` to prevent filter instability.
- `filtfilt` doubles the effective filter order (a 3rd-order design effectively becomes 6th order in terms of roll-off).

### 7.2 R-Peak Detection

**Function:** `_detect_r_peaks(ecg, fs)`

R-peaks are the sharp spikes in the ECG — the "R" in the PQRST complex. Their timing is used for both HRV analysis and as the anchor points for EDR.

**Primary (neurokit2):** `nk.ecg_process()` applies Pan-Tompkins algorithm variants and signal quality checks. This is significantly more robust than a naive peak finder, especially on noisy ICU waveforms.

**Fallback (scipy):** `scipy.signal.find_peaks()` with:
- `distance = fs * 0.4` — minimum 400 ms between peaks (enforces a maximum of 150 BPM)
- `height = std(ecg)` — peak must exceed one standard deviation above the signal mean

If neurokit2 is installed and raises an exception on a particular segment, the scipy fallback is used for that segment only and the exception is logged as a warning.

### 7.3 ECG-Derived Respiration (EDR)

**Function:** `_compute_edr(ecg, r_peaks, fs_ecg, fs_resp=4)`

This is the **legacy EDR engine**, kept as a fallback. It produces a 4 Hz respiratory signal by fusing two independent estimators:

**Engine 1 — QRS-Area Tracking:**
Breathing causes the electrical axis of the heart to rotate, which changes the amplitude of the QRS complex. For each R-peak, the absolute area of the ECG in a ±60 ms window around that peak is computed. This area time-series, sampled at the heartbeat rate, is linearly detrended and interpolated onto a uniform 4 Hz grid.

**Engine 2 — QRS Principal Component Analysis (QRS-PCA):**
All QRS beats (each ±60 ms window) are stacked into a matrix. The first principal component (via SVD) captures the dominant mode of shape variation across beats. Since respiratory modulation is the strongest non-cardiac source of QRS shape change, the first PC score time-series is a clean respiratory proxy.

**Fusion:** Both engine outputs are z-score normalized and their element-wise median is taken. The result is bandpass filtered at 0.1–0.5 Hz to isolate the respiratory frequency band.

**Returns:** `np.ndarray` of shape `(SEGMENT_LEN_S * fs_resp,)` — i.e., 120 samples for a 30-second segment at 4 Hz.

### 7.4 EDR v3 — Precision Engine (`compute_edr_fixed.py`)

**Function:** `compute_edr_v3(ecg, r_peaks, fs_ecg, fs_resp=4, seg_len_s=30)`

The upgraded EDR engine. When `compute_edr_fixed.py` is present alongside `pipeline.py`, this is used instead of `_compute_edr`.

**Key improvements over the legacy engine:**

| Improvement | Detail |
|---|---|
| Internal rate | Runs at **10 Hz** internally (vs 4 Hz) — 3× finer PSD frequency resolution |
| Parabolic interpolation | Sub-bin frequency refinement around the PSD peak gives ~0.3 BPM effective resolution |
| Autocorrelation cross-check | Compares Welch PSD estimate against autocorrelation peak to catch harmonic confusion |
| Harmonic correction | If PSD and AC estimates disagree by >3 BPM, tests whether half/double of the PSD estimate agrees better with AC |
| Adaptive AC blending | Weights autocorrelation more heavily at low respiratory rates (<14 BPM) where harmonics are more likely to fall in-band |
| SNR-based engine selection | Runs both Engine A and Engine B, picks the one with higher in-band SNR |

**Returns:** A 3-tuple:
```python
edr_signal : np.ndarray    # shape (seg_len_s * fs_resp,) — the respiratory waveform
bpm        : float         # precision respiratory rate estimate in breaths per minute
quality    : float         # in-band SNR — values >= 1.5 are considered reliable
```

**How `bpm` and `quality` feed into the pipeline:**

- If `quality >= 1.5`: the precision `bpm` directly becomes `feats["resp_rate_bpm"]`, bypassing the downstream Welch spectral estimate entirely. This is the main accuracy gain.
- If `quality < 1.5`: falls back to computing `resp_rate_bpm` from the EDR signal using Welch PSD (same as the legacy path).
- `quality` is always stored as `feats["edr_snr"]` and `feats["edr_quality_ok"]` (binary flag) — these are saved in the database but are not currently fed to the LSTM. They can be added to `APNEA_FEATURE_COLS` when you want the model to learn to weight EDR-based features by their reliability.

---

## 8. Apnea Labelling Logic

Labels are generated automatically from the signal data — there are no manual annotations. The labelling mimics the AASM (American Academy of Sleep Medicine) 3-signal composite criterion.

### 8.1 Signal Flags

Each flag is a boolean: `True` = apnea-consistent finding detected in this segment.

**`_resp_flag(resp, fs)` — Respiratory suppression flag**

Looks for a sustained period of abnormally low EDR signal amplitude (≥10 seconds). During apnea, airflow stops and the respiratory excursions in the ECG signal flatline.

- Threshold: `mean(resp) - 1.5 * std(resp)`
- A segment is flagged if the longest continuous run of samples below this threshold spans ≥10 seconds (i.e., ≥ `10 * fs` samples).

**`_spo2_flag(pleth, fs, baseline_spo2)` — SpO₂ desaturation flag**

Detects oxygen desaturation events, which typically follow an apnea by 15–30 seconds (the lung oxygen stores buffer the signal).

- SpO₂ is smoothed with a 2-second rolling median to remove motion artifact spikes.
- Flagged if the minimum smoothed SpO₂ drops both:
  - ≥3% below the patient's personal baseline SpO₂, **and**
  - Below the absolute threshold of 94%

The two-condition requirement avoids false positives in patients who normally run low (e.g., COPD patients with baseline SpO₂ of 92%).

**`_hrv_flag(r_peaks, fs, baseline_rmssd, baseline_rr_ms)` — HRV autonomic flag**

Apnea triggers a well-documented autonomic response: vagal withdrawal during the event (HR increases) followed by a parasympathetic rebound after resumption of breathing (HR slows sharply, HRV surges). Either pattern is flagged:

- **HRV surge:** `RMSSD > 1.5 × baseline_RMSSD` — sudden increase in beat-to-beat variability
- **Bradycardia:** `mean RR > 1.2 × baseline_RR` — the heart is beating slower than that patient's normal rate

**`_abp_flag(abp, fs, baseline_map_std, baseline_sbp)` — Haemodynamic flag (reference only)**

Apnea causes surges in intrathoracic pressure, leading to blood pressure variability and systolic spikes. This flag is computed and stored in the database (`abp_flag`) but is **not used in the apnea label** — it is kept for research reference. ABP is not available on wearable devices.

- **Pressure variability:** `std(MAP) > 1.5 × baseline_MAP_std`
- **SBP surge:** `max(SBP) > baseline_SBP + 15 mmHg`

### 8.2 3-Signal Composite Label

**Function:** `label_apnea_segment(resp_flag, spo2_flag, hrv_flag)`

The three primary flags are summed and mapped to a label and a confidence string:

| Flags positive | Label (`true_label`) | Confidence (`label_confidence`) |
|---|---|---|
| 3 of 3 | **1** (Apnea) | `"definite_apnea"` |
| 2 of 3 | **1** (Apnea) | `"probable_apnea"` |
| 1 of 3 | **0** (Normal) | `"possible_hypopnea"` |
| 0 of 3 | **0** (Normal) | `"normal"` |

The threshold of 2+ signals is standard in clinical scoring — requiring multi-signal agreement reduces both false positives (a single noisy signal triggering a label) and false negatives (apnea without a complete physiological signature).

---

## 9. Feature Engineering

**Function:** `_extract_apnea_features(ecg, pleth, resp, abp, r_peaks, baseline, edr_bpm, edr_quality)`

28 features are extracted per segment. The full list is defined in `APNEA_FEATURE_COLS` and is what the LSTM trains on.

### 9.1 HRV Features

Derived from the RR interval series (time in ms between consecutive R-peaks).

| Feature | Description |
|---|---|
| `rr_mean` | Mean RR interval (ms) — inversely related to heart rate |
| `rr_std` | Standard deviation of RR intervals — overall HRV magnitude |
| `rmssd` | Root mean square of successive differences — short-term HRV, vagal tone marker |
| `pnn50` | Proportion of adjacent RR intervals differing by >50 ms — another vagal tone index |
| `mean_hr` | Mean heart rate in BPM, derived from `60000 / rr_mean` |
| `hr_range` | Difference between max and min heart rate in the segment |
| `lf_hf_ratio` | Low-frequency to high-frequency power ratio (requires neurokit2) — sympathovagal balance |

If fewer than 3 R-peaks are detected, all HRV features are set to 0.0.

### 9.2 Respiratory Features (EDR)

| Feature | Description |
|---|---|
| `resp_rate_bpm` | Dominant respiratory frequency converted to breaths per minute. Uses EDR v3 precision estimate if SNR ≥ 1.5, otherwise Welch PSD fallback |
| `resp_rate_variability` | Standard deviation of inter-breath intervals (seconds) — irregular breathing is an apnea marker |
| `flatline_duration_s` | Longest continuous period (seconds) where the EDR signal stays below the suppression threshold — direct apnea duration proxy |
| `resp_amplitude_mean` | Mean absolute EDR signal amplitude — effort proxy |
| `resp_amplitude_std` | Standard deviation of EDR amplitude — effort variability |

### 9.3 SpO₂ Features

| Feature | Description |
|---|---|
| `spo2_mean` | Mean SpO₂ over the segment |
| `spo2_min` | Minimum SpO₂ — the nadir of any desaturation event |
| `spo2_delta_index` | Max minus min SpO₂ — total desaturation range |
| `odi` | Oxygen Desaturation Index — number of times SpO₂ crosses below (baseline − 3%) in this segment |
| `t90` | Fraction of the segment spent below SpO₂ = 90% |
| `spo2_approx_entropy` | Approximate entropy of the SpO₂ signal — low entropy = irregular, pathological fluctuations |

SpO₂ gaps (NaN values from the intermittent simulation) are forward-filled then backward-filled before feature extraction.

### 9.4 ABP Features

These are included as low-weight features. The model may learn to use them, but their in-the-wild applicability is limited since wearables don't have ABP sensors.

| Feature | Description |
|---|---|
| `map_mean` | Mean arterial pressure — 2-second rolling mean of ABP |
| `map_std` | Standard deviation of MAP — haemodynamic stability |
| `map_variability` | Same as `map_std` (alias kept for schema compatibility) |
| `sbp_max` | Maximum systolic pressure — apnea-related surge marker |
| `dbp_min` | Minimum diastolic pressure |
| `pulse_pressure` | `sbp_max - dbp_min` — arterial stiffness proxy |

### 9.5 Cross-Signal Features

These capture **interactions between signals** that a single-signal analysis would miss.

| Feature | Description |
|---|---|
| `resp_spo2_lag_s` | Cross-correlation lag (seconds) between the EDR respiratory signal and the SpO₂ signal. Reflects the oxygen store delay between an apnea event and the resulting desaturation. Clinically expected: 15–30 seconds. |
| `ptt_ms` | Pulse Transit Time — time (ms) from R-peak to the foot of the corresponding ABP pulse. Reflects arterial stiffness and blood pressure changes. Computed per beat and averaged. Only beats where the resulting PTT is in the physiologically plausible 50–500 ms range are included. |
| `ecg_resp_coherence` | Spectral coherence between the RR interval series and the EDR signal in the high-frequency band (0.15–0.4 Hz). Measures Respiratory Sinus Arrhythmia (RSA) — the normal coupling between breathing and heart rate. High coherence = intact autonomic coupling; low coherence may indicate autonomic dysfunction. |

### 9.6 EDR Quality Metadata

These are stored in the database but are **not included in `APNEA_FEATURE_COLS`** and therefore do not feed the LSTM by default.

| Field | Description |
|---|---|
| `edr_quality_ok` | `1` if EDR v3 SNR ≥ 1.5 (reliable segment), `0` otherwise |
| `edr_snr` | Raw EDR in-band SNR value from v3 engine |

To make the LSTM aware of EDR reliability, add `"edr_quality_ok"` and/or `"edr_snr"` to `APNEA_FEATURE_COLS`. This is particularly useful if your dataset has variable ECG quality.

---

## 10. LSTM Model

### Architecture

```
Input: (TIMESTEPS=10, N_FEATURES=28)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dropout (0.2)
    ↓
Dense (32, ReLU)
    ↓
Dense (1, Sigmoid)
    ↓
Output: P(apnea) ∈ [0, 1]
```

### Why Bidirectional LSTM?

Sleep apnea has strong temporal context in both directions. The body shows pre-apnea autonomic changes (rising HR, decreasing HRV) *before* the apnea, and post-apnea recovery patterns *after* it. A bidirectional architecture reads both forward and backward through the 10-segment window, capturing both anticipatory and recovery signatures.

### Sequence construction

Each training sample is a window of 10 consecutive segments. The label for sample `i` is the label of segment `i + 10` (i.e., the model predicts the apnea status of the segment at the end of the window based on the 10 segments leading up to it).

```python
X_seq[i] = X_scaled[i : i+10]    # 10 consecutive feature vectors
y_seq[i] = y_all[i + 10]          # label of the next segment
```

### Training details

| Parameter | Value |
|---|---|
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Metric | AUC |
| Epochs | 10 |
| Batch size | 32 |
| Validation split | 10% of training data |
| Train/test split | 80% / 20% (by segment, not shuffled — preserves temporal order) |

### Output

AUC-ROC and a full classification report (precision, recall, F1 for Normal and Apnea classes) are printed to the console and `pipeline.log`, and stored in the database via `insert_apnea_results()`.

---

## 11. Database Schema Overview

The pipeline writes to an SQLite database managed by `CLI/db/database.py`. Five tables are populated per run:

| Table | Written by | Content |
|---|---|---|
| `apnea_raw` | `insert_apnea_raw()` | Raw ECG, Pleth, Resp (EDR), ABP arrays per segment |
| `apnea_preprocessed` | `insert_apnea_preprocessed()` | Smoothed signals, RR stats, R-peak count |
| `apnea_features` | `insert_apnea_features()` | Full feature dict as JSON |
| `apnea_segments` | `insert_apnea_segment()` | Flat row of all features + label (for LSTM training) |
| `apnea_ecg_plots` | `insert_apnea_ecg_plot()` | ECG + overlay signals for visualisation (max 2 per record) |
| `apnea_results` | `insert_apnea_results()` | AUC-ROC + classification report JSON |
| `module_log` | `log_module()` | Pipeline stage timestamps and status messages |

---

## 12. Logging

The pipeline logs to two sinks simultaneously:

- **Console (`stdout`)** — real-time progress
- **`pipeline.log`** — persistent log file in the working directory

Log format:
```
2025-08-01 14:32:11 | INFO     | __main__ | [APNEA] 83404654: 10 segments processed
```

Key log messages to watch for:

| Message | Meaning |
|---|---|
| `compute_edr_fixed.py not found` | EDR v3 engine not on path — using legacy EDR |
| `nk.ecg_process failed` | neurokit2 raised an exception; scipy fallback used for this segment |
| `[APNEA] Could not load <record>` | MIMIC-IV record fetch failed (network or record format issue) |
| `[APNEA] No segments processed` | No records yielded valid data — pipeline aborts before training |
| `[APNEA] Only one class in test set` | All test segments were the same class — AUC cannot be computed |
| `[APNEA] AUC-ROC: X.XXXX` | Training completed successfully |

---

## 13. Wearable Device Constraints Simulated

The pipeline deliberately models the limitations of a real wearable device:

**1. No chest respiratory belt**
Respiration is derived entirely from the ECG (EDR). This is noisier than a dedicated sensor and is the main source of error in the pipeline. The EDR v3 engine was specifically developed to improve this.

**2. Intermittent SpO₂**
Real pulse oximeters don't take continuous readings — they sample every 3–5 minutes to conserve battery. This is simulated by providing a real PPG segment every 6–10 segments (randomly chosen), and holding the last known value in between:

```python
take_reading = (i % np.random.randint(6, 11)) == 0
if take_reading:
    pleth_seg = pleth_full[s_ppg:e_ppg]
    last_spo2_val = float(np.mean(pleth_seg))
else:
    pleth_seg = np.full(samples_per_seg_ppg, last_spo2_val)
```

**3. Fixed low sampling rates**
ECG is capped at 125 Hz (typical BLE ECG chip rate) and PPG at 120 Hz, even though MIMIC-IV is natively at 320 Hz. This tests whether the model can work with reduced-resolution data.

**4. ABP treated as low-weight reference**
ABP is used in baseline computation and feature extraction but is explicitly called out as non-wearable in the code comments. The LSTM must learn to weight it accordingly.

---

## 14. Evaluation — EDR Benchmarking

A separate evaluation script `evaluate_edr.py` benchmarks the legacy `_compute_edr` against `compute_edr_v3` on a fixed set of MIMIC-IV records with verified Resp channels. Run it independently:

```bash
python evaluate_edr.py
```

It computes per-segment Mean Absolute Error (MAE) in BPM against the ground-truth Resp channel, and prints a side-by-side comparison table. Target performance for v3 is MAE 3.0–4.0 BPM (vs ~5.18 BPM for the legacy engine).

---

## 15. Troubleshooting

**`ImportError: No module named 'wfdb'`**
```bash
pip install wfdb
```

**`Failed to load MIMIC records` / network timeout**
The pipeline streams records from PhysioNet. Ensure you have a stable internet connection. PhysioNet may also rate-limit frequent requests. Try reducing `N_MIMIC_RECORDS` to 5 for a quick test run.

**`No segments processed — aborting`**
This means every record either failed to download or was missing the required `II` (ECG) and `Pleth` channels. Check `pipeline.log` for per-record warnings. MIMIC-IV records vary in which signals they contain.

**`Only one class in test set — AUC not computed`**
With few records or segments, it's possible all test segments end up labelled the same class. Increase `N_MIMIC_RECORDS` or check whether your signal quality thresholds in the baseline computation are too aggressive.

**EDR quality is consistently low (`edr_snr` < 1.5)**
This can happen with very noisy ECG segments. Check:
- Whether neurokit2 is installed (better R-peak detection → better EDR)
- Whether the records you're using have clean Lead II signals
- Try lowering the SNR threshold in `_extract_apnea_features` from 1.5 to 1.0 as an experiment

**`compute_edr_fixed.py not found` warning**
Copy `compute_edr_fixed.py` into the same directory as `pipeline.py`. The files must be co-located because the import path is relative.

---

## 16. Extending the Pipeline

### Adding a new feature

1. Compute the value inside `_extract_apnea_features()` and assign it to `feats["your_feature_name"]`.
2. Add `"your_feature_name"` to the `APNEA_FEATURE_COLS` list.
3. Make sure your database schema in `database.py` has a corresponding column (or that the segment is stored as JSON, which handles new keys automatically).

### Adding EDR quality to the LSTM

Open `pipeline.py` and add to `APNEA_FEATURE_COLS`:
```python
"edr_quality_ok", "edr_snr"
```
These are already being computed and stored — they just need to be included in the model's input.

### Using a different model

Replace the Keras Sequential block in `run_apnea_module()`. The `X_seq` and `y_seq` arrays follow standard sklearn/Keras conventions. For example, to use an XGBoost classifier on flattened sequences:
```python
from xgboost import XGBClassifier
X_flat = X_seq.reshape(len(X_seq), -1)
model = XGBClassifier().fit(X_flat[:split], y_seq[:split])
```

### Adding a new signal type

1. Extract the new signal from `rec.p_signal` using the `sig_map` dictionary.
2. Resample it to a consistent rate.
3. Slice it per-segment alongside ECG/PPG.
4. Add processing in `_extract_apnea_features()` and update `APNEA_FEATURE_COLS`.

---

## 17. Glossary

| Term | Definition |
|---|---|
| **AASM** | American Academy of Sleep Medicine — the clinical body that defines apnea scoring rules |
| **ABP** | Arterial Blood Pressure — invasive continuous blood pressure measurement |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic curve — classification performance metric; 0.5 = random, 1.0 = perfect |
| **BiLSTM** | Bidirectional Long Short-Term Memory — a recurrent neural network that processes sequences both forward and backward |
| **ECG / EKG** | Electrocardiogram — electrical signal of the heart |
| **EDR** | ECG-Derived Respiration — estimating the breathing signal from the ECG without a dedicated respiratory sensor |
| **HRV** | Heart Rate Variability — fluctuations in the time between heartbeats; a marker of autonomic nervous system activity |
| **MAP** | Mean Arterial Pressure — time-averaged arterial blood pressure |
| **MIMIC-IV** | Medical Information Mart for Intensive Care IV — a large ICU waveform database |
| **ODI** | Oxygen Desaturation Index — number of desaturation events per unit time |
| **OSA** | Obstructive Sleep Apnea — cessation of breathing due to airway obstruction during sleep |
| **PCA** | Principal Component Analysis — linear dimensionality reduction via singular value decomposition |
| **PLI** | Powerline interference (50/60 Hz) |
| **PPG / Pleth** | Photoplethysmography — optical blood volume pulse signal, used as a SpO₂ proxy |
| **PSD** | Power Spectral Density — distribution of signal power across frequencies |
| **PTT** | Pulse Transit Time — time for a pressure wave to travel from heart to a peripheral point; inversely related to blood pressure |
| **QRS complex** | The sharp spike in the ECG representing ventricular depolarisation |
| **RMSSD** | Root Mean Square of Successive Differences — short-term HRV metric |
| **RSA** | Respiratory Sinus Arrhythmia — the normal modulation of heart rate by breathing |
| **SNR** | Signal-to-Noise Ratio — ratio of in-band signal power to out-of-band noise power |
| **SpO₂** | Peripheral oxygen saturation — percentage of haemoglobin carrying oxygen |
| **SVD** | Singular Value Decomposition — matrix factorisation used in PCA |
| **T90** | Time below 90% SpO₂ — a standard clinical hypoxia severity measure |
| **Welch PSD** | A method for estimating power spectral density by averaging periodograms of overlapping signal segments |
