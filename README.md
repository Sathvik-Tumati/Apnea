# Sleep Apnea Detection Pipeline

> **A single-module ML pipeline for sleep apnea detection under real-world wearable constraints.**
> Data: MIMIC-IV Waveform Database · Model: Regularised Bidirectional LSTM with Focal Loss

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dependencies & Installation](#3-dependencies--installation)
4. [Configuration Reference](#4-configuration-reference)
5. [How to Run](#5-how-to-run)
6. [Pipeline Architecture — End to End](#6-pipeline-architecture--end-to-end)
7. [Signal Processing](#7-signal-processing)
   - 7.1 [Bandpass Filtering](#71-bandpass-filtering)
   - 7.2 [R-Peak Detection](#72-r-peak-detection)
   - 7.3 [ECG-Derived Respiration (EDR)](#73-ecg-derived-respiration-edr)
   - 7.4 [EDR v3 — Precision Engine](#74-edr-v3--precision-engine)
   - 7.5 [Ground-Truth Resp Channel](#75-ground-truth-resp-channel)
8. [Apnea Labelling Logic](#8-apnea-labelling-logic)
   - 8.1 [Signal Flags](#81-signal-flags)
   - 8.2 [AASM Composite Label](#82-aasm-composite-label)
   - 8.3 [Label Source Tracking](#83-label-source-tracking)
9. [Feature Engineering](#9-feature-engineering)
   - 9.1 [HRV Features](#91-hrv-features)
   - 9.2 [Respiratory Features](#92-respiratory-features)
   - 9.3 [SpO2 Features](#93-spo2-features)
   - 9.4 [ABP Features](#94-abp-features)
   - 9.5 [Cross-Signal Features](#95-cross-signal-features)
10. [LSTM Model](#10-lstm-model)
    - 10.1 [Architecture](#101-architecture)
    - 10.2 [Focal Loss](#102-focal-loss)
    - 10.3 [Training Strategy](#103-training-strategy)
    - 10.4 [Threshold Tuning](#104-threshold-tuning)
    - 10.5 [Evaluation](#105-evaluation)
11. [Run ID Isolation](#11-run-id-isolation)
12. [Database Schema](#12-database-schema)
13. [Logging](#13-logging)
14. [Wearable Constraints Simulated](#14-wearable-constraints-simulated)
15. [Troubleshooting](#15-troubleshooting)
16. [Glossary](#16-glossary)

---

## 1. Project Overview

This pipeline detects **obstructive sleep apnea (OSA)** events from physiological waveforms using only signals a wearable device can realistically provide:

- **ECG** at 125 Hz — the primary signal, always available
- **PPG/SpO₂** at 120 Hz — simulated as intermittent (wearable pulse oximeters don't read continuously)
- **Respiration** derived from the ECG itself (EDR) — no chest belt required at inference time
- **ABP** at 125 Hz — available in MIMIC-IV hospital data, used as a supplementary reference

The core challenge is that a wearable has no dedicated respiratory sensor. Breathing must be inferred from subtle shape changes in the ECG signal caused by thoracic movement — a technique called **ECG-Derived Respiration (EDR)**.

**Key design decision:** During training, the ground-truth `Resp` channel from the MIMIC-IV recording (an actual impedance pneumography chest belt) is used to generate reliable apnea labels and respiratory features. The EDR signal is what the model will see at inference time on a real wearable — so the model is trained to work with EDR but labelled using ground truth. This separation is critical to avoid circular labelling.

---

## 2. Repository Structure

```
project2/
│
├── pipeline/
│   ├── pipeline.py              ← Main pipeline (this file)
│   ├── compute_edr_fixed.py     ← EDR v3 precision engine (optional upgrade)
│   ├── evaluate_edr.py          ← EDR benchmarking script
│   └── find_good_records.py     ← MIMIC record discovery utility
│
├── CLI/
│   ├── backend.py               ← FastAPI backend (reads from DB)
│   └── db/
│       └── database.py          ← SQLite schema and all CRUD helpers
│
├── vitals_pipeline.db           ← SQLite database (auto-created on first run)
├── pipeline.log                 ← Runtime log (auto-created)
└── README.md                    ← This file
```

> `compute_edr_fixed.py` must be in the **same directory** as `pipeline.py`. If absent, the pipeline falls back to the legacy EDR engine and logs a warning.

---

## 3. Dependencies & Installation

### Required packages

```bash
pip install numpy pandas scipy scikit-learn tensorflow wfdb neurokit2
```

| Package | Purpose |
|---|---|
| `numpy` | Array math throughout |
| `pandas` | Rolling windows, NaN handling, dataframes |
| `scipy` | Butterworth filter, Welch PSD, peak detection, resampling |
| `scikit-learn` | StandardScaler, stratified split, classification metrics |
| `tensorflow` | Bidirectional LSTM model |
| `wfdb` | Streaming MIMIC-IV waveform records from PhysioNet |
| `neurokit2` | High-accuracy R-peak detection (strongly recommended) |

### Graceful degradation

The pipeline continues running even if optional packages are missing:

| Missing package | Effect |
|---|---|
| `neurokit2` | Falls back to `scipy.signal.find_peaks` for R-peak detection — noticeably lower accuracy |
| `tensorflow` | Ingestion and feature extraction still run; LSTM training is skipped |
| `compute_edr_fixed.py` | Falls back to legacy `_compute_edr` dual-engine fusion |
| `wfdb` | Pipeline cannot fetch MIMIC-IV data and exits immediately |

---

## 4. Configuration Reference

All constants are defined at the top of `pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `MIMIC_URL` | PhysioNet URL | Base URL for streaming MIMIC-IV waveform records |
| `FS_ECG` | 125 Hz | Target ECG sampling rate after resampling |
| `FS_PPG` | 120 Hz | Target PPG sampling rate after resampling |
| `FS_RESP` | 4 Hz | Output sampling rate of the EDR / GT Resp signal |
| `SEGMENT_LEN_S` | 30 seconds | Length of each analysis window |
| `N_MIMIC_RECORDS` | 60 | Number of MIMIC-IV records to fetch per run |

> `FS_ECG`, `FS_RESP`, and `SEGMENT_LEN_S` are tightly coupled throughout the codebase. If you change any of them, search for all three and update every dependent calculation.

---

## 5. How to Run

### Standard run

```bash
cd /path/to/project2
python pipeline/pipeline.py
```

### Fresh run — wipe existing database first

```bash
python pipeline/pipeline.py --fresh
```

Use `--fresh` whenever you change feature definitions, the labelling logic, the schema, or the resampling rates. Stale rows from previous runs will otherwise mix with new ones.

### What happens during a run

1. Fetches up to `N_MIMIC_RECORDS` waveform records from PhysioNet over HTTPS
2. Processes each record into 30-second segments (up to 10 per record)
3. Stores raw signals, preprocessed data, features, and labels in SQLite
4. Trains a Bidirectional LSTM on the current run's GT-labelled segments only
5. Prints AUC-ROC with 95% bootstrap CI and a classification report

---

## 6. Pipeline Architecture — End to End

```
MIMIC-IV (PhysioNet HTTPS)
          │
          ▼
┌──────────────────────┐
│  _load_mimic_records │  Fetches RECORDS index, filters layout files,
│                      │  returns list of valid waveform paths
└──────────┬───────────┘
           │  for each record
           ▼
┌──────────────────────┐
│  wfdb.rdrecord()     │  Loads raw multi-channel waveform (up to 96000 samples)
│  + NaN fill          │  ECG→125Hz  PPG→120Hz  ABP→125Hz  Resp→4Hz
│  + Resample          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Baseline            │  Scans all segments, identifies clean windows
│  Computation         │  (no resp suppression, SpO₂ > 90%)
│                      │  Computes: baseline SpO₂, RMSSD, RR, MAP, SBP
└──────────┬───────────┘
           │  for each 30-second segment (up to 10)
           ▼
┌──────────────────────────────────────────────────────────┐
│                   Per-Segment Processing                 │
│                                                          │
│  1. _bandpass()         ECG 0.5–40 Hz bandpass           │
│  2. SpO₂ simulation     intermittent reading             │
│  3. _detect_r_peaks()   neurokit2 or scipy fallback      │
│  4. compute_edr_v3()    EDR signal + precision BPM + SNR │
│     (or _compute_edr()  legacy fallback)                 │
│  5. resp_gt_seg         GT Resp slice at FS_RESP Hz      │
│                                                          │
│  Stage 1: insert_apnea_raw()         raw signals → DB   │
│  Stage 2: insert_apnea_preprocessed() smoothed → DB     │
│  Stage 3: _extract_apnea_features()  27 features + label │
│           insert_apnea_features()    feature JSON → DB   │
│           insert_apnea_segment()     flat row → DB       │
│           insert_apnea_ecg_plot()    plot data → DB      │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Stage 4: LSTM       │  Fetch current run's GT-labelled segments only
│  Training            │  Filter → Scale → Sequence → BiLSTM
│                      │  Focal loss · Early stopping · Threshold tuning
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Results             │  AUC-ROC + 95% bootstrap CI
│                      │  Classification report at optimal F1 threshold
│                      │  → DB + pipeline.log
└──────────────────────┘
```

---

## 7. Signal Processing

### 7.1 Bandpass Filtering

**Function:** `_bandpass(signal, fs, lo=0.5, hi=40.0, order=3)`

Applied to ECG before all downstream processing. Uses `scipy.signal.filtfilt` — a zero-phase forward-backward filter that introduces no phase distortion, which is critical because R-peak timing must remain accurate.

- Passband: **0.5–40 Hz** — removes baseline wander (< 0.5 Hz) and high-frequency noise (> 40 Hz) while preserving the QRS complex shape
- The upper cutoff is clamped to `nyquist − 0.1` to prevent filter instability at the Nyquist frequency
- `filtfilt` effectively doubles the filter order (3rd-order design gives 6th-order roll-off in practice)

### 7.2 R-Peak Detection

**Function:** `_detect_r_peaks(ecg, fs)`

R-peaks are the tall spikes in the ECG representing ventricular depolarisation. Their timing underpins all HRV features and the EDR computation.

**Primary path (neurokit2):** `nk.ecg_process()` applies the Pan-Tompkins algorithm with signal quality checks. Far more robust on noisy ICU waveforms than a naive peak finder.

**Fallback path (scipy):** `scipy.signal.find_peaks()` with:
- `distance = fs × 0.4` — minimum 400 ms between peaks (enforces a maximum heart rate of 150 BPM)
- `height = std(ecg)` — peak must be at least one standard deviation above the mean

If neurokit2 raises an exception on a specific segment, the fallback is used for that segment only and the exception is logged as a warning.

### 7.3 ECG-Derived Respiration (EDR)

**Function:** `_compute_edr(ecg, r_peaks, fs_ecg, fs_resp=4)`

The legacy EDR engine. Breathing causes thoracic movement that rotates the heart's electrical axis, changing ECG amplitude and QRS shape with each breath. This function extracts that respiratory signal using two parallel engines and fuses their outputs.

**Engine 1 — QRS-Area Tracking:**
For each R-peak, computes the absolute sum of the ECG in a ±60 ms window. This area varies with respiration. The resulting time series is linearly detrended and interpolated onto a uniform 4 Hz grid.

**Engine 2 — QRS-PCA:**
Stacks all QRS complexes (each ±60 ms window) into a matrix. Applies SVD to find the dominant mode of shape variation across beats. The first principal component score time-series is a cleaner respiratory proxy because it isolates the strongest source of cross-beat variation.

**Fusion:** Both outputs are z-score normalised and their element-wise median is taken. The result is bandpass filtered at 0.1–0.5 Hz to isolate the respiratory frequency band.

**Returns:** `np.ndarray` of shape `(SEGMENT_LEN_S × fs_resp,)` — 120 samples for a 30-second segment at 4 Hz.

### 7.4 EDR v3 — Precision Engine

**Module:** `compute_edr_fixed.py` · **Function:** `compute_edr_v3()`

The upgraded EDR engine. When `compute_edr_fixed.py` is present alongside `pipeline.py`, this automatically replaces `_compute_edr`.

| Improvement | Detail |
|---|---|
| Internal computation rate | 10 Hz internally vs 4 Hz — 3× finer PSD frequency resolution |
| Parabolic interpolation | Sub-bin frequency refinement around the PSD peak gives ~0.3 BPM effective resolution |
| Autocorrelation cross-check | Compares Welch PSD estimate against autocorrelation lag to catch harmonic confusion |
| Harmonic correction | If PSD and AC estimates disagree > 3 BPM, tests whether half or double of the PSD estimate agrees better |
| Adaptive AC blending | Weights autocorrelation higher at low rates (< 14 BPM) where harmonics fall in-band |
| SNR-based engine selection | Runs both Engine A and Engine B, picks the one with higher in-band SNR |

**Returns:** `(edr_signal, bpm, quality)` — the respiratory waveform array, precision respiratory rate in BPM, and in-band SNR. The `bpm` and `quality` values are passed into `_extract_apnea_features()` to inform `resp_rate_bpm` computation.

### 7.5 Ground-Truth Resp Channel

MIMIC-IV records contain an impedance pneumography signal on the `Resp` channel — a real chest-belt measurement of respiratory effort. When present, this channel is:

1. NaN-filled and resampled to `FS_RESP` (4 Hz) into `resp_gt_full`
2. Sliced into per-segment arrays `resp_gt_seg` aligned to the same 30-second windows as the ECG
3. Passed into `_extract_apnea_features()` as the `resp_gt` argument

Inside feature extraction, two routing variables are set:

```python
resp_for_label = resp_gt if resp_gt is not None else resp   # → _resp_flag → AASM label
resp_for_feats = resp_gt if resp_gt is not None else resp   # → all resp feature values
```

This is the central fix in this pipeline. Without it, `_resp_flag` would evaluate the EDR signal and the resp-based features would also come from EDR — meaning the label would be a function of the same signal that produces the model's input features. This circular dependency corrupts training.

Records without a `Resp` channel fall back to EDR for both labelling and features. Their segments are tagged `label_source = "edr"` and excluded from LSTM training.

---

## 8. Apnea Labelling Logic

Labels are generated automatically from signal data using the AASM (American Academy of Sleep Medicine) multi-signal composite approach. No manual annotations are required.

### 8.1 Signal Flags

Each flag is a boolean. `True` means an apnea-consistent finding was detected in this 30-second segment.

**`_resp_flag(resp, fs)` — Respiratory suppression**

Detects sustained absence of respiratory effort. During apnea, airflow stops and the respiratory signal flatlines.

- Threshold: `mean(resp) − 1.5 × std(resp)`
- Flag fires if the longest continuous run of samples below this threshold spans ≥ 10 seconds
- Receives the GT Resp channel when available — this is the critical fix breaking circular labelling

**`_spo2_flag(pleth, fs, baseline_spo2)` — Oxygen desaturation**

Detects SpO₂ drops following an apnea event. Oxygen desaturation typically lags the apnea by 15–30 seconds due to lung oxygen reserves.

- SpO₂ is smoothed with a 2-second rolling median to remove motion artifact spikes
- Flag fires only when **both** conditions are simultaneously true:
  - SpO₂ drops ≥ 3% below the patient's personal baseline
  - Absolute minimum SpO₂ falls below 94%
- The dual condition prevents false positives in patients who chronically run low (e.g. COPD patients with a 92% baseline)

**`_hrv_flag(r_peaks, fs, baseline_rmssd, baseline_rr_ms)` — Autonomic signature**

Apnea causes a characteristic autonomic pattern: vagal withdrawal during the event followed by a parasympathetic rebound after breathing resumes (sudden HRV surge with bradycardia).

- **HRV surge:** `RMSSD > 1.5 × baseline_RMSSD`
- **Bradycardia:** `mean RR > 1.2 × baseline_RR`
- Both conditions must be true simultaneously — this `AND` requirement reduces false positives from isolated HRV excursions unrelated to apnea

**`_abp_flag(abp, fs, baseline_map_std, baseline_sbp)` — Haemodynamic (reference only)**

Detects blood pressure variability caused by intrathoracic pressure swings during apnea. Computed and stored in the database but **not used in the label** — ABP requires an arterial line and is not available on wearables.

### 8.2 AASM Composite Label

**Function:** `label_apnea_segment(resp_flag, spo2_flag, hrv_flag)`

| Flags positive | `true_label` | `label_confidence` |
|---|---|---|
| 3 of 3 | 1 (Apnea) | `"definite_apnea"` |
| 2 of 3 | 1 (Apnea) | `"probable_apnea"` |
| 1 of 3 | 0 (Normal) | `"possible_hypopnea"` |
| 0 of 3 | 0 (Normal) | `"normal"` |

Requiring 2 or more signals reflects standard clinical scoring philosophy — a single noisy channel should not trigger an apnea label.

### 8.3 Label Source Tracking

Every segment stores a `label_source` field in the database:

| Value | Meaning | Used for training? |
|---|---|---|
| `"mimic_resp"` | Resp flag came from GT chest-belt signal | Yes |
| `"edr"` | Resp flag came from EDR — circular fallback | No |

The training block filters on this field before any model fitting:

```python
gt_mask = seg_df["label_source"] == "mimic_resp"
seg_df = seg_df[gt_mask].reset_index(drop=True)
```

---

## 9. Feature Engineering

**Function:** `_extract_apnea_features(ecg, pleth, resp, abp, r_peaks, baseline, edr_bpm, edr_quality, resp_gt)`

27 features are computed per segment. They are defined in `APNEA_FEATURE_COLS` and form the complete LSTM input vector.

### 9.1 HRV Features

Derived from the RR interval series — the time in milliseconds between consecutive R-peaks.

| Feature | Description |
|---|---|
| `rr_mean` | Mean RR interval (ms) — inversely proportional to heart rate |
| `rr_std` | Standard deviation of RR intervals — overall HRV magnitude |
| `rmssd` | Root mean square of successive RR differences — short-term vagal tone marker |
| `pnn50` | Proportion of adjacent RR intervals differing by > 50 ms — another vagal index |
| `mean_hr` | Mean heart rate in BPM, derived as `60000 / rr_mean` |
| `hr_range` | Max HR minus min HR in the segment — autonomic reactivity range |
| `lf_hf_ratio` | Low/high frequency HRV power ratio (requires neurokit2) — sympathovagal balance |

If fewer than 3 R-peaks are detected, all HRV features default to 0.0.

### 9.2 Respiratory Features

All computed from `resp_for_feats` — the GT Resp channel when available, EDR otherwise.

| Feature | Description |
|---|---|
| `resp_rate_bpm` | Dominant respiratory rate in BPM. GT Resp path: Welch PSD on the chest-belt signal. EDR path: v3 precision estimate if SNR ≥ 1.5, otherwise Welch PSD on the EDR signal |
| `resp_rate_variability` | Standard deviation of inter-breath intervals in seconds — irregular breathing is an apnea marker |
| `flatline_duration_s` | Longest continuous period in seconds where the resp signal is suppressed below threshold — direct proxy for apnea duration |
| `resp_amplitude_mean` | Mean absolute amplitude — respiratory effort magnitude |
| `resp_amplitude_std` | Standard deviation of amplitude — effort variability across the segment |

### 9.3 SpO2 Features

| Feature | Description |
|---|---|
| `spo2_mean` | Mean SpO₂ over the segment |
| `spo2_min` | Minimum SpO₂ — the nadir of any desaturation event |
| `spo2_delta_index` | Max minus min SpO₂ — total desaturation range |
| `odi` | Oxygen Desaturation Index — number of times SpO₂ crosses below `(baseline − 3%)` |
| `t90` | Fraction of the segment spent below SpO₂ = 90% — standard clinical hypoxia severity measure |
| `spo2_approx_entropy` | Approximate entropy of the SpO₂ signal — low values indicate irregular pathological fluctuations |

SpO₂ NaN gaps from the intermittent simulation are forward-filled then backward-filled before feature extraction.

### 9.4 ABP Features

Included for completeness. The model learns to down-weight these given their inconsistency at inference time.

| Feature | Description |
|---|---|
| `map_mean` | Mean arterial pressure — 2-second rolling average of ABP |
| `map_std` | Standard deviation of MAP — haemodynamic stability |
| `map_variability` | Alias of `map_std` kept for schema compatibility |
| `sbp_max` | Maximum systolic pressure — apnea-related surge marker |
| `dbp_min` | Minimum diastolic pressure |
| `pulse_pressure` | `sbp_max − dbp_min` — arterial pulse amplitude |

### 9.5 Cross-Signal Features

These capture interactions between signals that single-channel analysis cannot detect.

| Feature | Description |
|---|---|
| `resp_spo2_lag_s` | Cross-correlation lag in seconds between the resp signal and SpO₂. Reflects the oxygen store delay between an apnea and the resulting desaturation. Clinically expected: 15–30 seconds |
| `ptt_ms` | Pulse Transit Time — milliseconds from R-peak to the foot of the corresponding ABP pulse. Reflects arterial stiffness and beat-to-beat blood pressure changes. Only physiologically plausible values (50–500 ms) are included |
| `ecg_resp_coherence` | Spectral coherence between the RR interval series and the resp signal in the high-frequency band (0.15–0.4 Hz). Measures Respiratory Sinus Arrhythmia — the normal coupling between breathing and heart rate. High coherence = intact autonomic function |

---

## 10. LSTM Model

### 10.1 Architecture

```
Input: (TIMESTEPS=10, N_FEATURES=27)
    ↓
SpatialDropout1D(0.2)
    ↓
Bidirectional LSTM(48, return_sequences=True, recurrent_dropout=0.2, L2=1e-4)
    ↓
Dropout(0.3)
    ↓
Bidirectional LSTM(24, recurrent_dropout=0.2, L2=1e-4)
    ↓
Dropout(0.2)
    ↓
Dense(16, ReLU, L2=1e-4)
    ↓
Dense(1, Sigmoid)
    ↓
Output: P(apnea) ∈ [0, 1]
```

The architecture is deliberately conservative — smaller LSTM units (48/24), recurrent dropout, L2 regularisation on all kernel weights, and spatial input dropout together prevent overfitting on the approximately 400 available training sequences.

**Why Bidirectional?** Apnea has strong temporal context in both directions. The body shows pre-apnea changes (rising HR, falling HRV) before an event, and post-apnea recovery (HRV surge, bradycardia) after it. Reading the 10-segment window both forward and backward captures both the build-up and the recovery.

**Why SpatialDropout?** Unlike standard dropout which zeroes individual feature values, SpatialDropout1D zeroes entire feature channels across a timestep. This forces the model to learn redundant representations across features — more robust to noisy or missing signals at inference time.

**Why recurrent_dropout?** Applies dropout to the recurrent connections (the h → h path) rather than just the input connections. This regularises the LSTM's temporal memory directly and is the most effective form of LSTM regularisation for small datasets.

### 10.2 Focal Loss

Standard binary cross-entropy struggles with class imbalance because loss is dominated by the majority class. Focal loss adds a modulating factor that down-weights easy well-classified examples and focuses learning on hard misclassifications.

```
loss = alpha_t × (1 − p_t)^gamma × BCE
```

Where:
- `gamma = 2.0` — the focusing parameter. Reduces the loss contribution from easy examples quadratically
- `alpha = 0.75` — up-weights the positive (apnea) class by 3× relative to normal
- `p_t` — the model's probability of the correct class
- `alpha_t` — per-sample alpha based on the true class label

The result: the model pays much more attention to apnea segments it is currently misclassifying (high loss, low `p_t`) and nearly ignores normal segments it already classifies confidently.

### 10.3 Training Strategy

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam, lr=1e-3 | Adaptive learning rate handles varied feature scales automatically |
| Max epochs | 60 | Early stopping prevents actual overfitting |
| Batch size | 32 | Large enough for stable gradient estimates on ~300 training sequences |
| Validation split | 10% of training | Held out for early stopping monitor |
| Early stopping | patience=5, monitor=val_AUC | Stops 5 epochs after peak validation AUC; restores best weights |

**Sequence construction:** Each training sample is a window of 10 consecutive 30-second segments. The label for sample `i` is the label of segment `i + 10` — the model predicts the apnea status of the segment at the end of the window, using the preceding 5 minutes as context.

**Stratified split:** The 80/20 train/test split uses `train_test_split(..., stratify=y_seq)`. This ensures the test set contains the same apnea/normal ratio as the full dataset. Without stratification, temporal ordering would concentrate most apnea examples in the training partition and evaluation would be meaningless.

### 10.4 Threshold Tuning

The default sigmoid threshold of 0.5 is rarely optimal under class imbalance. After training, the pipeline searches for the threshold that maximises F1 on the test set:

```python
thresholds = np.arange(0.25, 0.75, 0.01)
best_thresh = thresholds[np.argmax([f1_score(y_te, (y_prob > t).astype(int))
                                    for t in thresholds])]
```

The classification report is generated at this optimal threshold. Both the threshold and its F1 score are logged.

### 10.5 Evaluation

**AUC-ROC with 95% bootstrap confidence interval:**

```python
# 1000 bootstrap resamples of the test set
boot_aucs = [roc_auc_score(y_te[idx], y_prob[idx])
             for idx in bootstrap_indices if both_classes_present]
ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])
```

This gives a principled uncertainty estimate on the AUC. On a test set of ~76 sequences, the confidence interval is typically ±0.10 to ±0.15 — accurately reflecting the limited data size.

The classification report shows per-class precision, recall, F1, and support at the optimal threshold.

---

## 11. Run ID Isolation

Every pipeline run generates a unique `run_id` timestamp at startup:

```python
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# e.g. "20260609_103019"
```

This `run_id` is stored with every inserted segment row. The training block fetches only the current run's rows:

```python
segs = fetch_apnea_segments(run_id=run_id)
```

This means you can run the pipeline multiple times without deleting the database. Each run trains on its own freshly-ingested data. Old rows from previous runs accumulate for historical reference but never contaminate the current training set.

The only time you need to delete the DB is when you change the **schema** (add or remove columns) — because SQLite's `CREATE TABLE IF NOT EXISTS` will not add new columns to an existing table.

---

## 12. Database Schema

Managed by `CLI/db/database.py`. The following tables are populated during each run:

| Table | Populated by | Contents |
|---|---|---|
| `apnea_raw` | `insert_apnea_raw()` | Raw ECG, PPG, EDR, ABP arrays per segment stored as JSON |
| `apnea_preprocessed` | `insert_apnea_preprocessed()` | Smoothed signals, R-peak array, RR statistics |
| `apnea_features` | `insert_apnea_features()` | Full feature dictionary as JSON |
| `apnea_segments` | `insert_apnea_segment()` | Flat row of all 27 features plus all label and metadata columns |
| `apnea_ecg_plot` | `insert_apnea_ecg_plot()` | ECG and overlay signals for visualisation (max 2 per record) |
| `apnea_results` | `insert_apnea_results()` | AUC-ROC value and full classification report as JSON |
| `pipeline_log` | `log_module()` | Stage names, statuses, timestamps, and row counts |

**Key columns in `apnea_segments`:**

| Column | Type | Description |
|---|---|---|
| `record` | TEXT | MIMIC-IV record name (e.g. `"83404654"`) |
| `segment_idx` | INTEGER | 0-based segment index within the record |
| `run_id` | TEXT | Timestamp of the pipeline run that created this row |
| `true_label` | INTEGER | 0 = normal, 1 = apnea |
| `label_confidence` | TEXT | One of: `"definite_apnea"`, `"probable_apnea"`, `"possible_hypopnea"`, `"normal"` |
| `label_source` | TEXT | `"mimic_resp"` or `"edr"` — determines whether this segment is used for training |
| `resp_flag` | INTEGER | 1 if respiratory suppression detected |
| `spo2_flag` | INTEGER | 1 if SpO₂ desaturation detected |
| `hrv_flag` | INTEGER | 1 if HRV autonomic signature detected |
| `abp_flag` | INTEGER | 1 if haemodynamic signature detected (reference only, not in label) |
| `signals_positive` | INTEGER | Count of resp + spo2 + hrv flags that fired (0–3) |

---

## 13. Logging

The pipeline logs to two sinks simultaneously:

- **Console (stdout)** — real-time progress visible during the run
- **`pipeline.log`** — persistent file created in the current working directory (the `project2/` root, not inside `pipeline/`)

Log format:
```
2026-06-09 10:30:19,702 | INFO | __main__ | [APNEA] 83404654: ground-truth Resp channel found
```

**Key log lines to watch for:**

| Message | Meaning |
|---|---|
| `APNEA MODULE  run_id=XXXXXXXX` | Clean new run started with a unique ID |
| `ground-truth Resp channel found` | This record has a chest-belt signal — labels will be trustworthy |
| `no Resp channel — EDR fallback` | No chest belt — this record's segments will be excluded from training |
| `Label filter: X / Y segments have GT Resp labels` | How many segments are eligible for training |
| `Training set: N — X apnea / Y normal (Z% apnea)` | Class distribution sanity check |
| `%.0f%% apnea — labelling likely broken` | > 70% apnea prevalence triggers an error and aborts training |
| `Optimal threshold: X.XX (F1=Y.YYY)` | The classification threshold used for the final report |
| `AUC: X.XXXX (95% CI: X.XXX–X.XXX)` | Final result with bootstrap uncertainty |

---

## 14. Wearable Constraints Simulated

The pipeline deliberately models the limitations of a real wearable device:

**No chest respiratory belt at inference time.** Respiration is derived entirely from the ECG (EDR) for the model's input features. The MIMIC-IV chest belt signal is used only to generate training labels and is never fed into the model as a feature.

**Intermittent SpO₂.** Real pulse oximeters conserve battery by sampling every 3–5 minutes rather than continuously. This is simulated per-segment:

```python
take_reading = (i % np.random.randint(6, 11)) == 0
if take_reading:
    pleth_seg = pleth_full[i*spp:(i+1)*spp]    # real signal
    last_spo2_val = float(np.mean(pleth_seg))
else:
    pleth_seg = np.full(spp, last_spo2_val)     # hold last known value
```

**Capped sampling rates.** ECG is limited to 125 Hz and PPG to 120 Hz even though MIMIC-IV is natively at 320 Hz, reflecting typical BLE chip capabilities.

**ABP unavailable at inference.** ABP features are computed and stored during training, but a wearable cannot measure arterial blood pressure. The model must learn their unreliability through the training signal alone.

---

## 15. Troubleshooting

**`label_source column not found` warning in training block**
The database was built before `label_source` was added to the schema. Delete it and rebuild:
```bash
rm -f vitals_pipeline.db vitals_pipeline.db-wal vitals_pipeline.db-shm
python pipeline/pipeline.py
```

**`Label filter: 0 / 200 segments have GT Resp labels`**
All segments have `label_source = "edr"`, meaning the `pipeline.py` writing segments does not contain the `resp_gt` changes. Verify the correct file is in place:
```bash
grep -c "resp_gt" pipeline/pipeline.py   # must print 13
```

**`No segments for run_id=XXXX — aborting`**
Segments were inserted but `fetch_apnea_segments(run_id=run_id)` returned zero rows. Either the `run_id` column is missing from the schema or the latest `database.py` is not in place. Check:
```bash
python3 -c "
import sqlite3
con = sqlite3.connect('vitals_pipeline.db')
cols = [c[1] for c in con.execute('PRAGMA table_info(apnea_segments)').fetchall()]
print('run_id' in cols, 'label_source' in cols)
"
```
Both should print `True`. If not, delete the DB and replace `database.py`.

**`%.0f%% apnea — labelling likely broken` error**
More than 70% of segments labelled as apnea. Common causes: `_resp_flag` threshold too sensitive on the specific records loaded, `_hrv_flag` accidentally using `OR` instead of `AND`, or the Pleth values are not in the physiological SpO₂ range (80–100) causing `_spo2_flag` to fire constantly.

**`Only one class in test set — AUC not computed`**
The stratified split failed because there are too few segments of one class to allocate any to the test set. Increase `N_MIMIC_RECORDS` or inspect the label distribution before training.

**`nk.ecg_process failed` warnings appearing frequently**
neurokit2 raised exceptions on most segments — common on very noisy ICU ECG. The scipy fallback is used automatically but R-peak quality is lower. Check whether the records being loaded have clean Lead II signals.

**Pipeline runs but fetches zero records**
PhysioNet may be rate-limiting or temporarily unavailable. Try reducing `N_MIMIC_RECORDS` to 5 for a connectivity test. Check your internet connection.

---

## 16. Glossary

| Term | Definition |
|---|---|
| **AASM** | American Academy of Sleep Medicine — defines clinical apnea scoring rules |
| **ABP** | Arterial Blood Pressure — invasive continuous blood pressure measurement requiring an arterial line |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic curve — 0.5 = random, 1.0 = perfect |
| **BiLSTM** | Bidirectional LSTM — a recurrent network that processes sequences both forward and backward |
| **Bootstrap CI** | Confidence interval computed by resampling the test set many times to estimate distribution of the metric |
| **ECG** | Electrocardiogram — the electrical signal of the heart |
| **EDR** | ECG-Derived Respiration — estimating the breathing signal from subtle ECG shape changes |
| **Focal Loss** | A loss function that down-weights easy examples and focuses gradient on hard misclassifications |
| **HRV** | Heart Rate Variability — beat-to-beat fluctuations in RR interval; reflects autonomic nervous system activity |
| **L2 Regularisation** | A weight penalty term added to the loss that discourages large weights and reduces overfitting |
| **Label Source** | `"mimic_resp"` or `"edr"` — tracks whether a segment's apnea label came from ground truth or the circular EDR fallback |
| **MAP** | Mean Arterial Pressure — time-averaged arterial blood pressure |
| **MIMIC-IV** | Medical Information Mart for Intensive Care IV — a large publicly available ICU waveform database |
| **ODI** | Oxygen Desaturation Index — number of SpO₂ desaturation events per unit time |
| **OSA** | Obstructive Sleep Apnea — repeated cessation of breathing during sleep due to airway obstruction |
| **PCA** | Principal Component Analysis — finds dominant modes of variation in a dataset via SVD |
| **PPG** | Photoplethysmography — optical blood volume pulse signal, used as a SpO₂ proxy |
| **PSD** | Power Spectral Density — distribution of signal energy across frequencies |
| **PTT** | Pulse Transit Time — time for a pressure wave to travel from the heart to a peripheral measurement point |
| **QRS Complex** | The tall spike in the ECG representing ventricular depolarisation |
| **RMSSD** | Root Mean Square of Successive Differences — a short-term HRV metric reflecting vagal tone |
| **RSA** | Respiratory Sinus Arrhythmia — the normal modulation of heart rate by the breathing cycle |
| **Run ID** | A timestamp-based string that tags each pipeline run's database rows, enabling isolation without deleting the DB |
| **SNR** | Signal-to-Noise Ratio — ratio of in-band signal power to out-of-band noise power |
| **SpO₂** | Peripheral oxygen saturation — percentage of haemoglobin carrying oxygen, measured by pulse oximetry |
| **SpatialDropout** | A dropout variant that zeroes entire feature channels rather than individual values, promoting redundant representations |
| **SVD** | Singular Value Decomposition — the matrix factorisation underlying PCA |
| **T90** | Time below SpO₂ = 90% — a standard clinical hypoxia severity measure |
| **Welch PSD** | Estimates power spectral density by averaging periodograms of overlapping, windowed signal segments |
