# Feature Engineering

The pipeline extracts **30 distinct physiological features** for every 30-second window of data. Features are extracted regardless of input source, with missing modalities zeroed out and flagged explicitly.

---

## Feature Vector Layout

| Index | Group | Features | Count |
|---|---|---|---|
| 0–11 | ECG + EDR | HRV, heart rate, EDR-derived respiration | 12 |
| 12–17 | SpO2 | Oxygenation statistics, desaturation events | 6 |
| 18–23 | ABP | Arterial blood pressure statistics | 6 |
| 24–26 | Cross-signal | Phase relationships between physiological systems | 3 |
| 27–29 | Modality flags | `has_spo2`, `has_abp`, `has_resp_gt` | 3 |
| **Total** | | | **30** |

---

## 1. ECG and EDR Features (indices 0–11)

Always present. Computed from the 30-second ECG segment at 125 Hz.

If ground-truth respiratory signals are unavailable (which is the case for all wearable/MongoDB recordings), we fall back to a **dual-engine ECG-Derived Respiration (EDR)** algorithm that fuses:
- **QRS-area tracking**: The area under each QRS complex varies with respiration due to thoracic impedance changes
- **PCA decomposition**: Principal component of R-peak amplitude and morphology variation, which encodes the respiratory envelope

| Feature | Description |
|---|---|
| `rr_mean` | Mean R-R interval (ms) |
| `rr_std` | Standard deviation of R-R intervals (ms) |
| `rmssd` | Root mean square of successive R-R differences — parasympathetic tone |
| `pnn50` | Fraction of successive R-R differences > 50 ms |
| `mean_hr` | Mean heart rate (bpm) |
| `hr_range` | Max HR − Min HR within the segment (bpm) |
| `lf_hf_ratio` | Low-frequency / high-frequency HRV power ratio — autonomic balance |
| `resp_rate_bpm` | Respiratory rate (breaths/min), from GT Resp if available, else EDR |
| `resp_rate_variability` | Std dev of breath-to-breath interval (s) |
| `flatline_duration_s` | Longest continuous respiratory cessation (s) — key apnea indicator |
| `resp_amplitude_mean` | Mean peak-to-trough respiratory amplitude |
| `resp_amplitude_std` | Variability of respiratory amplitude |

---

## 2. SpO2 Features (indices 12–17)

If SpO2 is missing, all 6 values are zeroed and `has_spo2 = 0`.

In the MongoDB pipeline, SpO2 comes from the device-computed 1 Hz stream (`spo2_unfiltered_data`). Alignment to ECG segments is done by timestamp matching.

| Feature | Description |
|---|---|
| `spo2_mean` | Mean SpO2 (%) over the 30-second window |
| `spo2_min` | Minimum SpO2 (%) — floor of oxygenation |
| `spo2_delta_index` | Sum of absolute successive differences — volatility metric |
| `odi` | Oxygen Desaturation Index: rate of ≥3% drops per hour |
| `t90` | Fraction of time below 90% saturation |
| `spo2_approx_entropy` | Approximate entropy — signal complexity/unpredictability |

**Hypopnea proxy:** ODI and T90 together approximate the hypopnea component of AHI. A ≥3% SpO2 drop within 30 seconds of a respiratory event (flagged by flatline_duration_s) is the primary detection mechanism.

---

## 3. ABP Features (indices 18–23)

If Arterial Blood Pressure is unavailable (all wearable/MongoDB recordings), all 6 values are zeroed and `has_abp = 0`.

Present only in MIMIC-IV training data (hospital arterial line monitoring).

| Feature | Description |
|---|---|
| `map_mean` | Mean Arterial Pressure (mmHg) mean |
| `map_std` | MAP standard deviation |
| `map_variability` | Coefficient of variation of MAP |
| `sbp_max` | Maximum systolic blood pressure (mmHg) |
| `dbp_min` | Minimum diastolic blood pressure (mmHg) |
| `pulse_pressure` | SBP − DBP (mmHg) — marker of arterial stiffness and arousal |

---

## 4. Cross-Signal Features (indices 24–26)

These capture phase and delay relationships between physiological systems. Present when multiple modalities are available.

| Feature | Description |
|---|---|
| `resp_spo2_lag_s` | Time delay between respiratory cessation and the resulting SpO2 drop (typically 10–30 s in apnea) |
| `ptt_ms` | Pulse Transit Time — delay between ECG R-peak and SpO2/PPG systolic peak; drops during sympathetic arousal at apnea termination |
| `ecg_resp_coherence` | Spectral coherence between heart rate oscillations and breathing frequency (measures Respiratory Sinus Arrhythmia) |

---

## 5. Modality Flags (indices 27–29)

Three binary indicators appended at the end of the feature vector. They are extracted from the last timestep by the `GatherFlags` layer and concatenated directly with the LSTM hidden state before classification.

| Feature | Value | Meaning |
|---|---|---|
| `has_spo2` | 1 | SpO2 features are real measured values |
| `has_spo2` | 0 | SpO2 features are zeroed (device not available) |
| `has_abp` | 1 | ABP features are real (hospital arterial line) |
| `has_abp` | 0 | ABP features are zeroed (wearable / no arterial line) |
| `has_resp_gt` | 1 | Respiratory features from ground-truth channel |
| `has_resp_gt` | 0 | Respiratory features derived from EDR (ECG-only) |

---

## AHI Proxy Calculation

The AHI proxy is derived from model predictions over the sleep-filtered segment set:

```
n_apnea_segments  =  count of 30-s windows where apnea_prob ≥ threshold
sleep_duration_h  =  n_scored_segments × 30 / 3600    (segment-count based)

AHI proxy  =  n_apnea_segments / sleep_duration_h
```

**Severity classification:**

| AHI | Severity |
|---|---|
| < 5 | Normal |
| 5–15 | Mild |
| 15–30 | Moderate |
| > 30 | Severe |

> **Note on denominator:** After sleep filtering, `sleep_duration_h` is computed from the number of scored segments (not wall-clock timestamps), because the sleep filter produces a non-contiguous subset of the original timeline. Using timestamp-derived duration would overcount the gap time between sleep windows.
