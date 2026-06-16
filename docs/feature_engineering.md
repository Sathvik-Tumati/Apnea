# Feature Engineering

The pipeline extracts **30 distinct physiological features** for every 30-second window of data. These features are extracted regardless of the input data source, with missing modalities intelligently imputed and flagged.

## 1. ECG and EDR Features (12 cols)
Always present. If ground-truth respiratory signals are missing, we fall back to a dual-engine ECG-Derived Respiration (EDR) algorithm that fuses QRS-area tracking with PCA to extract the respiratory envelope.

* `rr_mean`, `rr_std`: R-R interval statistics.
* `rmssd`, `pnn50`: Standard Heart Rate Variability (HRV) metrics indicating parasympathetic tone.
* `mean_hr`, `hr_range`: Heart rate boundaries.
* `lf_hf_ratio`: Autonomic nervous system balance.
* `resp_rate_bpm`, `resp_rate_variability`: Extracted from GT Resp if available, else from EDR.
* `flatline_duration_s`: Longest period of respiratory cessation.
* `resp_amplitude_mean`, `resp_amplitude_std`: Depth of breathing.

## 2. SpO2 / PPG Features (6 cols)
If SpO2 is missing, these are zeroed out and the `has_spo2` flag is set to `0`.

* `spo2_mean`, `spo2_min`: Absolute oxygenation levels.
* `spo2_delta_index`: Sum of absolute differences between successive measurements (measures volatility).
* `odi` (Oxygen Desaturation Index): Rate of >3% drops per hour.
* `t90`: Time spent below 90% saturation.
* `spo2_approx_entropy`: Signal complexity/unpredictability.

## 3. ABP Features (6 cols)
If Arterial Blood Pressure is missing, these are zeroed out and `has_abp` is set to `0`.

* `map_mean`, `map_std`: Mean Arterial Pressure statistics.
* `map_variability`: Coefficient of variation.
* `sbp_max`, `dbp_min`: Systolic and diastolic extremes.
* `pulse_pressure`: Difference between systolic and diastolic.

## 4. Cross-Signal Features (3 cols)
These measure the phase and delay relationships between different physiological systems.

* `resp_spo2_lag_s`: Time delay between a respiratory cessation and the resulting SpO2 drop (usually 10-30s).
* `ptt_ms` (Pulse Transit Time): Delay between ECG R-peak and PPG systolic peak. Drops during sympathetic arousal (apnea termination).
* `ecg_resp_coherence`: Spectral coherence between heart rate and breathing (measures Respiratory Sinus Arrhythmia).

## 5. Modality Flags (3 cols)
* `has_spo2`
* `has_abp`
* `has_resp_gt`

These are appended to the end of the feature vector so the neural network can dynamically route its attention.
