# Modality-Aware BiLSTM Architecture

This pipeline is built around a custom **Modality-Aware Bidirectional LSTM** designed to handle the realities of multi-sensor data where signals frequently drop out or are simply unavailable (like when transitioning from hospital monitors to consumer wearables).

---

## The Problem

Clinical datasets like MIMIC-IV are rich: they have ECG, SpO2/PPG, Respiratory Impedance, and Arterial Blood Pressure. Wearable devices and MongoDB-based hospital monitors usually only have ECG and sometimes device-computed SpO2 (1 Hz).

Training only on MIMIC-IV creates a model that assumes all modalities are present. Training only on ECG ignores valuable physiological relationships in the clinical data. We need a model that gracefully degrades: using SpO2 when available, falling back to ECG-only when it isn't.

---

## The Solution: Modality Flags + Modality Dropout

### 1. Shared Encoder
A shared Bidirectional LSTM trains on all features from both MIMIC-IV (ECG + SpO2 + ABP + Resp) and SLPDB (ECG only). The encoder sees all 30 features, with missing modalities **zeroed out** rather than omitted.

### 2. Modality Flags
The last three features in the 30-feature vector are boolean flags:
- `has_spo2` — 1 if SpO2 features are real values, 0 if zeroed
- `has_abp` — 1 if ABP features are real, 0 if zeroed (hospital-only)
- `has_resp_gt` — 1 if using ground-truth Resp channel, 0 if using EDR fallback

### 3. Modality Dropout (Training Augmentation)
During training, for MIMIC-IV segments only, we randomly zero out modality groups:
- 30% of MIMIC segments: SpO2 features zeroed + `has_spo2` set to 0
- 30% of MIMIC segments: ABP features zeroed + `has_abp` set to 0

This forces the BiLSTM encoder to learn robust ECG-only representations (using EDR for respiration) without forgetting how to exploit SpO2 and ABP when they are present.

SLPDB segments are never augmented this way — they are already ECG-only and provide the ECG-only signal path.

### 4. Modality-Aware Decision Head
A custom `GatherFlags` layer extracts `[has_spo2, has_abp, has_resp_gt]` from the final timestep and concatenates them directly with the BiLSTM hidden state before the decision head. This explicitly tells the classifier *which sensors to trust* at inference time.

```
Input (N, T, 30)
    │
    ├──→ Bidirectional LSTM (64) ──→ Bidirectional LSTM (32) ──→ LSTM output
    │                                                                │
    └──→ GatherFlags([has_spo2, has_abp, has_resp_gt]) ─────────────┤
                                                                     │
                                                               Concatenate
                                                                     │
                                                              Dense(32, relu)
                                                                     │
                                                           Dense(1, sigmoid)  → apnea_prob
```

---

## Feature Index Map

The 30 features are laid out in strict order (defined in `pipeline/modules/config.py`):

| Index range | Group | Columns |
|---|---|---|
| 0–11 | ECG + EDR | `rr_mean`, `rr_std`, `rmssd`, `pnn50`, `mean_hr`, `hr_range`, `lf_hf_ratio`, `resp_rate_bpm`, `resp_rate_variability`, `flatline_duration_s`, `resp_amplitude_mean`, `resp_amplitude_std` |
| 12–17 | SpO2 | `spo2_mean`, `spo2_min`, `spo2_delta_index`, `odi`, `t90`, `spo2_approx_entropy` |
| 18–23 | ABP | `map_mean`, `map_std`, `map_variability`, `sbp_max`, `dbp_min`, `pulse_pressure` |
| 24–26 | Cross-signal | `resp_spo2_lag_s`, `ptt_ms`, `ecg_resp_coherence` |
| 27–29 | Modality flags | `has_spo2`, `has_abp`, `has_resp_gt` |

---

## Keras Implementation Details

The `GatherFlags` layer is a registered custom Keras layer:

```python
@keras.saving.register_keras_serializable(package="pipeline")
class GatherFlags(tf.keras.layers.Layer):
    def __init__(self, idxs: list, **kwargs):
        super().__init__(**kwargs)
        self.idxs = idxs

    def call(self, x):
        # Extract flags from the last timestep only
        return tf.gather(x[:, -1, :], self.idxs, axis=1)

    def get_config(self):
        return {**super().get_config(), "idxs": self.idxs}
```

This replaces `Lambda` layers (which fail to deserialize when the `tf` name isn't available in Keras's deserialization scope) with a cleanly serializable subclass.

> **Critical:** To load `apnea_model.keras`, the module `pipeline.modules.model` **must be imported first** so that `GatherFlags` is registered with Keras before `load_model()` is called. `pipeline/infer.py` handles this automatically.

---

## Loss Function

We use **Focal Loss** (`γ=2.0`, `α=0.75`) to address class imbalance:
- Apnea events are minority classes (~15–30% of segments in training data)
- Focal loss down-weights easy-to-classify normal segments and focuses the gradient on hard borderline apneic segments

```python
def _focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -alpha * (1 - pt)**gamma * tf.math.log(pt + 1e-7)
    return loss
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Epochs | 80 |
| Batch size | 32 |
| BiLSTM units | 64 → 32 |
| Dense head | 32 → 1 |
| Recurrent dropout | 0.2 |
| L2 regularisation | 1e-4 on all dense/LSTM kernels |
| Class weighting | `{normal: 1.0, apnea: N_normal/N_apnea}` |
| Modality dropout rate | 30% (SpO2), 30% (ABP) on MIMIC segments |
| Sequence length (TIMESTEPS) | 10 (consecutive 30s epochs = 5 min context) |
