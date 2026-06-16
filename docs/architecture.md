# Modality-Aware BiLSTM Architecture

This pipeline is built around a custom **Modality-Aware Bidirectional LSTM** designed to handle the realities of multi-sensor data where signals frequently drop out or are simply unavailable (like when transitioning from hospital monitors to consumer wearables).

## The Problem
Clinical datasets like MIMIC-IV are rich: they have ECG, PPG/SpO2, Respiratory Impedance, and Arterial Blood Pressure. Wearable devices usually only have ECG (and sometimes PPG). 
Training only on MIMIC-IV creates a model that fails entirely when fed ECG-only data. Training only on ECG ignores valuable physiological relationships present in clinical data.

## The Solution

1. **Shared Encoder:** We train a shared Bidirectional LSTM on all available features from both MIMIC-IV (rich) and SLPDB (ECG-only). 
2. **Modality Flags:** The last three features in our 30-feature vector are boolean flags:
   - `has_spo2`
   - `has_abp`
   - `has_resp_gt` (Ground truth respiration, 0 if using EDR)
3. **Modality Dropout:** During training, we randomly drop (zero out) the SpO2, ABP, and Resp channels in 30% of the MIMIC-IV batches, setting the corresponding flags to `0`. This forces the network to learn robust ECG-only fallbacks (like EDR) without forgetting how to use the rich signals when they are present.
4. **Modality-Aware Head:** A custom `GatherFlags` layer extracts these flags from the final timestep and concatenates them with the BiLSTM's hidden state before the final dense decision head. This explicitly tells the decision head *which sensors to trust*.

## Keras Implementation Details

To ensure the model can be saved and loaded cleanly (e.g., inside an Android or web backend) without `custom_objects` headaches, we use a class-based custom layer registered with Keras:

```python
@keras.saving.register_keras_serializable(package="pipeline")
class GatherFlags(tf.keras.layers.Layer):
    ...
```

This replaces standard `Lambda` layers, which fail to deserialize properly when referencing `tf.gather` across module boundaries.

## Loss Function
We use **Focal Loss** (`γ=2.0`, `α=0.75`) because Apnea events are minority classes (~10-20% of segments). This forces the model to focus on hard-to-classify borderline apneic segments rather than overwhelming the gradient with easy normal segments.
