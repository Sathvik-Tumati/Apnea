# Model Architecture

This pipeline trains an **XGBoost (seq)** model on MIMIC-IV + SLPDB data using a stratified train/val/test split (`random_state=42`).

---

## Model: XGBoost (seq)

A gradient-boosted tree model that uses sequence data but flattens it: each input is `(T, F)` → `(T×F,)` where T=10 timesteps, F=30 features, producing a 300-dimensional feature vector per sample.

### Why XGBoost?

- **Interpretability**: XGBoost provides native feature importance, making it easier to explain which physiological signals drove a prediction.
- **Speed**: No GPU required at inference time — XGBoost runs purely on CPU.
- **Robustness**: Gradient boosted trees handle missing values and feature scale differences gracefully.

### Modality Dropout (Training Augmentation)

During training, for MIMIC-IV segments only, we randomly zero out modality groups to teach the model to handle sensor dropouts:
- 30% of MIMIC segments: SpO2 features zeroed + `has_spo2` set to 0
- 30% of MIMIC segments: ABP features zeroed + `has_abp` set to 0

### XGBoost Hyperparameters

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 (with early stopping) |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | `N_neg / N_pos` (handles class imbalance) |
| `eval_metric` | AUC |
| `early_stopping_rounds` | 20 (stops if val AUC doesn’t improve) |

### Input Format

XGBoost takes `(N, TIMESTEPS, N_FEATURES)` sequences, flattened to `(N, TIMESTEPS × N_FEATURES)` = `(N, 300)`. The `_build_combined_dataset()` scaler and `_apply_modality_dropout_sequences()` augmentation are applied.

### Saved Artefacts

| File | Contents |
|---|---|
| `apnea_model_xgb_seq.pkl` | Serialised `XGBClassifier` (pickled) |
| `apnea_scaler_tree.pkl` | Fitted `StandardScaler` |
