from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf
import keras
from pipeline.modules.config import *
from pipeline.modules.config import _HAS_SPO2_IDX, _HAS_ABP_IDX, _HAS_RESP_IDX

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE — MODALITY-AWARE BiLSTM
# ══════════════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable(package="pipeline")
class GatherFlags(tf.keras.layers.Layer):
    """
    Extracts a fixed set of feature indices from the last timestep of a
    sequence tensor.  Replaces the Lambda layer that called tf.gather()
    directly, which caused NameError('tf') on model reload because Lambda
    closures capture `tf` by name from their definition module — a name
    that isn't available in Keras's deserialization scope.

    This subclass serializes cleanly: the index list is stored in get_config()
    and reconstructed via from_config(), so load_model() works without
    custom_objects or tf.saved_model workarounds.
    """

    def __init__(self, indices: list, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)          # plain Python list — fully serializable

    def call(self, t):
        # t shape: (batch, timesteps, features)
        last = t[:, -1, :]                   # (batch, features)
        return tf.gather(last, self.indices, axis=-1)   # (batch, len(indices))

    def get_config(self):
        cfg = super().get_config()
        cfg["indices"] = self.indices
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)


def _build_model(n_features: int, timesteps: int) -> "tf.keras.Model":
    """
    Modality-aware BiLSTM.

    The modality flags (has_spo2, has_abp, has_resp_gt) are present inside
    the feature vector throughout the LSTM, then the final flag values are
    also concatenated with the LSTM output before the decision head.
    This gives the model both sequential and point-in-time modality context.
    """
    inp = tf.keras.layers.Input(shape=(timesteps, n_features), name="ecg_features")

    # Shared temporal encoder
    x = tf.keras.layers.SpatialDropout1D(0.2)(inp)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64, return_sequences=True, recurrent_dropout=0.2,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        name="bilstm_1",
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32, recurrent_dropout=0.2,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        ),
        name="bilstm_2",
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Extract the last timestep's modality flags and concatenate
    flag_idxs = [_HAS_SPO2_IDX, _HAS_ABP_IDX, _HAS_RESP_IDX]
    flags = GatherFlags(flag_idxs, name="modality_flags")(inp)   # ← replaces Lambda
    x = tf.keras.layers.Concatenate(name="lstm_with_flags")([x, flags])

    # Modality-aware decision head
    x = tf.keras.layers.Dense(
        32, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="decision_head",
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid", name="apnea_prob")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="modality_aware_bilstm")


def _focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    def loss_fn(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss_fn


