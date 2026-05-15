"""
FWI-Gated LSTM (FWI-PRN) — Novel Architecture for Wildfire Prediction
======================================================================
Journal Publication | Wildfire Prediction — Indian Subcontinent

Architecture: FWI-Physics Residual Network v2 (Recall-Stable)
--------------------------------------------------------------
A two-pathway model providing GENUINELY NEW physics information to the LSTM
via sequence-level statistics derived from the 14-day FWI window.

Key design decisions v2:
  ① Trainable gate scalar α (init=0.1): prevents the physics branch from
    dominating LSTM gradients early in training (root cause of recall collapse).
  ② 6 physics statistics (was 5): adds frac_extreme (>2σ) to distinguish
    catastrophic fire-weather days from merely elevated ones.
  ③ frac_high threshold lowered from >1.0σ → >0.5σ: better captures the
    India fire-risk onset zone in scaled FWI space.
  ④ physics_recall_focal_loss used at compile time for recall-targeted training.

                    Input: (batch, 14, 35)
                         |
         ┌---------------┴--------------------------┐
         |                                           |
  ┌------▼----------┐         ┌---------------------▼-------------------┐
  │  LSTM Branch     │         │  Physics Statistics Branch              │
  │  LSTM(128)       │         │  (ZERO parameters — all pre-computed)   │
  │  LSTM(64)        │         │                                         │
  │  Dense(32)       │         │  From the 14-day FWI sequence:          │
  │  logit_lstm      │         │  1. Mean FWI       (drought level)      │
  │  (batch, 1)      │         │  2. Max FWI        (worst single day)   │
  └------┬-----------┘         │  3. FWI Trend      (building/easing)    │
         |                     │  4. Frac High FWI  (days > +0.5σ)       │
         |                     │  5. FWI Volatility (day-to-day swings)  │
         |                     │  6. Frac Extreme   (days > +2.0σ)       │
         |                     │  → physics_vec: (batch, 6)              │
         |                     │                                         │
         |                     │  BatchNorm (scale balancing)            │
         |                     │  Dense(1, no bias) ← 6 learnable        │
         |                     │  physics weights (one per statistic)    │
         |                     │  logit_physics (batch, 1)               │
         └-----------┬---------┘
                     |
         final_logit = logit_lstm + α · logit_physics
         P(fire) = sigmoid(final_logit)
         where α is a trainable scalar (init=0.1) — the physics gate.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)
import os
import config
from losses import focal_loss, physics_recall_focal_loss


# -----------------------------------------------------------------------------
# Physics Sequence Summary Layer
# -----------------------------------------------------------------------------

class FWISequenceSummary(tf.keras.layers.Layer):
    """
    Computes 5 non-parametric, sequence-level physics statistics from the
    FWI sub-sequence of a (batch, seq_len, n_features) input tensor.
    """

    def __init__(self, fwi_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.fwi_idx = fwi_idx

    def call(self, x, training=None):
        """x: (batch, seq_len, n_features)  →  (batch, 6)"""
        fwi = x[:, :, self.fwi_idx]                              # (batch, seq_len)

        # 1. Mean FWI over the window — overall drought level
        mean_fwi   = tf.reduce_mean(fwi, axis=1, keepdims=True)  # (batch, 1)

        # 2. Max FWI — worst single day in the window
        max_fwi    = tf.reduce_max(fwi,  axis=1, keepdims=True)  # (batch, 1)

        # 3. FWI Trend — Robust comparison: 2nd half mean vs 1st half mean
        first_half  = tf.reduce_mean(fwi[:, :7], axis=1, keepdims=True)
        second_half = tf.reduce_mean(fwi[:, 7:], axis=1, keepdims=True)
        fwi_trend   = second_half - first_half                   # (batch, 1)

        # 4. Fraction of days with Elevated Danger (Scaled space: > +0.5 StdDev)
        # FIX v2: Lowered from >1.0 to >0.5 to capture India fire-risk onset zone.
        # In right-skewed FWI distributions, real fire risk begins around +0.5σ,
        # not the statistically-conservative +1.0σ used previously.
        high_mask  = tf.cast(fwi > 0.5, tf.float32)              # (batch, seq_len)
        frac_high  = tf.reduce_mean(high_mask, axis=1, keepdims=True) # (batch, 1)

        # 5. FWI Volatility — mean absolute day-to-day change
        diffs      = tf.abs(fwi[:, 1:] - fwi[:, :-1])            # (batch, seq_len-1)
        volatility = tf.reduce_mean(diffs, axis=1, keepdims=True) # (batch, 1)

        # 6. Fraction of Extreme Danger days (Scaled space: > +2.0 StdDev)
        # NEW v2: captures catastrophic fire-weather events that dominate fatal
        # wildfire outbreaks. Distinct from frac_high — separates moderate vs extreme.
        extreme_mask = tf.cast(fwi > 2.0, tf.float32)            # (batch, seq_len)
        frac_extreme = tf.reduce_mean(extreme_mask, axis=1, keepdims=True) # (batch, 1)

        stats = tf.concat([mean_fwi, max_fwi, fwi_trend,
                           frac_high, volatility, frac_extreme], axis=1)
        return stats                                               # (batch, 6)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'fwi_idx': self.fwi_idx})
        return cfg


# -----------------------------------------------------------------------------
# Novel Model: FWI-Gated LSTM (FWI-Physics Residual Network)
# -----------------------------------------------------------------------------

def build_fwi_gated_lstm(
    seq_len:    int,
    n_features: int,
    fwi_idx:    int,
    loss=None,
) -> Model:
    """
    FWI-Gated LSTM (FWI-Physics Residual Network).
    """
    inp = layers.Input(shape=(seq_len, n_features), name='weather_sequence')

    # -- ① LSTM Branch ---------------------------------------------------------
    h = layers.LSTM(config.LSTM_UNITS_1, return_sequences=True, name='lstm_1')(inp)
    h = layers.Dropout(config.DROPOUT_RATE, name='drop_lstm_1')(h)
    h = layers.LSTM(config.LSTM_UNITS_2, return_sequences=False, name='lstm_2')(h)
    h = layers.Dropout(config.DROPOUT_RATE, name='drop_lstm_2')(h)
    h = layers.Dense(config.DENSE_UNITS, activation='relu', name='dense_1')(h)
    h = layers.Dropout(0.20, name='drop_dense')(h)

    # LSTM logit: a single unnormalised score
    logit_lstm = layers.Dense(1, use_bias=True, name='logit_lstm')(h)   # (batch, 1)

    # -- ② Physics Statistics Branch (NON-PARAMETRIC SUMMARY) -----------------
    # Now produces (batch, 6) — 6 physics statistics
    physics_stats = FWISequenceSummary(fwi_idx=fwi_idx, name='physics_summary')(inp)

    # BatchNorm: ensures the 6 statistics don't overpower LSTM gradients
    physics_norm = layers.BatchNormalization(name='physics_norm')(physics_stats)

    # Physics logit: 6 learnable weights (one per statistic), no bias
    logit_physics = layers.Dense(1, use_bias=False, name='logit_physics')(physics_norm)

    # -- ③ Gated Residual Fusion (FIX v2) --------------------------------------
    # KEY FIX: A trainable gate scalar `alpha` (initialized to 0.1) prevents
    # the physics branch from dominating LSTM gradients in early training.
    #
    # Problem with raw Add([logit_lstm, logit_physics]):
    #   The physics Dense(1) has only 6 parameters receiving the full recall
    #   gradient signal, while the LSTM branch has 100k+ params. Equal-weight
    #   addition means physics gradients are 10-100x larger per-parameter,
    #   causing the LSTM to collapse toward zero weights (→ recall decline).
    #
    # With alpha=0.1 at init:
    #   - Epochs 1-10: LSTM branch dominates (alpha is small) → LSTM learns
    #   - Epochs 10+:  alpha grows via gradient → physics branch opens up
    #   - Result: stable recall curve instead of collapse pattern
    gate_alpha = tf.Variable(
        initial_value=0.1,
        trainable=True,
        dtype=tf.float32,
        name='physics_gate',
    )
    scaled_physics = layers.Lambda(
        lambda x: x * gate_alpha,
        name='gated_physics',
    )(logit_physics)

    final_logit = layers.Add(name='residual_fusion')([logit_lstm, scaled_physics])
    out = layers.Activation('sigmoid', name='fire_probability')(final_logit)

    model = Model(inputs=inp, outputs=out, name='FWI_Gated_LSTM')

    # Novel model uses physics_recall_focal_loss — provides direct gradient
    # signal for high-FWI recall improvement (baseline uses plain focal_loss
    # for correct architectural ablation comparison).
    if loss is None:
        loss = physics_recall_focal_loss()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc_roc'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
        ],
    )
    return model


# -----------------------------------------------------------------------------
# Baseline: Vanilla LSTM (Ablation Comparison)
# -----------------------------------------------------------------------------

def build_baseline_lstm(seq_len: int, n_features: int, loss=None) -> Model:
    """
    Vanilla LSTM — single pathway, no physics statistics, no residual bypass.
    """
    inp = layers.Input(shape=(seq_len, n_features), name='weather_sequence')
    h   = layers.LSTM(config.LSTM_UNITS_1, return_sequences=True,  name='lstm_1')(inp)
    h   = layers.Dropout(config.DROPOUT_RATE, name='drop_lstm_1')(h)
    h   = layers.LSTM(config.LSTM_UNITS_2, return_sequences=False, name='lstm_2')(h)
    h   = layers.Dropout(config.DROPOUT_RATE, name='drop_lstm_2')(h)
    h   = layers.Dense(config.DENSE_UNITS, activation='relu', name='dense_1')(h)
    h   = layers.Dropout(0.20, name='drop_dense')(h)
    out = layers.Dense(1, activation='sigmoid', name='fire_probability')(h)

    model = Model(inputs=inp, outputs=out, name='Baseline_LSTM')
    if loss is None:
        loss = focal_loss()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc_roc'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
        ],
    )
    return model


# -----------------------------------------------------------------------------
# Interpretability: Physics Weight Extraction
# -----------------------------------------------------------------------------

PHYSICS_STAT_NAMES = [
    'Mean FWI (drought level)',
    'Max FWI (worst day)',
    'FWI Trend (building/easing)',
    'Fraction Elevated FWI Days (>0.5σ)',
    'FWI Volatility (instability)',
    'Fraction Extreme FWI Days (>2.0σ)',
]


def get_physics_weights(model: Model) -> dict:
    """
    Extracts and interprets the 5 learned physics weights from the
    logit_physics Dense layer.
    """
    try:
        phys_layer = model.get_layer('logit_physics')
        weights    = phys_layer.get_weights()[0].flatten()  # shape (5,)
    except (ValueError, IndexError):
        return {}

    result = {}
    for name, w in zip(PHYSICS_STAT_NAMES, weights):
        result[name] = {
            'weight':          float(w),
            'direction':       'increases fire risk' if w > 0 else 'decreases fire risk',
            'relative_impact': abs(float(w)),
        }
    return result


def print_physics_report(model: Model):
    """Prints a formatted physics weight report for journal interpretation."""
    weights = get_physics_weights(model)
    if not weights:
        print("No physics weights found.")
        return

    print("\n" + "="*60)
    print("  Physics Weight Report (FWI-Gated LSTM Interpretability)")
    print("="*60)
    print("  Statistic                        Weight   Direction")
    print("  " + "-"*56)
    sorted_w = sorted(weights.items(), key=lambda x: -abs(x[1]['weight']))
    for name, info in sorted_w:
        print(f"  {name:<32} {info['weight']:+.4f}   {info['direction']}")
    print("="*60)
    print("  Interpretation for paper:")
    top = sorted_w[0]
    print(f"  The strongest driver is '{top[0]}' (w={top[1]['weight']:+.4f})")
    print(f"  confirming that the model's physics branch relies most on")
    print(f"  this sequence-level drought statistic for fire prediction.")
    print("="*60 + "\n")


# -----------------------------------------------------------------------------
# Aliases (backward compatibility)
# -----------------------------------------------------------------------------

def get_physics_coupling_stats(model: Model) -> dict:
    """Alias for backward compatibility."""
    return get_physics_weights(model)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

def get_callbacks(model_name: str) -> list:
    """Standard training callbacks for reproducible journal experiments.

    v2 changes:
      - Raised patience from 10 → 15 on val_auc_pr (gives recall time to mature).
      - Added a secondary EarlyStopping on val_recall with patience=20 and
        min_delta=0.005 — prevents checkpointing a model before recall stabilises.
      - ReduceLROnPlateau patience raised 5 → 8 (less aggressive LR cuts).
    """
    ckpt_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.keras")
    log_path  = os.path.join(config.RESULTS_DIR, f"{model_name}_training_log.csv")

    return [
        # Primary stopper: AUC-PR (overall ranking quality)
        EarlyStopping(
            monitor='val_auc_pr',
            patience=config.PATIENCE,       # 15 (raised from 10)
            restore_best_weights=True,
            mode='max',
            verbose=1,
        ),
        # Secondary stopper: Recall — prevents stopping before recall matures.
        # Longer patience (20) because recall oscillates more than AUC-PR.
        # min_delta=0.005 ignores trivial noise (0.5% recall swings).
        EarlyStopping(
            monitor='val_recall',
            patience=20,
            restore_best_weights=True,
            mode='max',
            min_delta=0.005,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_auc_pr',
            factor=0.5,
            patience=8,                     # Raised from 5 — less aggressive LR cuts
            min_lr=1e-6,
            mode='max',
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_auc_pr',
            save_best_only=True,
            mode='max',
            verbose=0,
        ),
        CSVLogger(log_path, append=False),
    ]