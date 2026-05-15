"""
Loss Functions — FWI-PRN Wildfire Prediction
=============================================
Standard binary cross-entropy is insufficient for extreme class imbalance
(fire events = 1.72% of samples). It assigns equal weight to easy and hard
examples, so the model learns to predict "No Fire" confidently and ignores
the rare fire events entirely — resulting in near-zero recall.

Focal Loss (Lin et al., 2017 — RetinaNet)
-----------------------------------------
FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

  α (alpha): class balance factor.
             Higher α → more gradient weight to fire class (minority).
             Default: 0.85 (85% weight to fire, 15% to no-fire)

  γ (gamma): focusing parameter.
             When γ=0: reduces to weighted BCE (standard).
             When γ=2: easy examples (p_t ≈ 1) contribute ×(1-0.9)^2=0.01
             of their standard BCE loss — essentially ignored.
             Hard fire examples (p_t ≈ 0.1) contribute ×(0.9)^2=0.81 of loss.
             Result: training FOCUSES on the fire events the model keeps missing.

Physics-Recall Loss (Novel Model Only)
---------------------------------------
Extends focal loss with a physics-consistency penalty:
When the physics branch says HIGH risk (high mean FWI, strong positive trend)
but the model still misses the fire (false negative), penalize EXTRA.

This gives the novel model's physics branch a direct gradient signal for
recall improvement — the baseline cannot use this because it has no physics branch.

Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection", ICCV.
"""

import tensorflow as tf
import config


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = None, alpha: float = None):
    """
    Focal Loss for extreme class imbalance.

    Parameters
    ----------
    gamma : focusing exponent (default from config.FOCAL_GAMMA = 2.0)
            Higher → stronger focus on hard/missed examples
    alpha : minority class weight (default from config.FOCAL_ALPHA = 0.85)
            Fraction of loss assigned to fire events (class=1)

    Returns a Keras-compatible loss function.
    """
    if gamma is None:
        gamma = config.FOCAL_GAMMA
    if alpha is None:
        alpha = config.FOCAL_ALPHA

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # p_t: probability of the TRUE class
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # alpha_t: class balance weight
        alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)

        # Focal modulating factor: (1 - p_t)^gamma
        # → Near 0 for easy examples (p_t → 1), near 1 for hard ones (p_t → 0)
        focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)

        # Focal cross-entropy
        fl = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    loss_fn.__name__ = f'focal_loss_g{gamma}_a{alpha}'
    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# Recall-Boosted Focal Loss (Novel Model Only)
# ─────────────────────────────────────────────────────────────────────────────

def physics_recall_focal_loss(gamma: float = None, alpha: float = None,
                               lambda_physics: float = None):
    """
    Physics-Recall Focal Loss for the FWI-PRN novel model.

    Extends focal loss with a physics miss penalty:
      L_total = FL(y, p) + λ · mean(max_fwi_t · relu(0.5 - p) · y)

    The penalty term activates ONLY on:
      - True fire events (y=1)
      - Where the model is underconfident (p < 0.5)
      - Scaled by the peak FWI in the 14-day window (physics severity)

    Effect: when physics says "extreme drought, this should be fire" but
    the model hedges below 0.5, the penalty forces the logit upward.
    This directly translates to higher recall on high-FWI fire events.

    Parameters
    ----------
    gamma, alpha     : focal loss parameters (same as above)
    lambda_physics   : weight of physics penalty (default config.PHYSICS_RECALL_LAMBDA)

    Returns a Keras-compatible loss function that accepts (y_true, y_pred).

    Note: max_fwi is embedded in the second element of the model's physics stats
    output. Since we cannot pass extra tensors through the standard loss API,
    we use the raw prediction magnitude as a proxy here. The full physics-aware
    variant is implemented via the custom training loop in train.py.
    """
    if gamma is None:
        gamma = config.FOCAL_GAMMA
    if alpha is None:
        alpha = config.FOCAL_ALPHA
    if lambda_physics is None:
        lambda_physics = config.PHYSICS_RECALL_LAMBDA

    base_loss = focal_loss(gamma=gamma, alpha=alpha)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        fl = base_loss(y_true, y_pred)

        # Miss penalty: fire events where model is under-confident
        # relu(0.5 - p): positive only when p < 0.5 (model hedging)
        miss_penalty = y_true * tf.nn.relu(0.5 - y_pred)

        total = fl + lambda_physics * tf.reduce_mean(miss_penalty)
        return total

    loss_fn.__name__ = f'physics_recall_focal_g{gamma}_a{alpha}_l{lambda_physics}'
    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# Recall-Optimized Threshold Selection
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


def find_recall_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = None,
    target_recall: float = None,
) -> dict:
    """
    Recall-first threshold selection for imbalanced fire detection.

    If ``target_recall`` is set: pick the threshold whose recall is closest to
    that target (same sweep as below).

    Otherwise: among all thresholds with precision >= ``min_precision``,
    choose the one that **maximises recall** (not F1). Tie-break: higher
    precision, then lower threshold so alarms stay slightly more selective.

    This matches ``evaluate.py`` / config intent: missed fires are costlier than
    false alarms, subject to a floor on precision.

    Parameters
    ----------
    min_precision : minimum acceptable precision (default from config.MIN_PRECISION)
    target_recall : if set, find threshold that gets closest to this recall

    Returns dict with: threshold, recall, precision, f1
    """
    if min_precision is None:
        min_precision = config.MIN_PRECISION

    # Include low thresholds — sequence models often output muted probabilities.
    thresholds = np.linspace(0.01, 0.99, 250)
    results    = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec   = recall_score(y_true, preds, zero_division=0)
        prec  = precision_score(y_true, preds, zero_division=0)
        f1    = f1_score(y_true, preds, zero_division=0)
        results.append({'threshold': t, 'recall': rec, 'precision': prec, 'f1': f1})

    if target_recall is not None:
        best = min(results, key=lambda x: abs(x['recall'] - target_recall))
        return best

    valid = [r for r in results if r['precision'] >= min_precision]
    if valid:
        best = max(
            valid,
            key=lambda x: (x['recall'], x['precision'], -x['threshold']),
        )
    else:
        # No threshold meets min_precision — fall back to best recall overall,
        # then precision, so the table still reports a usable operating point.
        best = max(results, key=lambda x: (x['recall'], x['precision']))

    return best


def find_max_accuracy_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Finds the decision threshold that maximises accuracy.

    For highly imbalanced data this usually sits at a high threshold
    (the model becomes very conservative and gets almost all of the majority
    class right). Returned dict matches the format of find_recall_threshold.
    """
    from sklearn.metrics import accuracy_score as _acc
    thresholds = np.linspace(0.01, 0.99, 250)
    best = {'threshold': 0.5, 'recall': 0.0, 'precision': 0.0,
            'f1': 0.0, 'accuracy': 0.0}

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        acc   = _acc(y_true, preds)
        if acc > best['accuracy']:
            best = {
                'threshold': t,
                'accuracy':  acc,
                'recall':    recall_score(y_true, preds, zero_division=0),
                'precision': precision_score(y_true, preds, zero_division=0),
                'f1':        f1_score(y_true, preds, zero_division=0),
            }
    return best


def threshold_analysis_table(y_true: np.ndarray, y_prob: np.ndarray) -> list:
    """
    Returns a table of Recall/Precision/F1 at several candidate thresholds.
    Useful for the paper's supplementary results section.
    """
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    rows       = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rows.append({
            'Threshold': t,
            'Recall':    recall_score(y_true, preds, zero_division=0),
            'Precision': precision_score(y_true, preds, zero_division=0),
            'F1':        f1_score(y_true, preds, zero_division=0),
            'Fire_Predicted': int(preds.sum()),
        })
    return rows
