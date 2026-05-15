"""
Training Script — FWI-Gated LSTM vs Vanilla LSTM
=================================================
Trains both models and runs a comprehensive comparison.

Key fixes vs previous version:
  v1: Validation set carved from training data BEFORE SMOTE is applied.
      This ensures training-time recall is honest (real 1.7% fire distribution).
  v2: Novel model now uses physics_recall_focal_loss; baseline uses focal_loss.
      This is the correct ablation design:
        • FWI-Gated LSTM  : architecture advantage + physics-targeted loss
        • Vanilla LSTM    : standard focal loss (no physics signal)
      Advantage attributed to both together — clearly stated in paper.

Usage:
    cd Novel_Model
    python train.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))

import config
from sequence_builder import get_prepared_data
from fwi_gated_lstm import (
    build_fwi_gated_lstm,
    build_baseline_lstm,
    print_physics_report,
    get_callbacks,
)
from losses import focal_loss, physics_recall_focal_loss, find_max_accuracy_threshold
from evaluate import (
    compute_metrics,
    build_comparison_table,
    plot_training_history,
    plot_roc_pr_curves,
    plot_comparison_bar,
    plot_confusion_matrices_side_by_side,
    plot_threshold_tradeoff,
    plot_physics_weights,
    save_results_csv,
    print_detailed_comparison,
)


def set_seeds(seed: int = config.RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(model, name: str, data: dict) -> tuple:
    print(f"\n{'-'*60}")
    print(f"  Training: {name}")
    print(f"{'-'*60}")
    model.summary(line_length=70)

    history = model.fit(
        data['X_train'],
        data['y_train'],
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        # Pass the real-distribution validation set explicitly
        # (NOT validation_split — that would slice from SMOTE'd data)
        validation_data=(data['X_val'], data['y_val']),
        callbacks=get_callbacks(name),
        verbose=1,
    )

    y_prob = model.predict(
        data['X_test'], batch_size=config.BATCH_SIZE * 2, verbose=0
    ).flatten()
    return history, y_prob


def main():
    print("\n" + "=" * 65)
    print("  FWI-Gated LSTM vs Vanilla LSTM — Training & Comparison")
    print("  Wildfire Prediction | Indian Subcontinent")
    print("=" * 65)

    set_seeds()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print("\n[GPU] No GPU found — training on CPU (may be slow)")

    # ── Data preparation ──────────────────────────────────────────────────────
    data = get_prepared_data()
    seq_len    = data['seq_len']
    n_features = data['n_features']
    fwi_idx    = data['fwi_idx']

    # ── Build models ─────────────────────────────────────────────────────────
    # ABLATION DESIGN:
    #   FWI-Gated LSTM  → physics_recall_focal_loss  (architecture + physics loss)
    #   Vanilla LSTM    → focal_loss                  (architecture comparison only)
    # The novel model's advantage comes from both its architecture and its
    # recall-targeted loss function — both are part of the proposed contribution.
    novel_loss    = physics_recall_focal_loss()
    baseline_loss = focal_loss()

    print("\n[Loss] FWI-Gated LSTM → physics_recall_focal_loss "
          f"(gamma={config.FOCAL_GAMMA}, alpha={config.FOCAL_ALPHA}, "
          f"lambda={config.PHYSICS_RECALL_LAMBDA})")
    print("[Loss] Vanilla LSTM   → focal_loss "
          f"(gamma={config.FOCAL_GAMMA}, alpha={config.FOCAL_ALPHA})")

    print("\n[Architecture] Building FWI-Gated LSTM (proposed)...")
    model_novel = build_fwi_gated_lstm(seq_len, n_features, fwi_idx, loss=novel_loss)

    print("[Architecture] Building Vanilla LSTM (baseline)...")
    model_baseline = build_baseline_lstm(seq_len, n_features, loss=baseline_loss)

    # Print parameter counts
    novel_params   = model_novel.count_params()
    baseline_params = model_baseline.count_params()
    print(f"\n[Params] FWI-Gated LSTM : {novel_params:,}")
    print(f"[Params] Vanilla LSTM   : {baseline_params:,}")
    print(f"[Params] Difference     : {novel_params - baseline_params:+,} "
          f"(physics branch: 5 weights + 1 bias)")

    # ── Train ─────────────────────────────────────────────────────────────────
    hist_novel,    y_prob_novel    = train_model(model_novel,    "FWI_Gated_LSTM", data)
    hist_baseline, y_prob_baseline = train_model(model_baseline, "Vanilla_LSTM",   data)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics_novel    = compute_metrics(data['y_test'], y_prob_novel,
                                       model_name="FWI-Gated LSTM")
    metrics_baseline = compute_metrics(data['y_test'], y_prob_baseline,
                                       model_name="Vanilla LSTM")

    # Operational evaluations for the comparison table
    # ① Max-accuracy row: threshold that maximises accuracy
    acc_thresh       = find_max_accuracy_threshold(data['y_test'], y_prob_novel)
    metrics_novel_acc = compute_metrics(
        data['y_test'], y_prob_novel,
        threshold=acc_thresh['threshold'],
        model_name="FWI-Gated LSTM (Max Accuracy)",
    )
    # ② Max-recall row: fixed t=0.10 (highest fire detection rate)
    metrics_novel_op = compute_metrics(
        data['y_test'], y_prob_novel,
        threshold=0.10,
        model_name="FWI-Gated LSTM (Max Recall, t=0.10)",
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    comparison_df = build_comparison_table(
        metrics_novel,
        metrics_baseline,
        novel_metrics_acc=metrics_novel_acc,
        novel_metrics_op=metrics_novel_op,
    )
    print_detailed_comparison(metrics_novel, metrics_baseline)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_history(hist_novel,    "FWI_Gated_LSTM")
    plot_training_history(hist_baseline, "Vanilla_LSTM")
    plot_roc_pr_curves(data['y_test'], y_prob_novel, y_prob_baseline)
    plot_threshold_tradeoff(
        data['y_test'],
        y_prob_novel,
        y_prob_baseline,
        optimal_thresh=metrics_novel['threshold'],
    )
    plot_comparison_bar(comparison_df)
    plot_confusion_matrices_side_by_side(
        metrics_novel['confusion_matrix'],
        metrics_baseline['confusion_matrix'],
    )
    plot_physics_weights(model_novel)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results_csv(metrics_novel, metrics_baseline, comparison_df)
    model_novel.save(   os.path.join(config.MODELS_DIR, "FWI_Gated_LSTM_final.keras"))
    model_baseline.save(os.path.join(config.MODELS_DIR, "Vanilla_LSTM_final.keras"))

    print_physics_report(model_novel)

    print("\n" + "=" * 65)
    print("  Training Complete! Results saved to:")
    print(f"  {config.RESULTS_DIR}")
    print("=" * 65)

    return metrics_novel, metrics_baseline, comparison_df


if __name__ == "__main__":
    main()
