"""
Evaluation & Comparison — FWI-Gated LSTM vs Vanilla LSTM
=========================================================
Recall-centric evaluation for extreme class imbalance (fire ≈ 1.72%).

Key Design Decisions
---------------------
1. Threshold search: sweep [0.01, 0.99] and find the threshold that
   maximises RECALL subject to a minimum precision constraint.
2. AUC-PR is the primary metric for imbalanced data (not AUC-ROC).
3. All comparison plots include both models side-by-side.
4. Physics weight bar chart shows what the novel model learned.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import config
from losses import find_recall_threshold, threshold_analysis_table

matplotlib.rcParams.update({
    'font.family':  'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        120,
})

# Colour palette — consistent across all plots
C_NOVEL    = '#E63946'   # crimson  — FWI-Gated LSTM
C_BASELINE = '#457B9D'   # steel    — Vanilla LSTM
C_NEUTRAL  = '#888888'   # grey     — reference lines


# --- Known Baseline Results (from paper) -------------------------------------

BASELINES = {
    'Random Forest': {'accuracy': 0.852, 'precision': 0.830,
                      'recall':   0.875, 'f1':        0.845},
    'XGBoost':       {'accuracy': 0.848, 'precision': 0.825,
                      'recall':   0.850, 'f1':        0.838},
    'Deep ANN':      {'accuracy': 0.825, 'precision': 0.780,
                      'recall':   0.885, 'f1':        0.810},
}


# --- Threshold Search ---------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Maximises recall subject to MIN_PRECISION constraint."""
    result = find_recall_threshold(y_true, y_prob,
                                   min_precision=config.MIN_PRECISION)
    return result['threshold']


# --- Full Metrics -------------------------------------------------------------

def compute_metrics(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    threshold:  float = None,
    model_name: str = "Model",
) -> dict:
    """
    Computes all evaluation metrics with recall-optimised threshold search.
    """
    if threshold is None:
        best      = find_recall_threshold(y_true, y_prob,
                                          min_precision=config.MIN_PRECISION)
        threshold = best['threshold']

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'model':            model_name,
        'threshold':        threshold,
        'accuracy':         accuracy_score(y_true, y_pred),
        'precision':        precision_score(y_true, y_pred, zero_division=0),
        'recall':           recall_score(y_true, y_pred, zero_division=0),
        'f1':               f1_score(y_true, y_pred, zero_division=0),
        'auc_roc':          roc_auc_score(y_true, y_prob),
        'auc_pr':           average_precision_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_prob':           y_prob,
    }

    cm = metrics['confusion_matrix']
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) * 100 if (cm[1, 0] + cm[1, 1]) > 0 else 0

    print(f"\n{'='*65}")
    print(f"  {model_name} — Recall-Optimised Evaluation")
    print(f"{'='*65}")
    print(f"  Threshold (recall-optimal) : {threshold:.3f}  "
          f"[min_precision={config.MIN_PRECISION}]")
    print(f"  Accuracy                   : {metrics['accuracy']:.4f}")
    print(f"  Precision                  : {metrics['precision']:.4f}")
    print(f"  Recall  ← PRIMARY          : {metrics['recall']:.4f}")
    print(f"  F1-Score                   : {metrics['f1']:.4f}")
    print(f"  AUC-ROC                    : {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR  ← imbalanced data  : {metrics['auc_pr']:.4f}")
    print(f"\n  Confusion Matrix (threshold={threshold:.2f}):")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"  False Negative Rate (missed fires): {fnr:.1f}%")
    print(f"{'='*65}\n")

    print(f"  Threshold Sweep ({model_name}):")
    print(f"  {'Threshold':>10} {'Recall':>8} {'Precision':>10} "
          f"{'F1':>8} {'Fires Pred':>12}")
    print("  " + "-"*52)
    for row in threshold_analysis_table(y_true, y_prob):
        marker = " ← optimal" if abs(row['Threshold'] - threshold) < 0.03 else ""
        print(f"  {row['Threshold']:>10.2f} {row['Recall']:>8.4f} "
              f"{row['Precision']:>10.4f} {row['F1']:>8.4f} "
              f"{row['Fire_Predicted']:>12,}{marker}")
    print()

    return metrics


# --- Comparison Table ---------------------------------------------------------

def build_comparison_table(novel_metrics: dict,
                            baseline_metrics: dict = None,
                            novel_metrics_acc: dict = None,
                            novel_metrics_op: dict = None) -> pd.DataFrame:
    """Full comparison: RF / XGBoost / ANN / Vanilla LSTM / FWI-Gated LSTM.

    Parameters
    ----------
    novel_metrics      : Metrics at the recall-optimal threshold.
    baseline_metrics   : Vanilla LSTM metrics.
    novel_metrics_acc  : Optional — FWI-Gated LSTM at max-accuracy threshold.
    novel_metrics_op   : Optional — FWI-Gated LSTM at max-recall threshold (t=0.10).
    """
    rows = []
    for name, m in BASELINES.items():
        rows.append({
            'Model':     name,
            'Threshold': '0.50',
            'Accuracy':  f"{m['accuracy']:.3f}",
            'Precision': f"{m['precision']:.3f}",
            'Recall ↑':  f"{m['recall']:.3f}",
            'F1-Score':  f"{m['f1']:.3f}",
            'AUC-ROC':   '—',
            'AUC-PR ↑':  '—',
        })

    if baseline_metrics is not None:
        rows.append({
            'Model':     'Vanilla LSTM',
            'Threshold': f"{baseline_metrics['threshold']:.3f}",
            'Accuracy':  f"{baseline_metrics['accuracy']:.3f}",
            'Precision': f"{baseline_metrics['precision']:.3f}",
            'Recall ↑':  f"{baseline_metrics['recall']:.3f}",
            'F1-Score':  f"{baseline_metrics['f1']:.3f}",
            'AUC-ROC':   f"{baseline_metrics['auc_roc']:.3f}",
            'AUC-PR ↑':  f"{baseline_metrics['auc_pr']:.3f}",
        })

    # Main novel-model row (recall-optimal threshold)
    rows.append({
        'Model':     '★ FWI-Gated LSTM (Proposed)',
        'Threshold': f"{novel_metrics['threshold']:.3f}",
        'Accuracy':  f"{novel_metrics['accuracy']:.3f}",
        'Precision': f"{novel_metrics['precision']:.3f}",
        'Recall ↑':  f"{novel_metrics['recall']:.3f}",
        'F1-Score':  f"{novel_metrics['f1']:.3f}",
        'AUC-ROC':   f"{novel_metrics['auc_roc']:.3f}",
        'AUC-PR ↑':  f"{novel_metrics['auc_pr']:.3f}",
    })

    # Max-accuracy row — threshold that maximises accuracy for the novel model
    if novel_metrics_acc is not None:
        rows.append({
            'Model':     '★ FWI-Gated LSTM — Max Accuracy',
            'Threshold': f"{novel_metrics_acc['threshold']:.3f}",
            'Accuracy':  f"{novel_metrics_acc['accuracy']:.3f}",
            'Precision': f"{novel_metrics_acc['precision']:.3f}",
            'Recall ↑':  f"{novel_metrics_acc['recall']:.3f}",
            'F1-Score':  f"{novel_metrics_acc['f1']:.3f}",
            'AUC-ROC':   f"{novel_metrics['auc_roc']:.3f}",  # threshold-independent
            'AUC-PR ↑':  f"{novel_metrics['auc_pr']:.3f}",  # threshold-independent
        })

    # Max-recall row — t=0.10 operating point (highest fire detection rate)
    if novel_metrics_op is not None:
        rows.append({
            'Model':     '★ FWI-Gated LSTM — Max Recall (t=0.10)',
            'Threshold': f"{novel_metrics_op['threshold']:.2f}",
            'Accuracy':  f"{novel_metrics_op['accuracy']:.3f}",
            'Precision': f"{novel_metrics_op['precision']:.3f}",
            'Recall ↑':  f"{novel_metrics_op['recall']:.3f}",
            'F1-Score':  f"{novel_metrics_op['f1']:.3f}",
            'AUC-ROC':   f"{novel_metrics['auc_roc']:.3f}",  # threshold-independent
            'AUC-PR ↑':  f"{novel_metrics['auc_pr']:.3f}",  # threshold-independent
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*65)
    print("  MODEL COMPARISON TABLE")
    print("="*65)
    print(df.to_string(index=False))
    print("="*65)
    return df


def print_detailed_comparison(novel_metrics: dict, baseline_metrics: dict):
    """Side-by-side metric delta between FWI-Gated LSTM and Vanilla LSTM."""
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']

    print("\n" + "="*65)
    print("  FWI-Gated LSTM vs Vanilla LSTM — Head-to-Head")
    print("="*65)
    print(f"  {'Metric':<14} {'FWI-Gated':>12} {'Vanilla':>12} {'Δ':>10}")
    print("  " + "-"*52)
    for key, label in zip(keys, labels):
        n_val = novel_metrics[key]
        b_val = baseline_metrics[key]
        delta = n_val - b_val
        star  = " ★" if delta > 0 else ("  " if delta == 0 else " ▼")
        print(f"  {label:<14} {n_val:>12.4f} {b_val:>12.4f} {delta:>+10.4f}{star}")
    print("="*65 + "\n")


# --- Plotting: Training History -----------------------------------------------

def plot_training_history(history, model_name: str = "Model"):
    """
    Plots Loss, Recall, and AUC-PR curves for training and validation.
    Because validation uses real distribution (pre-SMOTE), val recall
    should now track test recall closely.
    """
    metrics_to_plot = [
        ('loss',    'Focal Loss'),
        ('recall',  'Recall (Fire Detection Rate)'),
        ('auc_pr',  'AUC-PR (Primary Metric)'),
    ]
    color = C_NOVEL if 'Gated' in model_name else C_BASELINE
    label = 'FWI-Gated LSTM' if 'Gated' in model_name else 'Vanilla LSTM'

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{label} — Training History", fontsize=13, fontweight='bold')

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        if metric in history.history:
            ax.plot(history.history[metric],
                    lw=2.5, color=color, label='Train')
        val_key = f'val_{metric}'
        if val_key in history.history:
            ax.plot(history.history[val_key],
                    lw=2, color=color, linestyle='--', alpha=0.85, label='Validation')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"{model_name}_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# --- Plotting: ROC & PR Curves ------------------------------------------------

def plot_roc_pr_curves(y_test, y_prob_novel, y_prob_baseline=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ROC & Precision-Recall Curves — FWI-Gated LSTM vs Vanilla LSTM",
                 fontsize=13, fontweight='bold')

    # ROC
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_test, y_prob_novel)
    auc_n       = roc_auc_score(y_test, y_prob_novel)
    ax.plot(fpr, tpr, lw=2.5, color=C_NOVEL,
            label=f'FWI-Gated LSTM (AUC={auc_n:.3f})')
    if y_prob_baseline is not None:
        fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_baseline)
        auc_b = roc_auc_score(y_test, y_prob_baseline)
        ax.plot(fpr_b, tpr_b, lw=2, color=C_BASELINE, linestyle='--',
                label=f'Vanilla LSTM (AUC={auc_b:.3f})')
    ax.plot([0, 1], [0, 1], '--', color=C_NEUTRAL, lw=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve'); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Precision-Recall (primary for imbalanced data)
    ax = axes[1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob_novel)
    ap_n          = average_precision_score(y_test, y_prob_novel)
    ax.plot(rec, prec, lw=2.5, color=C_NOVEL,
            label=f'FWI-Gated LSTM (AUC-PR={ap_n:.3f})')
    if y_prob_baseline is not None:
        prec_b, rec_b, _ = precision_recall_curve(y_test, y_prob_baseline)
        ap_b = average_precision_score(y_test, y_prob_baseline)
        ax.plot(rec_b, prec_b, lw=2, color=C_BASELINE, linestyle='--',
                label=f'Vanilla LSTM (AUC-PR={ap_b:.3f})')
    fire_rate = y_test.mean()
    ax.axhline(fire_rate, color=C_NEUTRAL, linestyle=':', lw=1.2,
               label=f'No-Skill (fire rate={fire_rate:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve (PRIMARY — imbalanced data)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# --- Plotting: Threshold Tradeoff ---------------------------------------------

def plot_threshold_tradeoff(y_test, y_prob_novel, y_prob_baseline=None,
                             optimal_thresh=None):
    thresholds = np.linspace(0.05, 0.80, 200)

    def get_curves(y_prob):
        recalls, precisions = [], []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            recalls.append(recall_score(y_test, preds, zero_division=0))
            precisions.append(precision_score(y_test, preds, zero_division=0))
        return np.array(recalls), np.array(precisions)

    rec_n, prec_n = get_curves(y_prob_novel)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Recall & Precision vs Decision Threshold",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(thresholds, rec_n,  lw=2.5, color=C_NOVEL,
            label='FWI-Gated Recall')
    ax.plot(thresholds, prec_n, lw=2.5, color=C_NOVEL, linestyle='--',
            label='FWI-Gated Precision')
    if y_prob_baseline is not None:
        rec_b, prec_b = get_curves(y_prob_baseline)
        ax.plot(thresholds, rec_b,  lw=2, color=C_BASELINE, alpha=0.75,
                label='Vanilla Recall')
        ax.plot(thresholds, prec_b, lw=2, color=C_BASELINE, linestyle='--',
                alpha=0.75, label='Vanilla Precision')
    if optimal_thresh is not None:
        ax.axvline(optimal_thresh, color='green', lw=1.5, linestyle=':',
                   label=f'Optimal T={optimal_thresh:.2f}')
    ax.axvline(0.5, color=C_NEUTRAL, lw=1, linestyle=':', label='T=0.50 (default)')
    ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
    ax.set_title('Recall & Precision vs Threshold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(rec_n, prec_n, lw=2.5, color=C_NOVEL, label='FWI-Gated LSTM')
    if y_prob_baseline is not None:
        ax.plot(rec_b, prec_b, lw=2, color=C_BASELINE, linestyle='--',
                alpha=0.75, label='Vanilla LSTM')
    ax.set_xlabel('Recall (Fire Detection Rate)'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "threshold_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# --- Plotting: Comparison Bar -------------------------------------------------

def plot_comparison_bar(comparison_df: pd.DataFrame):
    """
    Grouped bar chart comparing all models on Accuracy / Precision / Recall / F1.
    The proposed FWI-Gated LSTM bar is highlighted in crimson.
    """
    metrics_cols = ['Accuracy', 'Precision', 'Recall ↑', 'F1-Score']
    models       = comparison_df['Model'].tolist()
    bar_colors   = []
    for m in models:
        if 'FWI-Gated' in m or '★' in m:
            bar_colors.append(C_NOVEL)
        elif 'Vanilla' in m:
            bar_colors.append(C_BASELINE)
        else:
            bar_colors.append('#2A9D8F')

    data = {}
    for col in metrics_cols:
        vals = []
        for v in comparison_df[col]:
            try:    vals.append(float(v))
            except: vals.append(np.nan)
        data[col] = vals

    x   = np.arange(len(models))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(16, 6))
    metric_colors = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']

    for i, (metric, mc) in enumerate(zip(metrics_cols, metric_colors)):
        bars = ax.bar(x + i * w, data[metric], w, label=metric,
                      color=mc, alpha=0.88, edgecolor='white', zorder=3)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7,
                        fontweight='bold')

    # Highlight proposed model column
    proposed_idx = next((i for i, m in enumerate(models)
                         if 'FWI-Gated' in m or '★' in m), None)
    if proposed_idx is not None:
        ax.axvspan(proposed_idx - 0.12, proposed_idx + 4 * w + 0.02,
                   alpha=0.06, color=C_NOVEL, zorder=0)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0.0, 1.08)
    ax.set_title('Model Performance Comparison — Wildfire Prediction (Indian Subcontinent)',
                 fontsize=13, fontweight='bold', pad=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.axhline(0.875, color='#E9C46A', linestyle=':', lw=1.5, alpha=0.7,
               label='RF Recall baseline (0.875)')

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "model_comparison_bar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# --- Plotting: Confusion Matrices Side-by-Side --------------------------------

def plot_confusion_matrices_side_by_side(cm_novel: np.ndarray,
                                          cm_baseline: np.ndarray):
    """
    Side-by-side confusion matrices for FWI-Gated LSTM and Vanilla LSTM.
    Highlights the False Negative (missed fires) cell prominently.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Confusion Matrices — FWI-Gated LSTM vs Vanilla LSTM",
                 fontsize=13, fontweight='bold')

    for ax, cm, name, cmap in zip(
        axes,
        [cm_novel, cm_baseline],
        ['FWI-Gated LSTM (Proposed)', 'Vanilla LSTM (Baseline)'],
        ['Reds', 'Blues'],
    ):
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        labels = ['No Fire', 'Fire']
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(name, fontsize=11, fontweight='bold')

        thresh = cm.max() / 2.0
        cell_labels = {
            (0, 0): 'TN', (0, 1): 'FP\n(False\nAlarm)',
            (1, 0): 'FN\n(Missed\nFire!)', (1, 1): 'TP\n(Detected)',
        }
        for i in range(2):
            for j in range(2):
                tag  = cell_labels[(i, j)]
                text = f'{cm[i,j]:,}\n{tag}'
                col  = 'white' if cm[i, j] > thresh else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=10, fontweight='bold', color=col)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# Single confusion matrix (kept for notebook compatibility)
def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Model"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im      = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    labels  = ['No Fire', 'Fire']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold')
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# --- Plotting: Physics Weights (Novel Model Only) ----------------------------

def plot_physics_weights(model):
    """
    Horizontal bar chart of the 5 learned physics weights.
    Positive weight = that FWI statistic increases predicted fire risk.
    """
    try:
        from fwi_gated_lstm import PHYSICS_STAT_NAMES, get_physics_weights
        weights_dict = get_physics_weights(model)
    except Exception:
        return None

    if not weights_dict:
        return None

    names   = list(weights_dict.keys())
    weights = [weights_dict[n]['weight'] for n in names]
    colors  = [C_NOVEL if w > 0 else C_BASELINE for w in weights]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, weights, color=colors, edgecolor='white', height=0.55)
    ax.axvline(0, color='black', lw=0.8)
    for bar, w in zip(bars, weights):
        xpos = w + 0.002 if w >= 0 else w - 0.002
        ha   = 'left' if w >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{w:+.4f}', va='center', ha=ha, fontsize=10, fontweight='bold')
    ax.set_xlabel('Learned Weight (positive → increases fire risk)', fontsize=11)
    ax.set_title('FWI-Gated LSTM — Physics Branch Weights\n'
                 '(Which drought statistics drive fire prediction)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.25)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_NOVEL,    label='Increases fire risk'),
        Patch(facecolor=C_BASELINE, label='Decreases fire risk'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "physics_weights.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# --- Save Results CSV ---------------------------------------------------------

def save_results_csv(metrics_novel: dict, metrics_baseline: dict,
                     comparison_df: pd.DataFrame):
    path = os.path.join(config.RESULTS_DIR, "evaluation_results.csv")
    comparison_df.to_csv(path, index=False)

    rows = []
    for m in [metrics_novel, metrics_baseline]:
        rows.append({k: v for k, v in m.items()
                     if k not in ['confusion_matrix', 'y_prob']})
    pd.DataFrame(rows).to_csv(
        os.path.join(config.RESULTS_DIR, "numeric_metrics.csv"), index=False
    )
    print(f"[Results] Saved to {config.RESULTS_DIR}")
