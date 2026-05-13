"""
evaluate.py  —  Recall-centric metrics & plots for the ST-GNN paper.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
import config


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def find_recall_threshold(y_true, y_prob, min_precision=config.MIN_PRECISION):
    """Sweep thresholds; return the one maximising recall above min_precision."""
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()
    # Guard against NaN predictions (can occur with unstable GCN gradients)
    nan_count = np.isnan(y_prob).sum()
    if nan_count > 0:
        print(f"  [WARNING] {nan_count} NaN predictions replaced with 0.0")
        y_prob = np.nan_to_num(y_prob, nan=0.0)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_recall, best_thresh = 0.0, config.RECALL_THRESHOLD
    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_thresh = float(t)
    return best_thresh


def compute_metrics(y_true, y_prob, model_name="Model"):
    """Full metric suite for multi-node (Batch, Nodes, 1) outputs."""
    y_flat   = y_true.flatten().astype(int)
    p_flat   = np.nan_to_num(y_prob.flatten(), nan=0.0)  # NaN-safe

    thresh   = find_recall_threshold(y_flat, p_flat)
    y_pred   = (p_flat >= thresh).astype(int)

    cm = confusion_matrix(y_flat, y_pred)

    try:
        auc_roc = roc_auc_score(y_flat, p_flat)
        auc_pr  = average_precision_score(y_flat, p_flat)
    except Exception:
        auc_roc = auc_pr = 0.0

    metrics = {
        "model_name": model_name,
        "threshold":  thresh,
        "accuracy":   accuracy_score(y_flat, y_pred),
        "precision":  precision_score(y_flat, y_pred, zero_division=0),
        "recall":     recall_score(y_flat, y_pred, zero_division=0),
        "f1":         f1_score(y_flat, y_pred, zero_division=0),
        "auc_roc":    auc_roc,
        "auc_pr":     auc_pr,
        "confusion_matrix": cm,
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Optimal Threshold : {thresh:.3f}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}  <-- primary metric")
    print(f"  F1-Score          : {metrics['f1']:.4f}")
    print(f"  AUC-ROC           : {auc_roc:.4f}")
    print(f"  AUC-PR            : {auc_pr:.4f}  <-- primary metric (imbalanced)")
    print(f"  Confusion Matrix  :\n{cm}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {"novel": "#E63946", "baseline": "#457B9D", "accent": "#F4A261"}
FIGSIZE = (10, 5)


def plot_training_history(history, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight='bold')

    axes[0].plot(history.history.get('loss', []),     label='Train Loss', color=PALETTE["novel"])
    axes[0].plot(history.history.get('val_loss', []), label='Val Loss',   color=PALETTE["baseline"], linestyle='--')
    axes[0].set_title("Loss (Focal)"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history.get('recall', []),     label='Train Recall', color=PALETTE["novel"])
    axes[1].plot(history.history.get('val_recall', []), label='Val Recall',   color=PALETTE["baseline"], linestyle='--')
    axes[1].set_title("Recall"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"{model_name}_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_roc_pr_curves(y_true, y_prob_novel, y_prob_baseline):
    y_flat  = y_true.flatten().astype(int)
    p_novel = y_prob_novel.flatten()
    p_base  = y_prob_baseline.flatten()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle("ST-GNN vs Baseline — ROC and PR Curves", fontsize=14, fontweight='bold')

    # ROC
    fpr_n, tpr_n, _ = roc_curve(y_flat, p_novel)
    fpr_b, tpr_b, _ = roc_curve(y_flat, p_base)
    axes[0].plot(fpr_n, tpr_n, color=PALETTE["novel"],    label=f"ST-GNN   AUC={roc_auc_score(y_flat,p_novel):.3f}")
    axes[0].plot(fpr_b, tpr_b, color=PALETTE["baseline"], label=f"Baseline AUC={roc_auc_score(y_flat,p_base):.3f}", linestyle='--')
    axes[0].plot([0,1],[0,1],'k--', alpha=0.3)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC Curve")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # PR
    prec_n, rec_n, _ = precision_recall_curve(y_flat, p_novel)
    prec_b, rec_b, _ = precision_recall_curve(y_flat, p_base)
    axes[1].plot(rec_n, prec_n, color=PALETTE["novel"],    label=f"ST-GNN   AP={average_precision_score(y_flat,p_novel):.3f}")
    axes[1].plot(rec_b, prec_b, color=PALETTE["baseline"], label=f"Baseline AP={average_precision_score(y_flat,p_base):.3f}", linestyle='--')
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_confusion_matrix(cm, model_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Reds')
    plt.colorbar(im, ax=ax)
    classes = ['No Fire', 'Fire']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_comparison_bar(metrics_stgnn, metrics_baseline):
    """Horizontal bar chart comparing ST-GNN vs Baseline on 4 key metrics."""
    labels  = ['Recall', 'Precision', 'F1-Score', 'AUC-PR']
    novel_v = [metrics_stgnn[k] for k in ['recall', 'precision', 'f1', 'auc_pr']]
    base_v  = [metrics_baseline[k] for k in ['recall', 'precision', 'f1', 'auc_pr']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, novel_v, width, label='ST-GNN (Proposed)', color=PALETTE["novel"],    alpha=0.85)
    b2 = ax.bar(x + width/2, base_v,  width, label='Baseline LSTM',     color=PALETTE["baseline"], alpha=0.85)

    ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("ST-GNN vs Baseline LSTM — Recall-Centric Comparison", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "model_comparison_bar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_spatial_fire_map(y_true, y_prob, nodes_df, time_idx: int = 0):
    """
    Plots actual vs predicted fire map for a single time step.
    nodes_df has columns ['latitude', 'longitude'].
    """
    y_true_step = y_true[time_idx].flatten()
    y_prob_step = y_prob[time_idx].flatten()
    y_pred_step = (y_prob_step >= config.RECALL_THRESHOLD).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, title, cmap in zip(
        axes,
        [y_true_step, y_pred_step],
        ["Actual Fire Map", "Predicted Fire Map (ST-GNN)"],
        ["Reds", "OrRd"],
    ):
        sc = ax.scatter(
            nodes_df['longitude'], nodes_df['latitude'],
            c=vals, cmap=cmap, s=80, edgecolors='grey', linewidths=0.3,
            vmin=0, vmax=1,
        )
        plt.colorbar(sc, ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)

    plt.suptitle(f"Spatial Fire Prediction — Time Step {time_idx}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"spatial_fire_map_t{time_idx}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def save_metrics_csv(metrics_stgnn, metrics_baseline):
    import csv
    path = os.path.join(config.RESULTS_DIR, "stgnn_evaluation.csv")
    fieldnames = ['Model', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'AUC-PR']
    rows = []
    for m in [metrics_stgnn, metrics_baseline]:
        rows.append({
            'Model':     m['model_name'],
            'Threshold': f"{m['threshold']:.3f}",
            'Accuracy':  f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall':    f"{m['recall']:.4f}",
            'F1':        f"{m['f1']:.4f}",
            'AUC-ROC':   f"{m['auc_roc']:.4f}",
            'AUC-PR':    f"{m['auc_pr']:.4f}",
        })
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Evaluate] Results saved -> {path}")
