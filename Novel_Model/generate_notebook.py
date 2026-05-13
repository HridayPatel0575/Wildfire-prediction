"""
Generate FWI_Gated_LSTM_Journal.ipynb
Focused comparison: FWI-Gated LSTM vs Vanilla LSTM
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# ── Title ──────────────────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("""\
# FWI-Gated LSTM: Physics-Residual Network for Wildfire Prediction
### Indian Subcontinent — Journal Publication

**Model comparison:** FWI-Gated LSTM (proposed) vs Vanilla LSTM (ablation)

#### Novel Architecture Summary
The **FWI-Gated LSTM** (FWI-Physics Residual Network) augments a standard stacked LSTM
with a non-parametric *Physics Statistics Branch* that computes five sequence-level
summaries of the 14-day FWI window:

| Statistic | Captures |
|-----------|----------|
| Mean FWI | Average drought level over the window |
| Max FWI | Worst single-day fire danger |
| FWI Trend | Is danger building (+) or easing (−)? |
| Fraction High-FWI | Days above mean danger threshold |
| FWI Volatility | Day-to-day instability |

These are **genuinely new information** the LSTM cannot access as point features.
They feed a 5-weight *physics logit* that adds directly to the LSTM logit as a
residual bypass — a structural difference a vanilla LSTM cannot replicate.

#### Why Training Recall ≈ Test Recall (Fix Applied)
Previous code carved the validation set *after* SMOTE, producing a 50/50 balanced
val set. This inflated training-time recall because the model never saw the real
1.7% fire distribution during training. **Fix:** the validation fold is now carved
from raw data *before* SMOTE, so `val_recall` during training honestly predicts
the final test recall.
"""))

# ── 1. Setup ───────────────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("## 1. Setup & Imports"))
nb.cells.append(nbf.v4.new_code_cell("""\
import os, sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams.update({'figure.dpi': 110, 'font.size': 11})

sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

import config
from sequence_builder import get_prepared_data
from fwi_gated_lstm import (
    build_fwi_gated_lstm, build_baseline_lstm,
    print_physics_report, get_callbacks
)
from losses import focal_loss
from evaluate import (
    compute_metrics, build_comparison_table, print_detailed_comparison,
    plot_training_history, plot_roc_pr_curves, plot_comparison_bar,
    plot_confusion_matrices_side_by_side, plot_threshold_tradeoff,
    plot_physics_weights,
)

print(f"TensorFlow: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
"""))

# ── 2. Data ────────────────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 2. Data Preparation

**Pipeline:**
1. Load `FLAGED_data.csv` and engineer all 35 features
2. Build 14-day rolling sequences per (lat, lon) location
3. Temporal hold-out split: **2019–2020 as test** (no data leakage)
4. Carve validation set from training **before SMOTE** (real 1.7% fire distribution)
5. StandardScaler fitted on training fold only
6. SMOTE applied to training fold only → 50/50 balance for training
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
data = get_prepared_data()

print(f"\\nX_train: {data['X_train'].shape}  (SMOTE-balanced, {data['y_train'].mean()*100:.1f}% fire)")
print(f"X_val  : {data['X_val'].shape}  (real distribution, {data['y_val'].mean()*100:.2f}% fire)")
print(f"X_test : {data['X_test'].shape}  (real distribution, {data['y_test'].mean()*100:.2f}% fire)")
print(f"\\nFeatures : {data['n_features']}")
print(f"Seq length: {data['seq_len']} days")
print(f"FWI index : {data['fwi_idx']}  ({config.FEATURE_COLS[data['fwi_idx']]})")
"""))

# ── 3. Architectures ───────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 3. Model Architectures

Both models use the **same stacked LSTM trunk** (128 → 64 → Dense 32).
The FWI-Gated LSTM adds:
- A zero-parameter Physics Statistics Branch (5 FWI sequence summaries)
- A 5-weight physics logit that adds directly to the LSTM logit (residual bypass)

This is the *only* structural difference — making it a clean ablation.
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
# Same loss on both so differences reflect architecture, not loss function
loss_fn = focal_loss()

print("=" * 60)
print("  FWI-Gated LSTM (Proposed)")
print("=" * 60)
model_novel = build_fwi_gated_lstm(
    data['seq_len'], data['n_features'], data['fwi_idx'], loss=loss_fn
)
model_novel.summary(line_length=65)

print("\\n" + "=" * 60)
print("  Vanilla LSTM (Baseline Ablation)")
print("=" * 60)
model_baseline = build_baseline_lstm(
    data['seq_len'], data['n_features'], loss=loss_fn
)
model_baseline.summary(line_length=65)

print(f"\\nParameter difference: {model_novel.count_params() - model_baseline.count_params():+,}")
print("(physics branch: 5 weights, 0 bias)")
"""))

# ── 4. Training ────────────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 4. Training

Both models use:
- **EarlyStopping** (monitor=`val_auc_pr`, patience=10)
- **ReduceLROnPlateau** (factor=0.5, patience=5)
- **ModelCheckpoint** (saves best val_auc_pr)
- Validation data = real-distribution fold (**not** from SMOTE'd pool)
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
print("Training FWI-Gated LSTM (Proposed)...")
hist_novel = model_novel.fit(
    data['X_train'], data['y_train'],
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_data=(data['X_val'], data['y_val']),
    callbacks=get_callbacks("FWI_Gated_LSTM"),
    verbose=1,
)
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
print("Training Vanilla LSTM (Baseline)...")
hist_baseline = model_baseline.fit(
    data['X_train'], data['y_train'],
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_data=(data['X_val'], data['y_val']),
    callbacks=get_callbacks("Vanilla_LSTM"),
    verbose=1,
)
"""))

# ── 5. Training Curves ─────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 5. Training Curves

Because validation uses the **real-distribution** fold, `val_recall` during training
should now be close to what we see in the final test evaluation (no more gap).
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
plot_training_history(hist_novel, "FWI_Gated_LSTM")
img = plt.imread(f"{config.RESULTS_DIR}/FWI_Gated_LSTM_training_history.png")
plt.figure(figsize=(16, 4)); plt.imshow(img); plt.axis('off'); plt.show()

plot_training_history(hist_baseline, "Vanilla_LSTM")
img = plt.imread(f"{config.RESULTS_DIR}/Vanilla_LSTM_training_history.png")
plt.figure(figsize=(16, 4)); plt.imshow(img); plt.axis('off'); plt.show()
"""))

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 6. Evaluation on Held-Out Test Set (2019–2020)

Metrics are computed with a **recall-optimised threshold** (maximises recall
subject to `min_precision = 0.10`). This is the right strategy for fire
prediction where missed fires are far costlier than false alarms.
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
y_prob_novel    = model_novel.predict(data['X_test'], verbose=0).flatten()
y_prob_baseline = model_baseline.predict(data['X_test'], verbose=0).flatten()

metrics_novel    = compute_metrics(data['y_test'], y_prob_novel,
                                   model_name="FWI-Gated LSTM")
metrics_baseline = compute_metrics(data['y_test'], y_prob_baseline,
                                   model_name="Vanilla LSTM")
"""))

# ── 7. Comparison Table ────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("## 7. Full Comparison Table"))
nb.cells.append(nbf.v4.new_code_cell("""\
comparison_df = build_comparison_table(metrics_novel, metrics_baseline)
print_detailed_comparison(metrics_novel, metrics_baseline)
"""))

# ── 8. Plots ──────────────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("## 8. Comparison Plots"))
nb.cells.append(nbf.v4.new_code_cell("""\
# ROC & Precision-Recall curves
plot_roc_pr_curves(data['y_test'], y_prob_novel, y_prob_baseline)
img = plt.imread(f"{config.RESULTS_DIR}/roc_pr_curves.png")
plt.figure(figsize=(13, 5)); plt.imshow(img); plt.axis('off'); plt.show()
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
# Threshold tradeoff
plot_threshold_tradeoff(data['y_test'], y_prob_novel, y_prob_baseline,
                        optimal_thresh=metrics_novel['threshold'])
img = plt.imread(f"{config.RESULTS_DIR}/threshold_tradeoff.png")
plt.figure(figsize=(14, 5)); plt.imshow(img); plt.axis('off'); plt.show()
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
# Grouped bar comparison (all models)
plot_comparison_bar(comparison_df)
img = plt.imread(f"{config.RESULTS_DIR}/model_comparison_bar.png")
plt.figure(figsize=(16, 6)); plt.imshow(img); plt.axis('off'); plt.show()
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
# Side-by-side confusion matrices
plot_confusion_matrices_side_by_side(
    metrics_novel['confusion_matrix'],
    metrics_baseline['confusion_matrix']
)
img = plt.imread(f"{config.RESULTS_DIR}/confusion_matrices.png")
plt.figure(figsize=(11, 4.5)); plt.imshow(img); plt.axis('off'); plt.show()
"""))

# ── 9. Physics Interpretability ───────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 9. Physics Branch Interpretability

The 5 weights learned by the physics logit reveal **which drought statistics**
are most important for Indian wildfire prediction — a novel, interpretable finding.
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
print_physics_report(model_novel)

plot_physics_weights(model_novel)
img = plt.imread(f"{config.RESULTS_DIR}/physics_weights.png")
plt.figure(figsize=(9, 4)); plt.imshow(img); plt.axis('off'); plt.show()
"""))

# ── 10. Save ──────────────────────────────────────────────────────────────────
nb.cells.append(nbf.v4.new_markdown_cell("## 10. Save Models & Results"))
nb.cells.append(nbf.v4.new_code_cell("""\
from evaluate import save_results_csv
save_results_csv(metrics_novel, metrics_baseline, comparison_df)

model_novel.save(   f"{config.MODELS_DIR}/FWI_Gated_LSTM_final.keras")
model_baseline.save(f"{config.MODELS_DIR}/Vanilla_LSTM_final.keras")

print(f"\\nModels saved to: {config.MODELS_DIR}")
print(f"Results saved to: {config.RESULTS_DIR}")
"""))

# Write notebook
out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "FWI_Gated_LSTM_Journal.ipynb"
)
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook generated: {out_path}")
