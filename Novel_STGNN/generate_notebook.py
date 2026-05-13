import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# ------------------------------------------------------------------
# 1. Title & Abstract
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""\
# Spatio-Temporal Graph Neural Network (ST-GNN)
### A Geography-Aware Deep Learning Architecture for Wildfire Prediction in the Indian Subcontinent

**Abstract:**  
Wildfire propagation is inherently a spatial phenomenon — drought conditions
and fire spread are correlated across geographic neighbours.  
Classical time-series models (LSTMs, Transformers) ignore this spatial structure,
processing every grid cell in isolation.

This notebook introduces the **Spatio-Temporal Graph Neural Network (ST-GNN)**,
a fundamentally novel architecture that:

1. **Constructs a Geographic Graph** over all 180 lat/lon grid cells of the Indian
   subcontinent using pairwise Haversine distances (threshold = 250 km).
2. **Temporal Branch (Shared LSTM)** learns each node's 14-day fire dynamics.
3. **Spatial Branch (Stacked GCN)** propagates those representations across
   geographic neighbours — if a neighbouring region is entering severe drought,
   the model *sees it and uses it*.
4. **Predicts simultaneously** for all 180 nodes per forward pass, making inference
   across the entire subcontinent instantaneous.

> **Why is this fundamentally different from all previous attempts?**  
> Random Forest, XGBoost, ANN, LSTM, DSCAN — none of them can use information  
> from surrounding locations. ST-GNN is the only model in this paper that can.
"""))

# ------------------------------------------------------------------
# 2. Setup
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup"))
nb.cells.append(nbf.v4.new_code_cell("""\
import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

tf.random.set_seed(42)
np.random.seed(42)

# Add ST-GNN folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

import config
from data_loader import get_graph_data
from stgnn_model import build_stgnn_model
from train import get_callbacks, build_baseline_lstm
from evaluate import (
    compute_metrics, plot_training_history, plot_roc_pr_curves,
    plot_comparison_bar, plot_confusion_matrix, plot_spatial_fire_map,
)

print(f"TensorFlow: {tf.__version__}")
print(f"Nodes in graph: 180 | Features: 35 | Sequence length: {config.SEQ_LEN} days")
"""))

# ------------------------------------------------------------------
# 3. Graph Construction
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 2. Geographic Graph Construction

We model the Indian subcontinent as a graph where:
- Each **node** is a lat/lon grid cell (180 total).
- Two nodes are **connected** if their Haversine distance < 250 km.
- The adjacency matrix is **symmetric-normalised** (D⁻⁰·⁵ A D⁻⁰·⁵) for stable GCN gradients.
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
data = get_graph_data()

A = data['A_norm']
print(f"Adjacency Matrix: {A.shape}")
print(f"Non-zero entries (edges + self-loops): {int((A > 0).sum())}")
print(f"Average neighbors per node: {(A > 0).sum(axis=1).mean():.1f}")

# Visualise the sparsity pattern
fig, ax = plt.subplots(figsize=(6, 6))
ax.spy(A > 0, markersize=1.5, color='steelblue')
ax.set_title("Adjacency Matrix Sparsity\\n(250 km distance threshold)", fontsize=12)
plt.tight_layout()
plt.show()

print(f"\\nData shapes:")
print(f"  X_train : {data['X_train'].shape}  (Batch, Seq, Nodes, Features)")
print(f"  y_train : {data['y_train'].shape}  (Batch, Nodes, 1)")
print(f"  X_test  : {data['X_test'].shape}")
"""))

# ------------------------------------------------------------------
# 4. Model Architecture
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 3. ST-GNN vs Baseline LSTM

| Component       | **ST-GNN (Novel)**                      | **Baseline LSTM**     |
|-----------------|------------------------------------------|-----------------------|
| Temporal        | Shared LSTM per node (14-day sequences)  | Same                  |
| **Spatial**     | **Two stacked GCN layers (A @ H @ W)**  | **None — isolated**   |
| Sees Neighbours | **Yes — 250 km radius**                  | No                    |
| Output          | All 180 nodes simultaneously             | Same                  |

The **only** architectural difference is the Graph Convolution block.  
This makes for a clean ablation study proving the spatial component's value.
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
# 1. ST-GNN (Novel)
model_stgnn = build_stgnn_model(
    data['seq_len'], data['n_nodes'], data['n_features'], data['A_norm']
)
print("==== ST-GNN (Novel) ====")
model_stgnn.summary()

print("\\n")

# 2. Baseline LSTM (Ablation — no GCN)
model_baseline = build_baseline_lstm(data['seq_len'], data['n_nodes'], data['n_features'])
print("==== Baseline LSTM (No GCN) ====")
model_baseline.summary()
"""))

# ------------------------------------------------------------------
# 5. Training
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 4. Training with Focal Loss

Both models are trained with **Focal Loss** (α=0.85, γ=2.0) to heavily penalise
missed fire events in the severely imbalanced dataset (1.89% fire rate).

Test set = years 2019-2020 (strict temporal hold-out, no data leakage).
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
print("Training ST-GNN...")
hist_stgnn = model_stgnn.fit(
    data['X_train'], data['y_train'],
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_split=0.15,
    callbacks=get_callbacks("ST_GNN"),
    verbose=1,
)

print("\\nTraining Baseline LSTM...")
hist_baseline = model_baseline.fit(
    data['X_train'], data['y_train'],
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_split=0.15,
    callbacks=get_callbacks("Baseline_LSTM"),
    verbose=1,
)
"""))

# ------------------------------------------------------------------
# 6. Evaluation
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 5. Recall-Optimised Evaluation

In wildfire prediction, **False Negatives are catastrophic** (missed fires).  
We sweep all decision thresholds and select the one maximising Recall while
maintaining Precision ≥ 10% (i.e. accepting up to a 10:1 false-alarm ratio).

The primary reported metrics are **Recall** and **AUC-PR**.
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
y_prob_stgnn    = model_stgnn.predict(data['X_test'],    batch_size=config.BATCH_SIZE, verbose=0)
y_prob_baseline = model_baseline.predict(data['X_test'], batch_size=config.BATCH_SIZE, verbose=0)

metrics_stgnn    = compute_metrics(data['y_test'], y_prob_stgnn,    model_name="ST-GNN (Spatial)")
metrics_baseline = compute_metrics(data['y_test'], y_prob_baseline, model_name="Baseline LSTM (No GCN)")
"""))

# ------------------------------------------------------------------
# 7. Visualisations
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 6. Visualisations
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
# Training history
plot_training_history(hist_stgnn, "ST_GNN");        plt.show()
plot_training_history(hist_baseline, "Baseline_LSTM"); plt.show()

# ROC & PR curves
plot_roc_pr_curves(data['y_test'], y_prob_stgnn, y_prob_baseline); plt.show()

# Grouped comparison bar
plot_comparison_bar(metrics_stgnn, metrics_baseline); plt.show()

# Confusion matrices
plot_confusion_matrix(metrics_stgnn['confusion_matrix'],    "ST_GNN");       plt.show()
plot_confusion_matrix(metrics_baseline['confusion_matrix'], "Baseline_LSTM"); plt.show()
"""))

# ------------------------------------------------------------------
# 8. Spatial Fire Map (unique to ST-GNN)
# ------------------------------------------------------------------
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 7. Spatial Fire Prediction Map

The ST-GNN outputs predictions for **all 180 geographic nodes simultaneously**.
This allows us to visualise the complete predicted vs. actual fire map across the
Indian subcontinent for any given time step — a capability no previous model in
this paper possesses.
"""))
nb.cells.append(nbf.v4.new_code_cell("""\
from graph_builder import build_adjacency_matrix
import pandas as pd

# Reload node coordinates for plotting
df_raw = pd.read_csv(config.RAW_DATA_FILE, usecols=['latitude', 'longitude'])
_, nodes_df = build_adjacency_matrix(df_raw, config.ADJ_DISTANCE_THRESH_KM)

# Plot the spatial fire map for the first test time step
plot_spatial_fire_map(data['y_test'], y_prob_stgnn, nodes_df, time_idx=0)
plt.show()
"""))

# ------------------------------------------------------------------
# Write file
# ------------------------------------------------------------------
nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.', "STGNN_Journal_Notebook.ipynb")
nb_path = "STGNN_Journal_Notebook.ipynb"
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written -> {nb_path}")
