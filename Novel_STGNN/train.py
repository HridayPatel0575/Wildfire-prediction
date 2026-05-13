"""
train.py — Full ST-GNN Training Pipeline
=========================================
Trains both:
  1. ST-GNN (Novel Proposed Model)
  2. Baseline LSTM (No spatial graph component — ablation)
Then evaluates both on the strict 2019-2020 temporal holdout.

Usage:
    cd Novel_STGNN
    python train.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger,
)

sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader import get_graph_data
from stgnn_model import build_stgnn_model, focal_loss
from evaluate import (
    compute_metrics, plot_training_history, plot_roc_pr_curves,
    plot_confusion_matrix, plot_comparison_bar, save_metrics_csv,
)


def set_seeds(seed=config.RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_callbacks(model_name: str) -> list:
    return [
        EarlyStopping(monitor='val_auc_pr', mode='max', patience=config.PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc_pr', mode='max', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            os.path.join(config.MODELS_DIR, f"{model_name}_best.keras"),
            monitor='val_auc_pr', mode='max', save_best_only=True,
        ),
        CSVLogger(os.path.join(config.RESULTS_DIR, f"{model_name}_log.csv")),
    ]


def build_baseline_lstm(seq_len, n_nodes, n_features):
    """
    Ablation baseline: single shared LSTM with no GCN spatial propagation.
    All nodes are still processed independently — it cannot see neighbors.
    """
    inp = layers.Input(shape=(seq_len, n_nodes, n_features), name='input')

    # Collapse nodes into batch for shared LSTM (same as ST-GNN temporal block)
    from stgnn_model import TemporalLSTMBlock
    temporal = TemporalLSTMBlock(
        n_nodes=n_nodes, seq_len=seq_len, n_features=n_features,
        lstm_units=config.LSTM_UNITS, name='baseline_lstm',
    )(inp)
    temporal = layers.Dropout(config.DROPOUT_RATE)(temporal)

    # NO graph convolution — predict directly from temporal repr only
    h   = layers.Dense(config.DENSE_UNITS, activation='relu')(temporal)
    out = layers.Dense(1, activation='sigmoid', name='fire_probability')(h)

    model = Model(inputs=inp, outputs=out, name='Baseline_LSTM')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=focal_loss(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc_roc'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
        ],
    )
    return model


def train_model(model, name, data):
    print(f"\n{'-'*60}\n  Training: {name}\n{'-'*60}")
    model.summary(line_length=75)

    history = model.fit(
        data['X_train'], data['y_train'],
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.15,
        callbacks=get_callbacks(name),
        verbose=1,
    )
    y_prob = model.predict(data['X_test'], batch_size=config.BATCH_SIZE, verbose=0)
    return history, y_prob


def main():
    print("\n" + "="*60)
    print("  Spatio-Temporal Graph Neural Network (ST-GNN)")
    print("  Wildfire Prediction | Indian Subcontinent")
    print("="*60)

    set_seeds()

    # GPU config
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"\n[GPU] {len(gpus)} GPU(s) detected.")
    else:
        print("\n[GPU] No GPU found — training on CPU.")

    # 1. Load Data
    data = get_graph_data()

    # 2. Build models
    print("\n[Architecture] Building ST-GNN (Novel)...")
    model_stgnn = build_stgnn_model(
        data['seq_len'], data['n_nodes'], data['n_features'], data['A_norm'],
    )

    print("[Architecture] Building Baseline LSTM (No GCN — Ablation)...")
    model_baseline = build_baseline_lstm(
        data['seq_len'], data['n_nodes'], data['n_features'],
    )

    # 3. Train
    hist_stgnn,    y_prob_stgnn    = train_model(model_stgnn,    "ST_GNN",       data)
    hist_baseline, y_prob_baseline = train_model(model_baseline, "Baseline_LSTM", data)

    # 4. Evaluate
    metrics_stgnn    = compute_metrics(data['y_test'], y_prob_stgnn,    model_name="ST-GNN (Spatial)")
    metrics_baseline = compute_metrics(data['y_test'], y_prob_baseline, model_name="Baseline LSTM (No GCN)")

    # 5. Plots
    plot_training_history(hist_stgnn,    "ST_GNN")
    plot_training_history(hist_baseline, "Baseline_LSTM")
    plot_roc_pr_curves(data['y_test'], y_prob_stgnn, y_prob_baseline)
    plot_comparison_bar(metrics_stgnn, metrics_baseline)
    plot_confusion_matrix(metrics_stgnn['confusion_matrix'],    "ST_GNN")
    plot_confusion_matrix(metrics_baseline['confusion_matrix'], "Baseline_LSTM")

    # 6. Save
    save_metrics_csv(metrics_stgnn, metrics_baseline)
    model_stgnn.save(os.path.join(config.MODELS_DIR,    "ST_GNN_final.keras"))
    model_baseline.save(os.path.join(config.MODELS_DIR, "Baseline_LSTM_final.keras"))

    print("\n" + "="*60)
    print(f"  Training Complete! Results -> {config.RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
