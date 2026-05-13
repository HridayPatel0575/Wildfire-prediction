import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import config

# ─────────────────────────────────────────────────────────────────────────────
# Custom Layer: Temporal LSTM Block (per-node shared LSTM)
# Wraps tf ops inside call() to be compatible with Keras 3 Functional API
# ─────────────────────────────────────────────────────────────────────────────

class TemporalLSTMBlock(layers.Layer):
    """
    Applies a shared LSTM independently to each node's time series.
    Input:  (Batch, Seq, Nodes, Features)
    Output: (Batch, Nodes, LSTM_Units)
    """
    def __init__(self, n_nodes, seq_len, n_features, lstm_units, **kwargs):
        super().__init__(**kwargs)
        self.n_nodes    = n_nodes
        self.seq_len    = seq_len
        self.n_features = n_features
        self.lstm       = layers.LSTM(lstm_units, return_sequences=False)

    def call(self, x):
        # x: (Batch, Seq, Nodes, Features)
        # Step 1: permute to (Batch, Nodes, Seq, Features)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # Step 2: merge Batch & Nodes -> (Batch * Nodes, Seq, Features)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size * self.n_nodes, self.seq_len, self.n_features))
        # Step 3: run shared LSTM -> (Batch * Nodes, LSTM_Units)
        x = self.lstm(x)
        # Step 4: restore -> (Batch, Nodes, LSTM_Units)
        x = tf.reshape(x, (batch_size, self.n_nodes, config.LSTM_UNITS))
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "n_nodes": self.n_nodes,
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "lstm_units": config.LSTM_UNITS,
        })
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Custom Layer: Graph Convolution (GCN)
# ─────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(layers.Layer):
    """
    Graph Convolution Layer: H_next = ReLU(A_hat * H * W + b)
    Input H shape:  (Batch, Nodes, Features)
    Output shape:   (Batch, Nodes, Units)
    The normalized adjacency matrix A_hat is baked in as a constant.
    """
    def __init__(self, units, adjacency_matrix, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # Store the pre-computed normalized adjacency matrix as a constant
        self.A_hat = tf.constant(adjacency_matrix, dtype=tf.float32)

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.W = self.add_weight(
            shape=(in_features, self.units),
            initializer='glorot_uniform',
            name='gcn_W',
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='gcn_b',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, H):
        # H:    (Batch, Nodes, Features)
        # HW:   (Batch, Nodes, Units)
        HW = tf.matmul(H, self.W)
        # A_hat @ HW per batch: (Batch, Nodes, Units)
        AHW = tf.einsum('vw,bwu->bvu', self.A_hat, HW)
        return tf.nn.relu(AHW + self.b)

    def get_config(self):
        base = super().get_config()
        base.update({"units": self.units})
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss (Handles 1.7% Class Imbalance at multi-node output level)
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA):
    """
    Focal loss compatible with output shape (Batch, Nodes, 1).
    """
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce      = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        loss    = alpha_t * tf.math.pow(1.0 - p_t, gamma) * ce
        return tf.reduce_mean(loss)
    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# ST-GNN: Full Spatio-Temporal Graph Neural Network
# ─────────────────────────────────────────────────────────────────────────────

def build_stgnn_model(seq_len: int, n_nodes: int, n_features: int,
                      adjacency_matrix: np.ndarray) -> Model:
    """
    ST-GNN Architecture:
      1. Temporal Branch  — Shared LSTM learns each node's 14-day fire dynamics.
      2. Spatial Branch   — GCN propagates those representations across
                            geographically close nodes via Haversine adjacency.
      3. Classification   — Dense layer predicts fire probability for all
                            180 grid cells simultaneously.

    Input:  (Batch, seq_len=14, n_nodes=180, n_features=35)
    Output: (Batch, n_nodes=180, 1)
    """
    inp = layers.Input(
        shape=(seq_len, n_nodes, n_features),
        name='spatio_temporal_input',
    )

    # 1. TEMPORAL — per-node shared LSTM
    temporal_repr = TemporalLSTMBlock(
        n_nodes=n_nodes, seq_len=seq_len, n_features=n_features,
        lstm_units=config.LSTM_UNITS, name='temporal_lstm_block',
    )(inp)                                       # -> (B, Nodes, LSTM_UNITS)
    temporal_repr = layers.LayerNormalization(name='temporal_norm')(temporal_repr)
    temporal_repr = layers.Dropout(config.DROPOUT_RATE)(temporal_repr)

    # 2. SPATIAL — two stacked GCN layers for multi-hop propagation
    gcn_1 = GraphConvLayer(config.GCN_UNITS, adjacency_matrix, name='gcn_1')(temporal_repr)
    gcn_1 = layers.LayerNormalization(name='gcn_1_norm')(gcn_1)
    gcn_1 = layers.Dropout(config.DROPOUT_RATE)(gcn_1)

    gcn_2 = GraphConvLayer(config.GCN_UNITS // 2, adjacency_matrix, name='gcn_2')(gcn_1)
    gcn_2 = layers.LayerNormalization(name='gcn_2_norm')(gcn_2)
    gcn_2 = layers.Dropout(config.DROPOUT_RATE)(gcn_2)

    # 3. CLASSIFICATION HEAD (Dense per node)
    h   = layers.Dense(config.DENSE_UNITS, activation='relu', name='dense_head')(gcn_2)
    out = layers.Dense(1, activation='sigmoid', name='fire_probability')(h)
    # out shape: (Batch, Nodes, 1)

    model = Model(inputs=inp, outputs=out, name='ST_GNN')

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            clipnorm=1.0,   # Prevent exploding gradients through GCN einsum
        ),
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
