import sys; sys.path.insert(0,'.')
import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

from data_loader import get_graph_data
from stgnn_model import build_stgnn_model

data = get_graph_data()

nan_count = int(np.isnan(data['X_train']).sum())
print("NaN in X_train:", nan_count)
assert nan_count == 0, "Still has NaN!"
print("X_train range:", data['X_train'].min(), "to", data['X_train'].max())

model = build_stgnn_model(data['seq_len'], data['n_nodes'], data['n_features'], data['A_norm'])
h = model.fit(data['X_train'][:200], data['y_train'][:200], epochs=5, batch_size=32, verbose=1)

losses  = h.history['loss']
recalls = h.history.get('recall', [])
print("Losses :", [round(l, 5) for l in losses])
print("Recalls:", [round(r, 4) for r in recalls])

has_nan = any(v != v for v in losses)
print("PASS - Training is stable!" if not has_nan else "FAIL - NaN loss persists")
