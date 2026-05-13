"""
Configuration — FWI-Gated LSTM vs Vanilla LSTM
Wildfire Prediction for the Indian Subcontinent
Journal Publication.
"""

import os

# --- Paths --------------------------------------------------------------------
BASE_DIR       = r"c:\Users\Admin\Desktop\Projects\FWI_Index_India"
NOVEL_DIR      = os.path.join(BASE_DIR, "Novel_Model")
RAW_DATA_FILE  = os.path.join(BASE_DIR, "Main_Data", "FLAGED_data.csv")
RESULTS_DIR    = os.path.join(NOVEL_DIR, "results")
MODELS_DIR     = os.path.join(NOVEL_DIR, "saved_models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# --- Sequence Settings --------------------------------------------------------
SEQ_LEN        = 14          # 14-day look-back window
STRIDE         = 1           # Step between consecutive sequences

# --- Model Hyperparameters ---------------------------------------------------
LSTM_UNITS_1   = 128
LSTM_UNITS_2   = 64
DENSE_UNITS    = 32
DROPOUT_RATE   = 0.30
LEARNING_RATE  = 0.001
BATCH_SIZE     = 512
EPOCHS         = 60
PATIENCE       = 10

# --- Data Balancing ----------------------------------------------------------
# SMOTE is applied ONLY to the training set (after carving out a real val set).
# The validation set is kept at natural distribution so training-time recall
# honestly reflects what will happen at test time.
USE_SMOTE      = True
VAL_SPLIT      = 0.15       # Fraction of (pre-SMOTE) training data for validation

# --- Focal Loss (for extreme class imbalance — fire events ≈ 1.72%) ----------
# alpha: weight for the FIRE class (1). Higher → stronger recall focus.
FOCAL_ALPHA    = 0.85
# gamma: focusing exponent. gamma=2 reduces easy-example loss by (1-p_t)^2.
FOCAL_GAMMA    = 2.0
# Physics-recall penalty weight (novel model only)
PHYSICS_RECALL_LAMBDA = 0.30

# --- Recall-Optimised Threshold ----------------------------------------------
# In fire prediction, false negatives (missed fires) are catastrophic.
# We use a LOWER threshold than 0.5 to push more predictions towards Fire.
# MIN_PRECISION prevents precision from collapsing entirely (false alarm control).
MIN_PRECISION    = 0.10    # Accept up to 10:1 false alarm ratio to maximise recall
RECALL_THRESHOLD = 0.25    # Default fixed threshold (for reference only)

# --- Train/Test Split --------------------------------------------------------
RANDOM_STATE   = 42
TEST_YEARS     = [2019, 2020]   # Temporal hold-out — no data leakage

# --- Feature Columns ---------------------------------------------------------
# Matches EXACTLY the features used by RF/XGBoost/ANN in the paper
# PLUS FWI sub-components (FFMC, DMC, DC, ISI, BUI) for richer physics signal.

# Base meteorological + vegetation (10 features)
BASE_COLS = [
    'temp', 'humidity', 'wind_speed', 'rainfall',
    'cvh', 'lai_hv', 'lai_lv', 'cvl', 'tvh', 'tvl',
]

# FWI system components — full physics chain (6 features)
FWI_COMPONENT_COLS = [
    'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI',
]

# Engineered interaction features (19 features)
ENGINEERED_COLS = [
    'temp_humidity_interaction',
    'wind_temp_interaction',
    'fwi_temp_ratio',
    'humidity_rainfall_interaction',
    'temp_squared',
    'fwi_squared',
    'wind_speed_squared',
    'humidity_squared',
    'dryness_index',
    'fire_danger_index',
    'wind_dryness',
    'temp_deviation',
    'fwi_humidity_ratio',
    'temp_rainfall_ratio',
    'wind_humidity_ratio',
    'temp_category',
    'humidity_category',
    'fwi_category',
    'log_fwi',
]

# Combined: all 35 features
FEATURE_COLS = BASE_COLS + FWI_COMPONENT_COLS + ENGINEERED_COLS

# Index of the FWI column (used by the physics branch)
FWI_FEATURE_IDX = FEATURE_COLS.index('FWI')

TARGET_COL     = 'fire_flag'
LOCATION_COLS  = ['latitude', 'longitude']
DATE_COL       = 'date'
