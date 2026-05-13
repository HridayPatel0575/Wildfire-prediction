import os

# --- Paths --------------------------------------------------------------------
BASE_DIR       = r"c:\Users\Admin\Desktop\Projects\FWI_Index_India"
STGNN_DIR      = os.path.join(BASE_DIR, "Novel_STGNN")
RAW_DATA_FILE  = os.path.join(BASE_DIR, "Main_Data", "FLAGED_data.csv")
RESULTS_DIR    = os.path.join(STGNN_DIR, "results")
MODELS_DIR     = os.path.join(STGNN_DIR, "saved_models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# --- Spatio-Temporal Settings ------------------------------------------------
SEQ_LEN        = 14          # 14-day look-back window
# For the graph adjacency matrix (Haversine distance in km)
# Two nodes (lat/lon coordinates) are connected if they are within this distance
# The Indian subcontinent is large; 250km is a reasonable radius for weather systems.
ADJ_DISTANCE_THRESH_KM = 250.0 

# --- Model Hyperparameters ---------------------------------------------------
GCN_UNITS      = 64          # Graph Convolution hidden units
LSTM_UNITS     = 64          # Temporal sequence hidden units
DENSE_UNITS    = 32
DROPOUT_RATE   = 0.30
LEARNING_RATE  = 0.0001
BATCH_SIZE     = 32          # Note: Batch is now sequences of all 180 nodes at once
EPOCHS         = 60
PATIENCE       = 10

# Focal Loss handles the 1.7% class imbalance
FOCAL_ALPHA    = 0.85
FOCAL_GAMMA    = 2.0

# Recall optimization threshold
MIN_PRECISION  = 0.10
RECALL_THRESHOLD = 0.25

# --- Train/Test Split --------------------------------------------------------
RANDOM_STATE   = 42
TEST_YEARS     = [2019, 2020]

# --- Feature Columns ---------------------------------------------------------
# Exact parity with all previous experiments (35 features)
BASE_COLS = [
    'temp', 'humidity', 'wind_speed', 'rainfall',
    'cvh', 'lai_hv', 'lai_lv', 'cvl', 'tvh', 'tvl',
]

FWI_COMPONENT_COLS = [
    'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI',
]

ENGINEERED_COLS = [
    'temp_humidity_interaction', 'wind_temp_interaction', 'fwi_temp_ratio', 
    'humidity_rainfall_interaction', 'temp_squared', 'fwi_squared', 
    'wind_speed_squared', 'humidity_squared', 'dryness_index', 
    'fire_danger_index', 'wind_dryness', 'temp_deviation', 
    'fwi_humidity_ratio', 'temp_rainfall_ratio', 'wind_humidity_ratio', 
    'temp_category', 'humidity_category', 'fwi_category', 'log_fwi',
]

FEATURE_COLS = BASE_COLS + FWI_COMPONENT_COLS + ENGINEERED_COLS

TARGET_COL     = 'fire_flag'
LOCATION_COLS  = ['latitude', 'longitude']
DATE_COL       = 'date'
