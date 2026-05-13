import os

# Base paths
BASE_DIR = r"c:\Users\Admin\Desktop\Projects\FWI_Index_India"
RAW_DATA_DIR = os.path.join(BASE_DIR, "Main_Data")

# File paths
ENHANCED_DATA_FILE = os.path.join(BASE_DIR, "FLAGED_data_enhanced.csv")
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "FLAGED_data.csv")
FIRE_DATA_FILE = os.path.join(RAW_DATA_DIR, "uttarakhand_fires.csv")

# FWI calculation defaults
FFMC0_DEFAULT = 85.0
DMC0_DEFAULT = 6.0
DC0_DEFAULT = 15.0

# Preprocessing configs
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 32
DROPOUT_RATE = 0.15

# Columns to use for FWI calculation interpolation map
TARGET_DATE = "2016-05-28"
