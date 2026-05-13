import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict

import config
from graph_builder import build_adjacency_matrix

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the feature engineering from the paper baseline exactly."""
    print("[DataLoader] Engineering features...")
    df['temp_humidity_interaction'] = df['temp'] * df['humidity'] / 100.0
    df['wind_temp_interaction'] = df['wind_speed'] * (df['temp'] + 273.15)
    df['fwi_temp_ratio'] = df['FWI'] / (df['temp'] + 50.0)
    df['humidity_rainfall_interaction'] = df['humidity'] * (df['rainfall'] + 0.001)
    
    df['temp_squared'] = df['temp']**2
    df['fwi_squared'] = df['FWI']**2
    df['wind_speed_squared'] = df['wind_speed']**2
    df['humidity_squared'] = df['humidity']**2
    
    df['dryness_index'] = (100 - df['humidity']) * (df['temp'] + 10) / 100.0
    df['fire_danger_index'] = df['FWI'] * (100 - df['humidity']) * (df['temp'] + 10) / (df['rainfall'] + 1)
    df['wind_dryness'] = df['wind_speed'] * (100 - df['humidity'])
    
    global_mean_temp = df['temp'].mean()
    df['temp_deviation'] = np.abs(df['temp'] - global_mean_temp)
    
    df['fwi_humidity_ratio'] = df['FWI'] / (df['humidity'] + 1)
    df['temp_rainfall_ratio'] = df['temp'] / (df['rainfall'] + 0.1)
    df['wind_humidity_ratio'] = df['wind_speed'] / (df['humidity'] + 1)
    
    df['temp_category'] = pd.cut(
        df['temp'], bins=[-np.inf, 0, 10, 20, 30, np.inf], labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(0)
    df['humidity_category'] = pd.cut(
        df['humidity'], bins=[-np.inf, 30, 50, 70, np.inf], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0)
    df['fwi_category'] = pd.cut(
        df['FWI'], bins=[-np.inf, 5, 10, 20, np.inf], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0)
    
    df['log_fwi'] = np.log1p(df['FWI'])
    return df

def get_graph_data() -> Dict:
    """
    Loads data, engineers features, builds Adjacency Matrix, 
    and returns perfectly formatted 4D tensors for ST-GNN:
    X shape: (num_batches, seq_len, num_nodes, n_features)
    y shape: (num_batches, num_nodes, 1)
    """
    print(f"[DataLoader] Loading: {config.RAW_DATA_FILE}")
    df = pd.read_csv(config.RAW_DATA_FILE, parse_dates=['date'])
    df = feature_engineering(df)
    
    # 1. Build Adjacency Matrix
    A_norm, nodes_df = build_adjacency_matrix(df, config.ADJ_DISTANCE_THRESH_KM)
    num_nodes = len(nodes_df)
    
    # 2. Sort strictly to guarantee grid structure
    # Sort by date, then latitude, then longitude
    df = df.sort_values(by=['date', 'latitude', 'longitude']).reset_index(drop=True)
    
    unique_dates = df['date'].unique()
    num_days = len(unique_dates)
    
    print(f"[DataLoader] Reshaping {num_days} days x {num_nodes} nodes...")
    
    # Ensure no missing rows
    assert len(df) == num_days * num_nodes, "Missing dates for some nodes!"
    
    # Extract features array and reshape
    features_arr = df[config.FEATURE_COLS].values
    target_arr = df[config.TARGET_COL].values
    
    # Shape: (Days, Nodes, Features)
    X_grid = features_arr.reshape((num_days, num_nodes, len(config.FEATURE_COLS)))
    y_grid = target_arr.reshape((num_days, num_nodes, 1))
    
    # We must scale features BEFORE sequence creation to save memory,
    # but we MUST NOT leak test data into the scaler.
    # Find cutoff index for test years (2019, 2020)
    test_dates_mask = pd.DatetimeIndex(unique_dates).year.isin(config.TEST_YEARS)
    train_dates_mask = ~test_dates_mask
    
    train_days_idx = np.where(train_dates_mask)[0]
    test_days_idx = np.where(test_dates_mask)[0]
    
    print("[DataLoader] Fitting RobustScaler strictly on Train data (2015-2018)...")
    print("             (RobustScaler uses median+IQR, resists extreme outliers like fire_danger_index)")
    X_train_raw = X_grid[train_days_idx].reshape(-1, len(config.FEATURE_COLS))
    scaler = RobustScaler()
    scaler.fit(X_train_raw)
    
    # Scale entire grid, then clip to [-10, 10] to prevent LSTM gate saturation
    X_grid_scaled = scaler.transform(
        X_grid.reshape(-1, len(config.FEATURE_COLS))
    ).reshape(num_days, num_nodes, len(config.FEATURE_COLS))
    # Clip to [-10, 10] and replace any residual NaN/Inf with 0
    X_grid_scaled = np.clip(X_grid_scaled, -10.0, 10.0)
    X_grid_scaled = np.nan_to_num(X_grid_scaled, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    
    # Save scaler
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 3. Build sliding windows
    print(f"[DataLoader] Building Spatio-Temporal sequences (Window={config.SEQ_LEN})...")
    X_seqs = []
    y_seqs = []
    years_seqs = [] # Track the year of the TARGET day for splitting
    
    for i in range(num_days - config.SEQ_LEN):
        # Window of 14 days for all nodes
        window = X_grid_scaled[i : i + config.SEQ_LEN] 
        # Target is the 15th day for all nodes
        target = y_grid[i + config.SEQ_LEN]
        # Target year
        target_year = pd.to_datetime(unique_dates[i + config.SEQ_LEN]).year
        
        X_seqs.append(window)
        y_seqs.append(target)
        years_seqs.append(target_year)
        
    X_seqs = np.array(X_seqs, dtype=np.float32) # (Batch, 14, 180, 35)
    y_seqs = np.array(y_seqs, dtype=np.float32) # (Batch, 180, 1)
    years_seqs = np.array(years_seqs)
    
    print(f"[DataLoader] Total sequences: {X_seqs.shape[0]} | Shape X: {X_seqs.shape} | Shape y: {y_seqs.shape}")
    
    # 4. Temporal Train/Test Split
    test_mask = np.isin(years_seqs, config.TEST_YEARS)
    train_mask = ~test_mask
    
    X_train, y_train = X_seqs[train_mask], y_seqs[train_mask]
    X_test,  y_test  = X_seqs[test_mask],  y_seqs[test_mask]
    
    # Because we are predicting 180 nodes at once, the total number of labels is Batch * 180
    train_fire_rate = y_train.sum() / y_train.size * 100
    test_fire_rate = y_test.sum() / y_test.size * 100
    
    print(f"[Split] Train sequences: {len(X_train)} (Fire rate: {train_fire_rate:.2f}%)")
    print(f"[Split] Test sequences:  {len(X_test)} (Fire rate: {test_fire_rate:.2f}%)")
    
    # NOTE: We do NOT use SMOTE here. SMOTE on 4D graph structures destroys the spatial relationship.
    # Instead, we rely entirely on Focal Loss to handle the 1.7% class imbalance during training.

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'A_norm': A_norm,
        'seq_len': config.SEQ_LEN,
        'n_nodes': num_nodes,
        'n_features': len(config.FEATURE_COLS)
    }
