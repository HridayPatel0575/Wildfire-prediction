"""
Sequence Builder — FWI-Gated LSTM Pipeline
==========================================
Loads FLAGED_data.csv, engineers all 35 features, builds 14-day rolling
windows per (lat, lon) location, and returns train/val/test arrays.

Key Design Decisions
---------------------
1. Temporal train/test split: TEST_YEARS (2019-2020) held out — no leakage.
2. Validation split carved from TRAINING data BEFORE SMOTE:
   - This ensures val set has the real ~1.7% fire distribution.
   - Training recall during Keras fit() therefore reflects test-time recall.
   - NOT doing this causes "recall looks great in training, collapses at test" bug.
3. SMOTE applied ONLY to training fold.
4. StandardScaler fitted on training data only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import config


# --- Feature Engineering ------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 19 engineered interaction features using EXACTLY the same
    formulas as All.ipynb (the notebook that produced FLAGED_data_enhanced.csv).
    """
    d = df.copy()

    # Interaction features
    d['temp_humidity_interaction']     = d['temp'] * d['humidity'] / 100
    d['wind_temp_interaction']         = d['wind_speed'] * (d['temp'] + 273.15)
    d['fwi_temp_ratio']                = d['FWI'] / (d['temp'] + 50)
    d['humidity_rainfall_interaction'] = d['humidity'] * (d['rainfall'] + 0.001)

    # Polynomial features
    d['temp_squared']       = d['temp'] ** 2
    d['fwi_squared']        = d['FWI'] ** 2
    d['wind_speed_squared'] = d['wind_speed'] ** 2
    d['humidity_squared']   = d['humidity'] ** 2

    # Physical drought indices
    d['dryness_index']     = (100 - d['humidity']) * (d['temp'] + 10) / 100
    d['fire_danger_index'] = (d['FWI'] * (100 - d['humidity']) * (d['temp'] + 10)) \
                             / (d['rainfall'] + 1)
    d['wind_dryness']      = d['wind_speed'] * (100 - d['humidity'])

    # Temperature deviation (global mean over full dataset)
    temp_mean = d['temp'].mean()
    d['temp_deviation'] = (d['temp'] - temp_mean).abs()

    # Ratio features
    d['fwi_humidity_ratio']  = d['FWI']       / (d['humidity']  + 1)
    d['temp_rainfall_ratio'] = d['temp']       / (d['rainfall']  + 0.1)
    d['wind_humidity_ratio'] = d['wind_speed'] / (d['humidity']  + 1)

    # Categorical features (same bins as All.ipynb)
    d['temp_category'] = pd.cut(
        d['temp'],
        bins=[-np.inf, 0, 10, 20, 30, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).fillna(0).astype(int)

    d['humidity_category'] = pd.cut(
        d['humidity'],
        bins=[0, 30, 50, 70, 100],
        labels=[0, 1, 2, 3]
    ).fillna(0).astype(int)

    d['fwi_category'] = pd.cut(
        d['FWI'],
        bins=[0, 5, 10, 20, np.inf],
        labels=[0, 1, 2, 3]
    ).fillna(0).astype(int)

    # Log transform
    d['log_fwi'] = np.log1p(d['FWI'])

    return d


# --- Load ---------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    print(f"[DataLoader] Loading: {config.RAW_DATA_FILE}")
    df = pd.read_csv(config.RAW_DATA_FILE)
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    print("[DataLoader] Engineering features...")
    df = engineer_features(df)

    missing = [c for c in config.FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[DataLoader] Missing feature columns: {missing}")

    print(f"[DataLoader] Loaded {len(df):,} rows | "
          f"{df['latitude'].nunique()} lats x {df['longitude'].nunique()} lons | "
          f"Dates: {df['date'].min().date()} to {df['date'].max().date()} | "
          f"Features: {len(config.FEATURE_COLS)}")
    return df


# --- Sequence Builder ---------------------------------------------------------

def build_sequences_for_location(
    loc_df:  pd.DataFrame,
    seq_len: int = config.SEQ_LEN,
) -> tuple:
    """
    For a single (lat, lon) group sorted by date, creates rolling windows.

    Returns
    -------
    X : np.ndarray  shape (N_windows, seq_len, n_features)
    y : np.ndarray  shape (N_windows,)  - label of LAST day in window
    """
    features = loc_df[config.FEATURE_COLS].values.astype(np.float32)
    labels   = loc_df[config.TARGET_COL].values.astype(np.float32)

    X_list, y_list = [], []
    n = len(features)
    for i in range(0, n - seq_len, config.STRIDE):
        X_list.append(features[i : i + seq_len])
        y_list.append(labels[i + seq_len - 1])

    if len(X_list) == 0:
        return (np.empty((0, seq_len, len(config.FEATURE_COLS)), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)


def build_all_sequences(df: pd.DataFrame) -> tuple:
    """
    Iterates over all (lat, lon) locations and builds sequences.

    Returns
    -------
    X      : (N, seq_len, n_features)
    y      : (N,)
    years  : (N,)  - year of the last day in each window (for split)
    """
    print(f"\n[SeqBuilder] Building {config.SEQ_LEN}-day sequences "
          f"over {len(config.FEATURE_COLS)} features...")
    all_X, all_y, all_years = [], [], []

    locations = df.groupby(config.LOCATION_COLS)
    n_locs    = len(locations)

    for idx, ((lat, lon), grp) in enumerate(locations):
        grp_sorted = grp.sort_values(config.DATE_COL).reset_index(drop=True)
        X_loc, y_loc = build_sequences_for_location(grp_sorted)

        if len(X_loc) == 0:
            continue

        years_loc = grp_sorted['year'].values[
            config.SEQ_LEN - 1 : config.SEQ_LEN - 1 + len(X_loc)
        ]

        all_X.append(X_loc)
        all_y.append(y_loc)
        all_years.append(years_loc)

        if (idx + 1) % 30 == 0 or (idx + 1) == n_locs:
            print(f"  Processed {idx + 1}/{n_locs} locations...")

    X     = np.concatenate(all_X,     axis=0)
    y     = np.concatenate(all_y,     axis=0)
    years = np.concatenate(all_years, axis=0)

    print(f"[SeqBuilder] Sequences: {len(X):,} | Shape: {X.shape} | "
          f"Fire rate: {y.mean() * 100:.2f}%")
    return X, y, years


# --- Temporal Train/Test Split ------------------------------------------------

def temporal_split(X, y, years) -> tuple:
    """
    Hold-out test split: TEST_YEARS (2019, 2020).
    Prevents temporal data leakage — required for journal evaluation.
    """
    test_mask  = np.isin(years, config.TEST_YEARS)
    train_mask = ~test_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\n[Split] Train: {len(X_train):,} sequences "
          f"(fire rate {y_train.mean()*100:.2f}%)")
    print(f"[Split] Test:  {len(X_test):,}  sequences "
          f"(fire rate {y_test.mean()*100:.2f}%)")
    return X_train, X_test, y_train, y_test


# --- Feature Scaling ----------------------------------------------------------

def scale_sequences(X_train, X_val, X_test, save_scaler=True) -> tuple:
    """
    StandardScaler fitted on training data only.
    Flattens (N, seq, feat) → (N*seq, feat), scales, reshapes back.
    """
    n_train, seq_len, n_feat = X_train.shape
    n_val                    = X_val.shape[0]
    n_test                   = X_test.shape[0]

    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(
                    X_train.reshape(-1, n_feat)
                ).reshape(n_train, seq_len, n_feat).astype(np.float32)
    X_va_sc   = scaler.transform(
                    X_val.reshape(-1, n_feat)
                ).reshape(n_val, seq_len, n_feat).astype(np.float32)
    X_te_sc   = scaler.transform(
                    X_test.reshape(-1, n_feat)
                ).reshape(n_test, seq_len, n_feat).astype(np.float32)

    if save_scaler:
        path = os.path.join(config.MODELS_DIR, "scaler.pkl")
        joblib.dump(scaler, path)
        print(f"[Scaler] Saved -> {path}")

    return X_tr_sc, X_va_sc, X_te_sc, scaler


# --- SMOTE (applied ONLY to training fold) ------------------------------------

from imblearn.over_sampling import SMOTE

def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Applies SMOTE to 3D temporal sequences to address class imbalance.
    MUST be called AFTER validation split so val set keeps natural distribution.

    Flattens (N, seq_len, n_features) -> (N, seq_len * n_features),
    applies SMOTE, reshapes back.
    """
    n_samples, seq_len, n_features = X_train.shape

    print(f"\n[SMOTE] Before: {len(y_train):,} samples | Fire rate: {y_train.mean()*100:.2f}%")

    X_flat          = X_train.reshape(n_samples, seq_len * n_features)
    smote           = SMOTE(random_state=config.RANDOM_STATE)
    X_sm_flat, y_sm = smote.fit_resample(X_flat, y_train)
    X_sm            = X_sm_flat.reshape(-1, seq_len, n_features)

    print(f"[SMOTE] After : {len(y_sm):,} samples | Fire rate: {y_sm.mean()*100:.2f}%")
    return X_sm, y_sm


# --- Master Pipeline ----------------------------------------------------------

def get_prepared_data() -> dict:
    """
    End-to-end data preparation.

    Pipeline:
    1. Load & engineer features
    2. Build 14-day sequences per location
    3. Temporal train/test split (test = 2019-2020)
    4. Carve out validation set from training (PRE-SMOTE) — real distribution
    5. Scale (fitted on training fold only)
    6. Apply SMOTE to training fold only

    Returns dict of all arrays + metadata for training.
    """
    df                               = load_raw_data()
    X, y, years                      = build_all_sequences(df)
    X_train_full, X_test, y_train_full, y_test = temporal_split(X, y, years)

    # ── CRITICAL: split validation BEFORE SMOTE ──────────────────────────────
    # Validation set must have the real fire distribution (~1.7%) so that
    # Keras training recall is honest and matches final test recall.
    # If val is carved from SMOTE'd data it looks great in training but
    # collapses at test time (the "recall gap" bug).
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y_train_full,
    )
    print(f"\n[Val split] Val: {len(X_val):,} sequences "
          f"(fire rate {y_val.mean()*100:.2f}%) — REAL distribution (pre-SMOTE)")

    # Scale using training fold statistics
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_sequences(
        X_train_raw, X_val, X_test
    )

    # Apply SMOTE only to training
    if getattr(config, 'USE_SMOTE', False):
        X_train_sc, y_train_raw = apply_smote(X_train_sc, y_train_raw)

    print(f"\n[Ready] Features: {len(config.FEATURE_COLS)} | "
          f"FWI idx: {config.FWI_FEATURE_IDX} ({config.FEATURE_COLS[config.FWI_FEATURE_IDX]})")
    print(f"[Ready] X_train: {X_train_sc.shape} | X_val: {X_val_sc.shape} | X_test: {X_test_sc.shape}")

    return {
        "X_train":    X_train_sc,
        "X_val":      X_val_sc,
        "y_train":    y_train_raw,
        "y_val":      y_val,
        "X_test":     X_test_sc,
        "y_test":     y_test,
        "scaler":     scaler,
        "n_features": X_train_sc.shape[2],
        "seq_len":    X_train_sc.shape[1],
        "fwi_idx":    config.FWI_FEATURE_IDX,
    }


if __name__ == "__main__":
    data = get_prepared_data()
    print("\nFeature list used by LSTM:")
    for i, col in enumerate(config.FEATURE_COLS):
        tag = " <- FWI gate" if i == config.FWI_FEATURE_IDX else ""
        print(f"  [{i:2d}] {col}{tag}")
