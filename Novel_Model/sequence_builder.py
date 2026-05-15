"""
Sequence Builder — FWI-Gated LSTM (Novel Model)
================================================
Journal Publication | Wildfire Prediction — Indian Subcontinent

Builds per-location 14-day rolling sequences for the LSTM models.

Pipeline (leak-free):
  1. Load FLAGED_data.csv and engineer all 35 features.
  2. Sort by (date, latitude, longitude) and build one 14-day sequence
     per (location, date) pair — treating each grid cell independently.
  3. Temporal hold-out: sequences whose TARGET day falls in TEST_YEARS
     become the test set. All prior sequences become training candidates.
  4. Validation carve-out: 15% of training sequences are held out BEFORE
     SMOTE — so val_recall during training reflects real ~1.7% fire rate.
  5. StandardScaler fitted ONLY on training fold (prevents leakage).
  6. SMOTE applied to training fold ONLY → balanced training set.

Returns:
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                    seq_len, n_features, fwi_idx
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config


# ---------------------------------------------------------------------------
# Feature Engineering (mirrors the paper's baseline exactly)
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all 19 engineered interaction features in-place."""
    print("[SequenceBuilder] Engineering features...")

    df['temp_humidity_interaction']   = df['temp'] * df['humidity'] / 100.0
    df['wind_temp_interaction']       = df['wind_speed'] * (df['temp'] + 273.15)
    df['fwi_temp_ratio']              = df['FWI'] / (df['temp'] + 50.0)
    df['humidity_rainfall_interaction'] = df['humidity'] * (df['rainfall'] + 0.001)

    df['temp_squared']      = df['temp'] ** 2
    df['fwi_squared']       = df['FWI'] ** 2
    df['wind_speed_squared']= df['wind_speed'] ** 2
    df['humidity_squared']  = df['humidity'] ** 2

    df['dryness_index']     = (100 - df['humidity']) * (df['temp'] + 10) / 100.0
    df['fire_danger_index'] = (
        df['FWI'] * (100 - df['humidity']) * (df['temp'] + 10)
        / (df['rainfall'] + 1)
    )
    df['wind_dryness']      = df['wind_speed'] * (100 - df['humidity'])

    global_mean_temp        = df['temp'].mean()
    df['temp_deviation']    = np.abs(df['temp'] - global_mean_temp)

    df['fwi_humidity_ratio']  = df['FWI'] / (df['humidity'] + 1)
    df['temp_rainfall_ratio'] = df['temp'] / (df['rainfall'] + 0.1)
    df['wind_humidity_ratio'] = df['wind_speed'] / (df['humidity'] + 1)

    df['temp_category']     = pd.cut(
        df['temp'], bins=[-np.inf, 0, 10, 20, 30, np.inf], labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(0)
    df['humidity_category'] = pd.cut(
        df['humidity'], bins=[-np.inf, 30, 50, 70, np.inf], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0)
    df['fwi_category']      = pd.cut(
        df['FWI'], bins=[-np.inf, 5, 10, 20, np.inf], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0)

    df['log_fwi'] = np.log1p(df['FWI'])
    return df


# ---------------------------------------------------------------------------
# Sequence Construction
# ---------------------------------------------------------------------------

def _build_sequences(
    location_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    seq_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds rolling windows of length `seq_len` for a single location.

    Returns:
        X      : (n_seqs, seq_len, n_features) float32
        y      : (n_seqs,) int32
        years  : (n_seqs,) int  — year of the TARGET (last+1) day, for splitting
    """
    loc_df = location_df.sort_values('date').reset_index(drop=True)
    feat   = loc_df[feature_cols].values.astype(np.float32)
    targ   = loc_df[target_col].values.astype(np.int32)
    dates  = pd.to_datetime(loc_df['date'])

    X, y, years = [], [], []
    n = len(loc_df)
    for i in range(0, n - seq_len, stride):
        X.append(feat[i : i + seq_len])
        y.append(targ[i + seq_len])
        years.append(dates.iloc[i + seq_len].year)

    if not X:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), \
               np.empty(0, dtype=np.int32), \
               np.empty(0, dtype=np.int32)

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(years, dtype=np.int32))


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def get_prepared_data() -> dict:
    """
    Full, leak-free data preparation pipeline for the LSTM models.

    Returns a dict containing:
        X_train, y_train  : SMOTE-balanced training set
        X_val,   y_val    : Real-distribution validation set (pre-SMOTE carve)
        X_test,  y_test   : Temporal hold-out (2019-2020)
        seq_len            : int
        n_features         : int
        fwi_idx            : int  — column index of 'FWI' in FEATURE_COLS
    """
    # ------------------------------------------------------------------
    # 1. Load & engineer
    # ------------------------------------------------------------------
    print(f"[SequenceBuilder] Loading: {config.RAW_DATA_FILE}")
    df = pd.read_csv(config.RAW_DATA_FILE, parse_dates=['date'])
    df = _engineer_features(df)

    # Validate all expected feature columns are present
    missing = [c for c in config.FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[SequenceBuilder] Missing feature columns: {missing}")

    # ------------------------------------------------------------------
    # 2. Build per-location sequences (flat — all locations pooled)
    # ------------------------------------------------------------------
    print(f"[SequenceBuilder] Building {config.SEQ_LEN}-day sequences per location...")
    locations = df.groupby(config.LOCATION_COLS)
    n_locs    = len(locations)

    all_X, all_y, all_years = [], [], []
    for i, (loc_key, loc_df) in enumerate(locations):
        X_loc, y_loc, yrs_loc = _build_sequences(
            loc_df, config.FEATURE_COLS, config.TARGET_COL,
            seq_len=config.SEQ_LEN, stride=config.STRIDE,
        )
        if len(X_loc) > 0:
            all_X.append(X_loc)
            all_y.append(y_loc)
            all_years.append(yrs_loc)
        if (i + 1) % 30 == 0:
            print(f"  ... {i+1}/{n_locs} locations processed")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    years_all = np.concatenate(all_years, axis=0)

    print(f"[SequenceBuilder] Total sequences : {len(X_all):,}")
    print(f"[SequenceBuilder] Fire rate (all)  : {y_all.mean()*100:.2f}%")

    # ------------------------------------------------------------------
    # 3. Temporal train / test split — NO shuffling to preserve temporal order
    # ------------------------------------------------------------------
    test_mask  = np.isin(years_all, config.TEST_YEARS)
    train_mask = ~test_mask

    X_trainval, y_trainval = X_all[train_mask], y_all[train_mask]
    X_test,     y_test     = X_all[test_mask],  y_all[test_mask]

    print(f"[Split] Train+Val sequences: {len(X_trainval):,} "
          f"(fire={y_trainval.mean()*100:.2f}%)")
    print(f"[Split] Test sequences     : {len(X_test):,} "
          f"(fire={y_test.mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # 4. Carve validation fold from training data BEFORE SMOTE & BEFORE scaling
    #    Use stratified split to ensure val has some fire examples.
    # ------------------------------------------------------------------
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_trainval, y_trainval,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y_trainval,
    )
    print(f"[Split] Train (pre-SMOTE)  : {len(X_train_raw):,} "
          f"(fire={y_train_raw.mean()*100:.2f}%)")
    print(f"[Split] Validation (real)  : {len(X_val_raw):,} "
          f"(fire={y_val_raw.mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # 5. Fit scaler on training fold ONLY (prevent leakage)
    # ------------------------------------------------------------------
    print("[SequenceBuilder] Fitting StandardScaler on training fold...")
    n_train, seq_len, n_feat = X_train_raw.shape
    scaler = StandardScaler()
    # Reshape to 2D for scaler, then back
    X_train_2d = X_train_raw.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    def _scale(X: np.ndarray) -> np.ndarray:
        sh = X.shape
        scaled = scaler.transform(X.reshape(-1, sh[-1]))
        scaled = np.clip(scaled, -10.0, 10.0)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0)
        return scaled.reshape(sh).astype(np.float32)

    X_train_scaled = _scale(X_train_raw)
    X_val_scaled   = _scale(X_val_raw)
    X_test_scaled  = _scale(X_test)

    # Save scaler for inference / notebook reproducibility
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[SequenceBuilder] Scaler saved: {scaler_path}")

    # ------------------------------------------------------------------
    # 6. SMOTE on training fold ONLY
    # ------------------------------------------------------------------
    if config.USE_SMOTE:
        try:
            from imblearn.over_sampling import SMOTE
            print("[SMOTE] Applying SMOTE to training fold...")
            n_tr, sl, nf = X_train_scaled.shape
            X_2d         = X_train_scaled.reshape(n_tr, sl * nf)
            sm           = SMOTE(random_state=config.RANDOM_STATE)
            X_2d_res, y_train_balanced = sm.fit_resample(X_2d, y_train_raw)
            X_train_balanced = X_2d_res.reshape(-1, sl, nf).astype(np.float32)
            y_train_balanced = y_train_balanced.astype(np.float32)
            print(f"[SMOTE] Training after SMOTE: {len(X_train_balanced):,} "
                  f"(fire={y_train_balanced.mean()*100:.1f}%)")
        except ImportError:
            print("[SMOTE] imbalanced-learn not installed — skipping SMOTE. "
                  "Install with: pip install imbalanced-learn")
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train_raw.astype(np.float32)
    else:
        print("[SMOTE] Disabled in config — training on natural distribution.")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train_raw.astype(np.float32)

    # ------------------------------------------------------------------
    # 7. Return prepared data dict
    # ------------------------------------------------------------------
    fwi_idx = config.FEATURE_COLS.index('FWI')

    return {
        'X_train':    X_train_balanced,
        'y_train':    y_train_balanced,
        'X_val':      X_val_scaled,
        'y_val':      y_val_raw.astype(np.float32),
        'X_test':     X_test_scaled,
        'y_test':     y_test.astype(np.float32),
        'seq_len':    seq_len,
        'n_features': n_feat,
        'fwi_idx':    fwi_idx,
    }