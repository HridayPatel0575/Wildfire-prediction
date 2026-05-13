import pandas as pd
import os
import config

def load_enhanced_data():
    """
    Loads the processed FLAGGED_data_enhanced.csv which contains engineered features,
    and drops the 'Unnamed: 0' file.
    """
    if not os.path.exists(config.ENHANCED_DATA_FILE):
        raise FileNotFoundError(f"Data file not found at {config.ENHANCED_DATA_FILE}")
        
    df = pd.read_csv(config.ENHANCED_DATA_FILE)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

def load_raw_data():
    """
    Loads historical fire data for plotting and interpolation mapping.
    """
    if not os.path.exists(config.RAW_DATA_FILE):
        raise FileNotFoundError(f"Data file not found at {config.RAW_DATA_FILE}")
    df = pd.read_csv(config.RAW_DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_fire_data():
    """
    Loads the historical fire locations data.
    """
    if not os.path.exists(config.FIRE_DATA_FILE):
        raise FileNotFoundError(f"Data file not found at {config.FIRE_DATA_FILE}")
    df = pd.read_csv(config.FIRE_DATA_FILE)
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    return df


def load_all_data():
    fwi_df = load_enhanced_data()
    fires_df = load_fire_data()
    return fwi_df, fires_df


    