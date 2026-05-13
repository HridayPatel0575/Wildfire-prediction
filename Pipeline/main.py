import os
import config
from data_loader import load_enhanced_data, load_raw_data
from preprocessing import get_preprocessed_features
from model_training import build_model, train_model, evaluate_model
from plotting import plot_fwi_heatmap

def main():
    print("--- Fire Prediction Pipeline ---")
    
    # 1. Load Data
    print("\n1. Loading enhanced data...")
    df_enhanced = load_enhanced_data()
    print(f"Loaded {len(df_enhanced)} rows of dataset.")
    
    # 2. Preprocess Data
    print("\n2. Preprocessing data (Balancing & Scaling)...")
    # Separate identifier columns like valid_time, longitude, latitude if not needed for DL
    # Let's assume they were already dropped in FLAGED_data_enhanced.csv or need to be discarded for model
    drop_cols = ['valid_time', 'longitude', 'latitude', 'date']
    dl_features = df_enhanced.copy()
    for col in drop_cols:
        if col in dl_features.columns:
            dl_features = dl_features.drop(columns=[col])

    x_train, x_test, y_train, y_test = get_preprocessed_features(dl_features)
    print(f"Training shape: {x_train.shape}, Test shape: {x_test.shape}")
    
    # 3. Model Training
    print("\n3. Building & Training Model...")
    input_shape = x_train.shape[1]
    model = build_model(input_shape)
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # 4. Model Evaluation
    print("\n4. Evaluating Model...")
    metrics = evaluate_model(model, x_test, y_test)
    
    # 5. FWI Visualization Check
    print("\n5. Generating Visualization...")
    print(f"Loading raw data to plot FWI for date {config.TARGET_DATE}...")
    df_raw = load_raw_data()
    
    out_img = os.path.join(config.BASE_DIR, 'Pipeline', f'fwi_heatmap_{config.TARGET_DATE}.png')
    plot_fwi_heatmap(df_raw, config.TARGET_DATE, output_path=out_img)
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
