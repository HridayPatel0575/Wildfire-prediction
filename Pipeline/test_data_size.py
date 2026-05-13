import pandas as pd
import os
import time

filepath = "FLAGED_data_enhanced.csv"
print(f"Checking file: {filepath}")
start = time.time()
df = pd.read_csv(filepath)
end = time.time()
print(f"Loaded {len(df)} rows in {end-start:.2f} seconds.")

unique_locs = df.groupby(['latitude', 'longitude']).size().reset_index()
print(f"Number of unique locations: {len(unique_locs)}")
print("First few unique locations:")
print(unique_locs.head())
