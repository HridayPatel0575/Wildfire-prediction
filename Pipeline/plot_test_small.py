import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Minimal configuration
csv_file = "FLAGED_data_enhanced.csv"
fire_file = "Main_Data/uttarakhand_fires.csv"
output_img = "test_heatmap_small.png"

print("Loading a subset of FWI data...")
df = pd.read_csv(csv_file, nrows=5000) # Only 5000 rows for speed
print(f"Loaded {len(df)} rows.")

print("Loading fire data...")
fires = pd.read_csv(fire_file)

print("Interpolating...")
lon = df['longitude'].values
lat = df['latitude'].values
fwi = df['FWI'].values

grid_x, grid_y = np.mgrid[lon.min():lon.max():100j, lat.min():lat.max():100j]
grid_z = griddata((lon, lat), fwi, (grid_x, grid_y), method='linear')

print("Plotting...")
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='YlOrRd')
plt.scatter(fires['longitude'], fires['latitude'], c='blue', s=10, alpha=0.5)
plt.savefig(output_img)
plt.close()
print(f"Test heatmap saved to {output_img}")
