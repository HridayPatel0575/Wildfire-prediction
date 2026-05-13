import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd

import config
import data_loader

def plot_fwi_and_fires_heatmap(fwi_df, fires_df, target_date, output_path=None):
    """
    Plots a heatmap of FWI scores for a specific date and overlays actual fire locations.
    """
    # Filter FWI data by date
    if 'date' in fwi_df.columns:
        df_date = fwi_df[fwi_df['date'] == target_date].copy()
    else:
        # If it's already filtered or doesn't have a date column
        df_date = fwi_df.copy()

    if df_date.empty:
        print(f"No FWI data found for date {target_date}")
        return

    # Filter fire locations by date
    fires_date_df = fires_df[fires_df['acq_date'] == target_date].copy()
    
    if not all(col in df_date.columns for col in ['longitude', 'latitude', 'FWI']):
        print("Missing required columns for plotting FWI (longitude, latitude, FWI).")
        return

    lon = df_date['longitude'].values
    lat = df_date['latitude'].values
    fwi = df_date['FWI'].values

    # Determine grid boundaries
    lon_min, lon_max = lon.min() - 0.5, lon.max() + 0.5
    lat_min, lat_max = lat.min() - 0.5, lat.max() + 0.5

    # Create grid
    grid_x, grid_y = np.mgrid[lon_min:lon_max:500j, lat_min:lat_max:500j]

    # Interpolate using scipy.interpolate.griddata
    grid_z = griddata((lon, lat), fwi, (grid_x, grid_y), method='cubic')

    # Plot
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='YlOrRd', extend='both')
    plt.colorbar(contour, label='FWI Score')
    
    # Plot data points used for interpolation (optional, made less visible)
    plt.scatter(lon, lat, c='gray', s=5, alpha=0.2, label='Weather Stations')
    
    # Overlay actual fire locations
    if not fires_date_df.empty:
        fire_lons = fires_date_df['longitude'].values
        fire_lats = fires_date_df['latitude'].values
        plt.scatter(fire_lons, fire_lats, c='blue', marker='*', s=150, edgecolor='black', 
                    linewidth=0.5, label='Actual Fire Locations')
        print(f"Found {len(fire_lons)} fires on {target_date}")
    else:
        print(f"No historical fires found on {target_date}")

    plt.title(f'Fire Weather Index (FWI) & Fire Locations on {target_date}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap with fire locations saved to {output_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    print("Loading FWI data...")
    try:
        fwi_df = data_loader.load_enhanced_data()
    except Exception as e:
        print(f"Could not load enhanced data, falling back to raw data: {e}")
        fwi_df = data_loader.load_raw_data()
        
    print("Loading historical fire data...")
    fires_df = data_loader.load_fire_data()
    
    date_to_plot = config.TARGET_DATE
    output_img = os.path.join(config.BASE_DIR, f"fwi_fires_heatmap_{date_to_plot}.png")
    
    print(f"Plotting heatmap for {date_to_plot}...")
    plot_fwi_and_fires_heatmap(fwi_df, fires_df, target_date=date_to_plot, output_path=output_img)
    print("Process complete.")
