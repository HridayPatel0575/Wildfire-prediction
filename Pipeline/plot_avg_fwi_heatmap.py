import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd

import config
import data_loader

def plot_avg_fwi_and_all_fires(fwi_df, fires_df, output_path=None):
    """
    Plots a heatmap of AVERAGE FWI scores across all timestamps and overlays all fire locations.
    """
    if not all(col in fwi_df.columns for col in ['longitude', 'latitude', 'FWI']):
        print("Missing required columns for plotting FWI (longitude, latitude, FWI).")
        return

    print("Calculating average FWI per location...")
    # Group by location and average FWI
    avg_fwi_df = fwi_df.groupby(['longitude', 'latitude'])['FWI'].mean().reset_index()

    lon = avg_fwi_df['longitude'].values
    lat = avg_fwi_df['latitude'].values
    fwi = avg_fwi_df['FWI'].values

    # Determine grid boundaries
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    # Create grid
    grid_x, grid_y = np.mgrid[lon_min:lon_max:200j, lat_min:lat_max:200j]

    print(f"Interpolating FWI data on a {200}x{200} grid...")
    # Interpolate using scipy.interpolate.griddata
    grid_z = griddata((lon, lat), fwi, (grid_x, grid_y), method='cubic')

    # Plot
    plt.figure(figsize=(14, 10))
    
    # Heatmap
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='YlOrRd', extend='both')
    plt.colorbar(contour, label='Average FWI Score')
    
    # Overlay data points used (optional)
    plt.scatter(lon, lat, c='gray', s=2, alpha=0.1, label='Data Points')
    
    # Overlay ALL fire locations
    if not fires_df.empty:
        # Filter fires to be within FWI grid boundaries
        mask = (fires_df['longitude'] >= lon_min) & (fires_df['longitude'] <= lon_max) & \
               (fires_df['latitude'] >= lat_min) & (fires_df['latitude'] <= lat_max)
        filtered_fires = fires_df[mask]
        
        fire_lons = filtered_fires['longitude'].values
        fire_lats = filtered_fires['latitude'].values
        plt.scatter(fire_lons, fire_lats, c='blue', marker='o', s=30, alpha=0.5, 
                    edgecolor='white', linewidth=0.2, label='Recorded Fire Locations')
        print(f"Overlaying {len(fire_lons)} historical fire locations (filtered to boundary).")
    else:
        print("No historical fire data provided.")

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)

    plt.title('Average Fire Weather Index (FWI) & Cumulative Fire Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Average FWI heatmap saved to {output_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    print("Loading FWI data...")
    try:
        # Prefer enhanced data/processed data
        fwi_df = data_loader.load_enhanced_data()
    except Exception as e:
        print(f"Could not load enhanced data, falling back to raw: {e}")
        fwi_df = data_loader.load_raw_data()
        
    print("Loading historical fire data...")
    fires_df = data_loader.load_fire_data()
    
    output_img = os.path.join(config.BASE_DIR, "avg_fwi_fires_heatmap.png")
    
    print("Generating comprehensive heatmap...")
    plot_avg_fwi_and_all_fires(fwi_df, fires_df, output_path=output_img)
    print("Process complete.")
