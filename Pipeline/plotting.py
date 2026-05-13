import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import os

def plot_fwi_heatmap(df, date, output_path=None):
    """
    Plots a heatmap of FWI scores for a specific date using interpolation.
    """
    df_date = df[df['date'] == date].copy()
    if df_date.empty:
        print(f"No data found for date {date}")
        return
        
    # We assume 'longitude', 'latitude', and 'FWI' are present in the dataframe
    if not all(col in df_date.columns for col in ['longitude', 'latitude', 'FWI']):
        print("Missing required columns for plotting (longitude, latitude, FWI).")
        return

    lon = df_date['longitude'].values
    lat = df_date['latitude'].values
    fwi = df_date['FWI'].values

    # Determine grid boundaries
    lon_min, lon_max = lon.min() - 1, lon.max() + 1
    lat_min, lat_max = lat.min() - 1, lat.max() + 1

    # Create grid
    grid_x, grid_y = np.mgrid[lon_min:lon_max:500j, lat_min:lat_max:500j]

    # Interpolate using scipy.interpolate.griddata
    grid_z = griddata((lon, lat), fwi, (grid_x, grid_y), method='cubic')

    # Plot
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='hot_r', extend='both')
    plt.colorbar(contour, label='FWI Score')
    
    plt.scatter(lon, lat, c=fwi, cmap='hot_r', edgecolor='k', s=20, alpha=0.5)
    
    plt.title(f'Fire Weather Index (FWI) Heatmap on {date}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    else:
        plt.show()
    plt.close()
