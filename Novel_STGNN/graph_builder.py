import numpy as np
import pandas as pd
from typing import Tuple

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def build_adjacency_matrix(df: pd.DataFrame, threshold_km: float) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extracts unique geographic nodes and builds an adjacency matrix.
    Two nodes are connected (A_ij = 1) if their Haversine distance < threshold_km.
    Also adds self-loops (A_ii = 1).
    """
    # Extract unique lat/lon pairs to form the nodes of the graph
    nodes = df[['latitude', 'longitude']].drop_duplicates().sort_values(by=['latitude', 'longitude']).reset_index(drop=True)
    num_nodes = len(nodes)
    print(f"[GraphBuilder] Found {num_nodes} unique geographic nodes.")
    
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    edges_count = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                A[i, j] = 1.0 # Self-loop
            else:
                dist = haversine(
                    nodes.loc[i, 'latitude'], nodes.loc[i, 'longitude'],
                    nodes.loc[j, 'latitude'], nodes.loc[j, 'longitude']
                )
                if dist <= threshold_km:
                    A[i, j] = 1.0
                    edges_count += 1
                    
    # Note: edges_count includes both i->j and j->i, so we divide by 2 for unique undirected edges.
    print(f"[GraphBuilder] Adjacency Matrix Built: shape {A.shape}, "
          f"{edges_count // 2} undirected edges (distance threshold {threshold_km} km).")
          
    # Normalize adjacency matrix (A_hat = D^-0.5 * A * D^-0.5) for GCN stability
    degree = np.sum(A, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    A_normalized = D_inv_sqrt.dot(A).dot(D_inv_sqrt)
    
    return A_normalized, nodes
