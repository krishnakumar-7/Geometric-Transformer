import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def profile_dataset():
    # --- Configuration ---
    processed_data_dir = '../processed_data/'
    # -------------------

    print(f"--- Starting Profiling for Dataset at '{processed_data_dir}' ---")

    if not os.path.exists(processed_data_dir):
        print(f"❌ ERROR: Directory not found at '{processed_data_dir}'")
        return

    # 1. Find all processed data files
    files = sorted([f for f in os.listdir(processed_data_dir) if f.endswith('.pt') and f != 'normalization_stats.pt'])
    if not files:
        print("❌ ERROR: No .pt files found in the directory.")
        return
        
    print(f"Found {len(files)} simulation files to analyze.")

    # 2. Initialize lists to store statistics from each file
    node_counts = []
    # We have 5 input features and 4 output features
    x_mins, x_maxs, x_means, x_stds = [], [], [], []
    y_mins, y_maxs, y_means, y_stds = [], [], [], []

    # 3. Iterate through all files and collect data
    for filename in tqdm(files, desc="Profiling files"):
        file_path = os.path.join(processed_data_dir, filename)
        try:
            data = torch.load(file_path, weights_only=False)

            # Basic validation
            if data.num_nodes == 0:
                continue
            if data.x.shape[1] != 5 or data.y.shape[1] != 4:
                print(f"\nWarning: Skipping {filename} due to incorrect shape.")
                continue

            node_counts.append(data.num_nodes)

            # Collect stats for input features 'x'
            x_mins.append(data.x.min(dim=0).values)
            x_maxs.append(data.x.max(dim=0).values)
            x_means.append(data.x.mean(dim=0))
            x_stds.append(data.x.std(dim=0))
            
            # Collect stats for output targets 'y'
            y_mins.append(data.y.min(dim=0).values)
            y_maxs.append(data.y.max(dim=0).values)
            y_means.append(data.y.mean(dim=0))
            y_stds.append(data.y.std(dim=0))

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            
    if not node_counts:
        print("❌ ERROR: Could not process any files.")
        return

    # 4. Aggregate and Display Statistics 📊
    print("\n--- DATASET PROFILE ---")
    
    # Node count statistics
    print("\n## Mesh Size (Nodes per Simulation)")
    print(f"  - Min Nodes:    {np.min(node_counts):,}")
    print(f"  - Max Nodes:    {np.max(node_counts):,}")
    print(f"  - Average Nodes:  {int(np.mean(node_counts)):,}")
    print(f"  - Median Nodes:   {int(np.median(node_counts)):,}")
    
    # Feature statistics
    x_mins, x_maxs = torch.stack(x_mins), torch.stack(x_maxs)
    x_means, x_stds = torch.stack(x_means), torch.stack(x_stds)
    y_mins, y_maxs = torch.stack(y_mins), torch.stack(y_maxs)
    y_means, y_stds = torch.stack(y_means), torch.stack(y_stds)

    stats_data = {
        'Feature': [
            'SDF', 'Normal X', 'Normal Y', 'Freestream Vx', 'Freestream Vy',
            'Velocity X (ux)', 'Velocity Y (uy)', 'Pressure (p)', 'Turbulent Viscosity (nut)'
        ],
        'Global Min': torch.cat([x_mins.min(dim=0).values, y_mins.min(dim=0).values]).tolist(),
        'Global Max': torch.cat([x_maxs.max(dim=0).values, y_maxs.max(dim=0).values]).tolist(),
        'Average Mean': torch.cat([x_means.mean(dim=0), y_means.mean(dim=0)]).tolist(),
        'Average Std Dev': torch.cat([x_stds.mean(dim=0), y_stds.mean(dim=0)]).tolist(),
    }
    
    df = pd.DataFrame(stats_data)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n## Feature & Target Field Statistics")
    print(df.to_string(index=False))

    # 5. Visualize Distributions 📈
    plt.figure(figsize=(10, 6))
    plt.hist(node_counts, bins=50, edgecolor='black')
    plt.title('Distribution of Nodes per Simulation')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency (Number of Simulations)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    print("\nDisplaying plot for node distribution...")
    plt.show()

if __name__ == '__main__':
    profile_dataset()