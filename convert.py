import os
import torch
import pyvista as pv
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from tqdm import tqdm

def process_folder(folder_path, folder_name, output_dir):
    """
    Processes a single simulation folder, CROPS the data, and creates
    a single PyTorch Geometric Data object.
    """
    try:
        vtu_path = os.path.join(folder_path, f"{folder_name}_internal.vtu")
        aerofoil_vtp_path = os.path.join(folder_path, f"{folder_name}_aerofoil.vtp")

        if not os.path.exists(vtu_path) or not os.path.exists(aerofoil_vtp_path):
            return

        vtu_mesh = pv.read(vtu_path)
        pos_full = torch.tensor(vtu_mesh.points[:, :2], dtype=torch.float32)
        
        mask = (pos_full[:, 0] >= -2.0) & (pos_full[:, 0] <= 4.0) & \
               (pos_full[:, 1] >= -1.5) & (pos_full[:, 1] <= 1.5)
        
        pos = pos_full[mask]
        num_nodes = pos.shape[0]

        velocity = torch.tensor(vtu_mesh.point_data['U'][:, :2], dtype=torch.float32)[mask]
        pressure = torch.tensor(vtu_mesh.point_data['p'], dtype=torch.float32).unsqueeze(-1)[mask]
        nut = torch.tensor(vtu_mesh.point_data['nut'], dtype=torch.float32).unsqueeze(-1)[mask]
        y = torch.cat([velocity, pressure, nut], dim=1)

        sdf = torch.tensor(vtu_mesh.point_data['implicit_distance'], dtype=torch.float32).unsqueeze(-1)[mask]

        try:
            angle_of_attack_deg = float(folder_name.split('_')[-1])
            angle_of_attack_rad = np.deg2rad(angle_of_attack_deg)
            vx_freestream = np.cos(angle_of_attack_rad)
            vy_freestream = np.sin(angle_of_attack_rad)
            freestream = torch.tensor([vx_freestream, vy_freestream], dtype=torch.float32).repeat(num_nodes, 1)
        except (ValueError, IndexError):
            freestream = torch.zeros((num_nodes, 2), dtype=torch.float32)

        normals = torch.zeros((num_nodes, 2), dtype=torch.float32)
        aerofoil_mesh = pv.read(aerofoil_vtp_path)
        surface_points = aerofoil_mesh.points[:, :2]
        surface_normals = torch.tensor(aerofoil_mesh.point_data['Normals'][:, :2], dtype=torch.float32)
        
        kdtree = cKDTree(pos.numpy())
        _, indices = kdtree.query(surface_points)
        normals[indices] = surface_normals
        
        x = torch.cat([sdf, normals, freestream], dim=1)
        
        data = Data(x=x, y=y, pos=pos)
        output_path = os.path.join(output_dir, f"{folder_name}.pt")
        torch.save(data, output_path)

    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")

def main():
    # --- Configuration ---
    dataset_root = "../data/"
    output_dir = "../processed_data/"
    train_split_ratio = 0.8  # 80% for training, 20% for validation/testing
    # ---------------------

    os.makedirs(output_dir, exist_ok=True)
    
    # ==================================================================
    # PART 1: CONVERSION AND CROPPING
    # ==================================================================
    print("--- PART 1: CONVERSION AND CROPPING ---")
    
    # Clean the output directory first
    print(f"Cleaning old files in '{output_dir}'...")
    for filename in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, filename))
        
    print("Starting conversion and cropping of raw data...")
    folder_names = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    for folder_name in tqdm(folder_names, desc="Processing folders"):
        folder_path = os.path.join(dataset_root, folder_name)
        process_folder(folder_path, folder_name, output_dir)
        
    print(f"\n✅ Conversion and cropping complete!")

    # ==================================================================
    # PART 2: CALCULATION OF NORMALIZATION STATISTICS
    # ==================================================================
    print("\n--- PART 2: CALCULATING NORMALIZATION STATS ---")
    
    all_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])
    split_index = int(len(all_files) * train_split_ratio)
    train_files = all_files[:split_index]
    
    if not train_files:
        print("❌ Error: No training files found to calculate statistics. Halting.")
        return
        
    print(f"Calculating statistics from {len(train_files)} training files...")

    # Accumulators for calculating mean and std for 'y' (4 features: ux, uy, p, nut)
    count = 0
    mean = torch.zeros(4)
    std = torch.zeros(4)
    
    for filename in tqdm(train_files, desc="Calculating Mean"):
        file_path = os.path.join(output_dir, filename)
        data = torch.load(file_path, weights_only=False)
        mean += data.y.mean(dim=0)
    mean /= len(train_files)

    for filename in tqdm(train_files, desc="Calculating Std Dev"):
        file_path = os.path.join(output_dir, filename)
        data = torch.load(file_path, weights_only=False)
        std += (data.y - mean).pow(2).mean(dim=0)
    std = torch.sqrt(std / len(train_files))
    
    stats_path = os.path.join(output_dir, 'normalization_stats.pt')
    torch.save({'mean': mean, 'std': std}, stats_path)

    print("\n✅ Normalization statistics calculated and saved!")
    print(f"   - Mean (ux, uy, p, nut): {mean.tolist()}")
    print(f"   - Std Dev (ux, uy, p, nut): {std.tolist()}")
    print(f"   - Saved to: {stats_path}")
    print("\n--- DATA PREPARATION PIPELINE COMPLETE ---")

if __name__ == '__main__':
    # Ensure necessary libraries are installed
    try:
        import scipy
    except ImportError:
        print("Error: 'scipy' is not installed. Please run 'pip install scipy'")
    else:
        main()