import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import os

# Import your custom classes
from src.data_handling.dataset import AirfoilDataset
from src.model.transformer import GeometricTransformer

def quick_test():
    """
    A lightweight version of the training script that runs on a small
    subset of data for a few batches to quickly test the pipeline.
    """
    print("--- 🚀 Running Quick Training Pipeline Test ---")

    # ==========================================================================
    # 1. Lightweight Configuration
    # ==========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Test params ---
    SUBSET_SIZE = 32         # Use only 32 samples from the dataset
    BATCH_SIZE = 8
    MAX_BATCHES_TO_RUN = 4   # Stop after this many batches

    # --- Lightweight Model params ---
    IN_FEATURES = 5
    OUT_FEATURES = 4
    D_MODEL = 128            # Smaller model for faster forward/backward pass
    NUM_LAYERS = 4
    NUM_HEADS = 4
    D_FF = 512

    # ==========================================================================
    # 2. Data Loading (using a small subset)
    # ==========================================================================
    full_dataset = AirfoilDataset(root='.')
    
    # Create a small subset of the full dataset
    subset_indices = range(SUBSET_SIZE)
    test_subset = Subset(full_dataset, subset_indices)
    
    print(f"Using a subset of {len(test_subset)} samples.")
    
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ==========================================================================
    # 3. Model, Optimizer, and Loss Function
    # ==========================================================================
    model = GeometricTransformer(
        in_features=IN_FEATURES,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        out_features=OUT_FEATURES
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters.")

    # ==========================================================================
    # 4. Simplified Training Loop
    # ==========================================================================
    model.train()
    print("\nStarting training for a few batches...")
    
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        predictions = model(batch)
        loss = loss_fn(predictions, batch.y)
        
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {i+1}/{MAX_BATCHES_TO_RUN} -> Loss: {loss.item():.6f}")
        
        # Stop after a few batches
        if i >= MAX_BATCHES_TO_RUN - 1:
            break
            
    print("\n--- ✅ Quick training test completed successfully! ---")


if __name__ == '__main__':
    quick_test()