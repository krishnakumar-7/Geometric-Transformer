import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Import your custom classes
from src.data_handling.dataset import AirfoilDataset
from src.model.transformer import GeometricTransformer
from src.data_handling.dataset import SubsampleNodes 

def train():
    """Main function to run the training and validation process."""

    # ==========================================================================
    # 1. Configuration & Hyperparameters
    # ==========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Training params ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8          # If you still get memory errors, try reducing this to 4 or 2
    EPOCHS = 100
    VALIDATION_SPLIT = 0.1
    SUBSAMPLE_NODES = 32000 # <-- DEFINE SUBSAMPLE SIZE (as per your project plan)
    
    # --- Model params ---
    IN_FEATURES = 5
    OUT_FEATURES = 4
    D_MODEL = 256
    NUM_LAYERS = 6
    NUM_HEADS = 8
    D_FF = 1024

    # ==========================================================================
    # 2. Data Loading & Splitting
    # ==========================================================================
    # Create an instance of the transform
    node_transform = SubsampleNodes(num_nodes=SUBSAMPLE_NODES)

    # Pass the transform to your dataset during initialization
    full_dataset = AirfoilDataset(root='.', transform=node_transform)
    
    num_samples = len(full_dataset)
    num_val = int(VALIDATION_SPLIT * num_samples)
    num_train = num_samples - num_val
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
    
    print(f"Dataset size: {num_samples}")
    print(f"Subsampling each graph to {SUBSAMPLE_NODES} nodes.")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Set num_workers=0 for Colab with Google Drive
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
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

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters.")

    # ==========================================================================
    # 4. The Training Loop
    # ==========================================================================
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            predictions = model(batch)
            loss = loss_fn(predictions, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in val_pbar:
                batch = batch.to(device)
                predictions = model(batch)
                loss = loss_fn(predictions, batch.y)
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save models to Google Drive to persist them
            save_dir = '/content/drive/My Drive/colab_models'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"✨ New best model saved to Google Drive with validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    train()