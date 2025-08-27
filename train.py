import sys
import os

# Add the project's root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Import your custom classes
from src.data_handling.dataset import AirfoilDataset
from src.data_handling.transforms import SubsampleNodes
from src.model.transformer import GeometricTransformer

def train():
    """Main function to run the training and validation process."""

    # ==========================================================================
    # 1. Configuration & Hyperparameters
    # ==========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Training params ---
    LEARNING_RATE = 0.001
    BATCH_SIZE = 2       # Reduced to 1 for extreme memory constraints
    EPOCHS = 50
    VALIDATION_SPLIT = 0.1
    SUBSAMPLE_NODES = 4000  # Drastically reduced from 12000
    
    # --- Model params (much smaller) ---
    IN_FEATURES = 5
    OUT_FEATURES = 4
    D_MODEL = 128           # Reduced from 256
    NUM_LAYERS = 3          # Reduced from 6
    NUM_HEADS = 4           # Reduced from 8
    D_FF = 256              # Reduced from 1024

    # ==========================================================================
    # 2. Data Loading & Splitting
    # ==========================================================================
    try:
        node_transform = SubsampleNodes(num_nodes=SUBSAMPLE_NODES)
        full_dataset = AirfoilDataset(root='.', transform=node_transform)
        
        num_samples = len(full_dataset)
        if num_samples == 0:
            raise ValueError("Dataset is empty. Check your data files.")
        
        num_val = int(VALIDATION_SPLIT * num_samples)
        num_train = num_samples - num_val
        
        # Ensure we have at least one sample for training and validation
        if num_train == 0 or num_val == 0:
            raise ValueError(f"Insufficient data for split. Total samples: {num_samples}")
        
        train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
        
        print(f"Dataset size: {num_samples}")
        print(f"Subsampling each graph to {SUBSAMPLE_NODES} nodes.")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # ==========================================================================
    # 3. Model, Optimizer, and Loss Function
    # ==========================================================================
    try:
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
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # ==========================================================================
    # 4. The Training Loop
    # ==========================================================================
    best_val_loss = float('inf')

    try:
        for epoch in range(EPOCHS):
            # --- Training Phase ---
            model.train()
            total_train_loss = 0
            train_batches = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            
            for batch in train_pbar:
                try:
                    # Clear cache before each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    # Check if batch has required attributes
                    if not hasattr(batch, 'y') or batch.y is None:
                        print(f"Warning: Batch missing target values, skipping...")
                        continue
                    
                    # Enable mixed precision to save memory
                    with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                        predictions = model(batch)
                        loss = loss_fn(predictions.float(), batch.y.float())
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected at epoch {epoch+1}, skipping batch")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    train_batches += 1
                    train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Clear cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"GPU out of memory at epoch {epoch+1}. Clearing cache and skipping batch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Training error: {e}")
                        continue
            
            if train_batches == 0:
                print(f"No valid training batches in epoch {epoch+1}")
                continue
                
            avg_train_loss = total_train_loss / train_batches

            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
                for batch in val_pbar:
                    try:
                        # Clear cache before each validation batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        batch = batch.to(device)
                        
                        if not hasattr(batch, 'y') or batch.y is None:
                            continue
                            
                        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                            predictions = model(batch)
                            loss = loss_fn(predictions.float(), batch.y.float())
                        
                        if not torch.isnan(loss):
                            total_val_loss += loss.item()
                            val_batches += 1
                            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                        
                        # Clear cache after each validation batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"GPU out of memory during validation at epoch {epoch+1}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            print(f"Validation error: {e}")
                            continue

            if val_batches == 0:
                print(f"No valid validation batches in epoch {epoch+1}")
                continue
                
            avg_val_loss = total_val_loss / val_batches
            
            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                try:
                    # Create save directory
                    save_dir = '/content/drive/My Drive/colab_models'
                    if not os.path.exists('/content/drive'):
                        # If not in Colab, save locally
                        save_dir = './models'
                    
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(model.state_dict(), model_path)
                    print(f"✨ New best model saved with validation loss: {best_val_loss:.6f}")
                    
                except Exception as e:
                    print(f"Error saving model: {e}")
                    
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Unexpected error during training: {e}")
    
    print("Training completed!")


if __name__ == '__main__':
    train()