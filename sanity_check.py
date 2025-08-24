import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Import your custom model class
from src.model.transformer import GeometricTransformer

def run_sanity_check():
    """
    Performs a forward and backward pass to ensure the model is working correctly.
    """
    print("--- 🚀 Running Model Sanity Check ---")

    # --- 1. Define Model Hyperparameters for a small test model ---
    IN_FEATURES = 6         # IMPORTANT: Adjust this to match your .pt files
    OUT_FEATURES = 4        # [p, u, v, nu_t]
    D_MODEL = 128
    NUM_LAYERS = 4
    NUM_HEADS = 4
    D_FF = 512

    # --- 2. Instantiate the Model ---
    model = GeometricTransformer(
        in_features=IN_FEATURES,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        out_features=OUT_FEATURES
    )
    print("✅ Model instantiated successfully.")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # --- 3. Create a Dummy Batch of Data to mimic the DataLoader ---
    # Simulate a batch of two small graphs
    data1 = Data(x=torch.randn(50, IN_FEATURES), edge_index=torch.randint(0, 50, (2, 100)), y=torch.randn(50, OUT_FEATURES))
    data2 = Data(x=torch.randn(75, IN_FEATURES), edge_index=torch.randint(0, 75, (2, 150)), y=torch.randn(75, OUT_FEATURES))

    # PyG's Batch object combines the graphs into one large graph object
    dummy_batch = Batch.from_data_list([data1, data2])
    print("\n✅ Dummy batch created successfully.")
    print(f"   - Total nodes in batch: {dummy_batch.num_nodes}")
    print(f"   - Input feature shape (data.x): {dummy_batch.x.shape}")

    # --- 4. Perform Forward and Backward Pass ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("\nPerforming forward pass...")
    predictions = model(dummy_batch)
    print(f"   - Prediction shape: {predictions.shape}")
    print(f"   - Target shape (data.y): {dummy_batch.y.shape}")

    loss = loss_fn(predictions, dummy_batch.y)
    print(f"   - Calculated loss: {loss.item():.4f}")

    print("Performing backward pass...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("   - Backward pass and optimizer step completed.")

    print("\n--- ✅ Sanity Check Passed ---")


if __name__ == '__main__':
    run_sanity_check()