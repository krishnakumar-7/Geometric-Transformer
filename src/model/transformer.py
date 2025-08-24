import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# Part 1: Core Building Blocks (from Day 2)
# ==============================================================================

class FourierFeatureEmbedding(nn.Module):
    """
    Embeds input features and enhances coordinate information with Fourier features.
    This helps the model learn high-frequency variations in the data, which is
    crucial for fluid dynamics problems.
    """
    def __init__(self, in_features, d_model, fourier_features=64, fourier_scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.d_fourier = fourier_features
        
        # Input features include coordinates, so we separate them for Fourier encoding.
        self.num_coord_feats = 2
        self.num_other_feats = in_features - self.num_coord_feats
        
        # Random Gaussian matrix for Fourier features
        self.register_buffer('B', torch.randn(self.num_coord_feats, self.d_fourier) * fourier_scale)
        
        # Projection MLP
        mlp_in_dim = (self.d_fourier * 2) + self.num_other_feats
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, num_points, in_features)
        Returns:
            Tensor: Embedded tensor of shape (batch_size, num_points, d_model)
        """
        # Separate coordinates from other features
        coords = x[..., :self.num_coord_feats]
        other_feats = x[..., self.num_coord_feats:]
        
        # Project coordinates using the Gaussian matrix and apply sin/cos
        proj = coords @ self.B
        fourier_feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
        # Combine and project through the MLP
        combined_feats = torch.cat([other_feats, fourier_feats], dim=-1)
        embedding = self.mlp(combined_feats)
        return embedding

class MultiHeadAttention(nn.Module):
    """
    Implements the core self-attention mechanism, allowing each point to
    attend to all other points in the domain to capture global context.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of each head's key/query
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, x):
        batch_size, num_points, _ = x.shape
        
        # 1. Project to Q, K, V and reshape for multi-head processing
        Q = self.W_q(x).view(batch_size, num_points, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_points, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_points, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        
        # 3. Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_points, self.d_model)
        output = self.W_o(attn_output)
        return output

class FeedForward(nn.Module):
    """
    A simple two-layer MLP applied independently to each point's feature vector.
    This is the Position-wise Feed-Forward Network from your Day 3 morning plan.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))

class TransformerEncoderBlock(nn.Module):
    """
    Combines Multi-Head Attention and the Feed-Forward Network with
    residual connections and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Sub-layer 1: Multi-Head Attention with residual connection
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)
        
        # Sub-layer 2: Feed-Forward Network with residual connection
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x

# ==============================================================================
# Part 2: Full Network Assembly (from Day 3)
# ==============================================================================

class GeometricTransformer(nn.Module):
    """
    The complete Geometric Transformer model, including the encoder and a final
    MLP decoder head to predict the flow fields.
    """
    def __init__(self, in_features, d_model, num_layers, num_heads, d_ff, out_features):
        super().__init__()
        
        # --- 1. Input Embedding ---
        self.embedding = FourierFeatureEmbedding(in_features, d_model)
        
        # --- 2. Encoder ---
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        # --- 3. Decoder ---
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_features)
        )

    def forward(self, data):
        """
        Processes a batch of graphs by iterating through them, applying the
        Transformer to each one individually to avoid the N^2 memory explosion.
        
        Args:
            data (torch_geometric.data.Batch): A batch of graph data.
        Returns:
            Tensor: Predicted flow field of shape (num_total_nodes_in_batch, out_features)
        """
        # --- THIS IS THE CRITICAL FIX ---
        # Unbatch the giant graph object into a list of individual graphs.
        graph_list = data.to_data_list()
        output_list = []

        # Loop through each graph in the batch.
        for graph in graph_list:
            # Get features for the current graph.
            # The unsqueeze adds the singleton batch dimension required by the Transformer.
            # The shape of x is now [1, N_nodes_in_this_graph, F_in], which is manageable.
            x = graph.x.unsqueeze(0) 

            # 1. Embed input features
            embedding = self.embedding(x)
            
            # 2. Pass through Transformer encoder blocks
            for layer in self.encoder_layers:
                embedding = layer(embedding)
            
            # 3. Project to output flow variables using the decoder head
            output = self.decoder_head(embedding)
            
            # Remove the singleton batch dimension and add to our list of outputs
            output_list.append(output.squeeze(0))

        # Concatenate the outputs from all graphs back into a single tensor.
        # The final shape is [N_total, F_out], which matches the target `data.y`.
        return torch.cat(output_list, dim=0)