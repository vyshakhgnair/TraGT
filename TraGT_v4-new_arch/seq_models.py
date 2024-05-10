import torch 
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GraphToSequenceEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers):
        super(GraphToSequenceEncoder, self).__init__()
        # Transformer Encoder Layer with given embedding dimension
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=512
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)  # Should retain shape [35, 32]