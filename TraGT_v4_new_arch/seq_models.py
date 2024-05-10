import torch 
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=128):
        super(TransformerDecoder, self).__init__()
        self.transformer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, memory):
        x = self.transformer(x,memory)
        x = self.fc(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=512
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = TransformerDecoder(embedding_dim, hidden_dim, 128)

    def forward(self, x):
        encoded = self.transformer_encoder(x).to(self.device)
        reconstructed = self.decoder(encoded,encoded).to(self.device)
        #print("Reconstructed shape: ", reconstructed,reconstructed.shape)
        #print("Encoded shape: ", encoded,encoded.shape)
        return encoded,reconstructed   # Should retain shape [35, 32]