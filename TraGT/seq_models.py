import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward,out_channels, max_length=100):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward), num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, out_channels)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 0, 1)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x