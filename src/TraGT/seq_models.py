import torch 
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_length=100):
        super(TransformerModel, self).__init__()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward), num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x).to(self.device)
        x = torch.transpose(x, 0, 1).to(self.device)
        x = self.transformer_encoder(x).to(self.device)
        x = torch.mean(x, dim=0).to(self.device)
        x = self.fc(x).to(self.device)
        x = x.mean(dim=0, keepdim=True).to(self.device)
        return x