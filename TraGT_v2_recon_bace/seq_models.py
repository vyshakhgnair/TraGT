import torch 
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformerDecoder, self).__init__()
        self.transformer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, memory):
        x = self.transformer(x,memory)
        x = self.fc(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_length=100):
        super(TransformerModel, self).__init__()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward), num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)
        self.decoder = TransformerDecoder(d_model, d_model, vocab_size)
        
        
    def forward(self, x):
        x = self.embedding(x).to(self.device)
        x = torch.transpose(x, 0, 1).to(self.device)
        encoded = self.transformer_encoder(x).to(self.device)
        x = torch.mean(encoded, dim=0).to(self.device)
        x = self.fc(x).to(self.device)
        x = x.mean(dim=0, keepdim=True).to(self.device)
        reconstructed = self.decoder(encoded,encoded).to(self.device)
        reconstructed = self.fc(reconstructed).to(self.device)
        
        return x, reconstructed