import torch

import torch.nn.functional as F
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, graph_model, sequence_model):
        super(FusionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.graph_model = graph_model.to(self.device)
        self.sequence_model = sequence_model.to(self.device)
        self.attention = nn.Linear(sequence_model.hidden_size, graph_model.hidden_size)

    def forward(self, graph_data, sequence_data):
        # Move graph_data to device
        graph_data = graph_data.to(self.device)
        sequence_data = sequence_data.to(self.device)
        graph_embedding = self.graph_model(graph_data).to(self.device)

        if torch.isnan(graph_embedding).any():
            raise ValueError("NaN value encountered in graph_embedding. Stopping execution.")

        sequence_embedding, reconstruction = self.sequence_model(graph_embedding)
        sequence_embedding = sequence_embedding.to(self.device)
        reconstruction = reconstruction.to(self.device)
        encoded = sequence_embedding

        if torch.isnan(sequence_embedding).any():
            raise ValueError("NaN value encountered in sequence_embedding. Stopping execution.")

        graph_mean = F.normalize(torch.mean(graph_embedding, dim=0, keepdim=True), p=2, dim=1)
        encoder_mean = F.normalize(torch.mean(sequence_embedding, dim=0, keepdim=True), p=2, dim=1)

        # Apply attention
        attention_weights = torch.softmax(self.attention(sequence_embedding), dim=1)
        attended_sequence_embedding = torch.matmul(attention_weights.transpose(1, 2), sequence_embedding)

        output = graph_mean + encoder_mean + attended_sequence_embedding
        if torch.isnan(output).any():
            raise ValueError("NaN value encountered in output. Stopping execution.")

        return output, encoded, reconstruction
