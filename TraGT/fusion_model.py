import torch.nn as nn
import torch

class FusionModel(nn.Module):
    def __init__(self, graph_model, sequence_model):
        super(FusionModel, self).__init__()
        self.graph_model = graph_model
        self.sequence_model = sequence_model
        self.attention = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)
        self.fusion_linear = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)

    def forward(self, graph_data, sequence_data):
        # Move graph_data to device
        #graph_data = graph_data.to(self.graph_model.device)
        graph_embedding = self.graph_model(graph_data)

        # Move sequence_data to device
        #sequence_data = sequence_data.to(self.sequence_model.device)
        sequence_embedding = self.sequence_model(sequence_data)

        # Move sequence_embedding to the same device as graph_embedding
        #sequence_embedding = sequence_embedding.to(graph_embedding.device)

        sequence_embedding = sequence_embedding.unsqueeze(0)
        combined_embedding = torch.cat((graph_embedding, sequence_embedding), dim=1)
        attention_weights = torch.sigmoid(self.attention(combined_embedding))
        weighted_sequence_embedding = attention_weights * sequence_embedding
        # Fusion
        fused_embedding = torch.cat((graph_embedding, weighted_sequence_embedding), dim=1)
        # Apply fusion layer
        output = self.fusion_linear(fused_embedding)
        return output


