import torch.nn.functional as F
import torch 
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, graph_model, sequence_model):
        super(FusionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.graph_model = graph_model
        self.sequence_model = sequence_model
        self.attention = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)
        self.fusion_linear = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)

    def forward(self, graph_data, sequence_data):
        # Move graph_data to device
        graph_data = graph_data.to(self.device)
        sequence_data=sequence_data.to(self.device)
        
        graph_embedding = self.graph_model(graph_data).to(self.device)
        sequence_embedding = self.sequence_model(sequence_data).to(self.device)


        sequence_embedding = sequence_embedding.unsqueeze(0).to(self.device)
        combined_embedding = torch.cat((graph_embedding, sequence_embedding), dim=1).to(self.device)
        attention_weights = torch.sigmoid(self.attention(combined_embedding)).to(self.device)
        weighted_sequence_embedding = (attention_weights * sequence_embedding).to(self.device)
        # Fusion
        fused_embedding = torch.cat((graph_embedding, weighted_sequence_embedding), dim=1).to(self.device)
        # Apply fusion layer
        output = self.fusion_linear(fused_embedding).to(self.device)
        return output