import torch.nn.functional as F
import torch 
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, graph_model, sequence_model,options):
        super(FusionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.graph_model = graph_model
        self.sequence_model = sequence_model
        self.attention = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)
        self.fusion_linear = nn.Linear(200, 1)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, graph_data, sequence_data,options):
        sequence_data = sequence_data.unsqueeze(0)
        node_embedding = sequence_data
        #print(node_embedding.shape)
        graph_data = graph_data.to(self.device)
        sequence_data=sequence_data.to(self.device)
        node_embedding = node_embedding.to(self.device)
        if options[0]:
            node_embedding = self.graph_model(graph_data).to(self.device)
            
            graph_embedding = node_embedding.to(self.device)
            if torch.isnan(node_embedding).any():
                print("graph None")
            #print(graph_embedding.shape)
        if options[1]:
            node_embedding = node_embedding.long()
            sequence_embedding = self.sequence_model(node_embedding).to(self.device)
            #print(sequence_embedding.shape)
            node_embedding = sequence_embedding.to(self.device)
            if torch.isnan(node_embedding).any():
                print("seq None")
            
        if options[2]:
            sequence_embedding = sequence_embedding.squeeze(0)
            print(graph_embedding.shape, sequence_embedding.shape)
            #combined_embedding = F.normalize(torch.mean(graph_embedding, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(sequence_embedding, dim=0, keepdim=True), p=2, dim=1)
            combined_embedding = torch.cat((graph_embedding, sequence_embedding), dim=1)
            #output = F.leaky_relu(combined_embedding).to(self.device)
            output = self.fusion_linear(combined_embedding).to(self.device)
            #output = output.squeeze(0)
            if torch.isnan(output).any():
                print("fusion None")
                exit()
        print(output)
        return output