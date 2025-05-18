import torch
from torch_geometric.nn import TransformerConv

import torch.nn as nn
import torch.nn.functional as F

class Graph_Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads):
        super(Graph_Transformer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x=x.to(self.device)
        edge_index=edge_index.to(self.device)
        #print("x shape: ", x)
        #print("edge_index shape: ", edge_index)
        if torch.isnan(x).any():
            raise ValueError("NaN value encountered in x. Stopping execution.")
        if torch.isnan(edge_index).any():
            raise ValueError("NaN value encountered in edge index. Stopping execution.")
        x = self.conv1(x, edge_index).to(self.device)
        if torch.isnan(x).any():
            raise ValueError("NaN value encountered in first conv. Stopping execution.")
        x = torch.relu(x)  
        x = self.conv2(x, edge_index).to(self.device)
        if torch.isnan(x).any():
            raise ValueError("NaN value encountered in second conv. Stopping execution.")
        
        return x

