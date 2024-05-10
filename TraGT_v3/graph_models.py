import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GCNConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, edge_weight=None):
        super(GCNConvLayer, self).__init__()
        self.conv = TransformerConv(in_channels, out_channels, heads)
        self.edge_weight = edge_weight

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class Graph_Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout_rate=0.5):
        super(Graph_Transformer, self).__init__()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = GCNConvLayer(in_channels, hidden_channels, heads)
        self.conv2 = GCNConvLayer(hidden_channels * heads, out_channels, heads)
        self.linear = nn.Linear(heads, 100)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x.to(self.device) , data.edge_index.to(self.device)
        x = self.conv1(x, edge_index).to(self.device)
        x = self.conv2(x, edge_index).to(self.device)
        x = self.linear(x).to(self.device)
        x = torch.relu(x).to(self.device)
        x = self.dropout(x).to(self.device)
        x = x.mean(dim=0, keepdim=True).to(self.device)
        return x
