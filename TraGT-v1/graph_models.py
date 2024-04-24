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


class GCN_2l(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout_rate=0.5):
        super(GCN_2l, self).__init__()
        self.conv1 = GCNConvLayer(in_channels, hidden_channels, heads)
        self.conv2 = GCNConvLayer(hidden_channels*heads, out_channels, heads)
        self.linear=nn.Linear(heads,1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print("before conv1: ", x.shape)
        x = self.conv1(x, edge_index)
        #print("after conv1: ", x.shape)
        x = torch.relu(x)
        x = self.dropout(x)
        #print("before conv2: ", x.shape)
        x = self.conv2(x, edge_index)
        #print("after conv2: ", x.shape)
        x=self.linear(x)

        return x.mean(dim=0, keepdim=True)  # Aggregating to a single output
