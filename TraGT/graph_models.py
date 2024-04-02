from torch_geometric.nn import TransformerConv
import torch.nn as nn
import torch

class GTConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, edge_weight=None):
        super(GTConvLayer, self).__init__()
        self.conv = TransformerConv(in_channels, out_channels, heads)
        self.edge_weight = edge_weight

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class Graph_Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout_rate=0.5):
        super(Graph_Transformer, self).__init__()
        self.conv1 = GTConvLayer(in_channels, hidden_channels, heads)
        self.conv2 = GTConvLayer(hidden_channels*heads, out_channels, heads)
        self.linear=nn.Linear(heads*out_channels,out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(x.shape, edge_index.shape)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x=self.linear(x)
        x=x.mean(dim=0, keepdim=True)  
        return x 

