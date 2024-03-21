from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch

class Graph_SageMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Graph_SageMLP, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        #print(x.shape)
        return x


