import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier(nn.Module):
    def __init__(self, latent_size, device):
        super(classifier, self).__init__()

        self.latent_size = latent_size

        self.classifier = nn.Sequential(nn.Linear(self.latent_size, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 1))

        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        out = self.classifier(x)
        return out


# class FusionLayer(nn.Module):
#     def __init__(self, fusion_size=128, device=None):
#         super(FusionLayer, self).__init__()
#         self.device = device
#         self.linear = nn.Linear(fusion_size, 16)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(16, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.to(device)  # Move the FusionLayer to the specified device

#     def forward(self, graph_emb, seq_emb):
#         graph_emb = graph_emb[0]
#         #print(graph_emb.shape, seq_emb.shape)
#         x = torch.cat((graph_emb, seq_emb), dim=0)  # Concatenate the embeddings
#         x = self.linear(x)  # Pass through a linear layer
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = self.sigmoid(x)  # Apply sigmoid activation function
#         return x
    
class FusionLayer(nn.Module):
    def __init__(self, fusion_size=128, device=None):
        super(FusionLayer, self).__init__()
        self.device = device
        self.linear = nn.Linear(fusion_size, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(device)  # Move the FusionLayer to the specified device

    def forward(self, graph_emb, seq_emb):
        graph_emb = graph_emb[0]
        #print(graph_emb.shape, seq_emb.shape)
        x = torch.cat((graph_emb, seq_emb), dim=0)  # Concatenate the embeddings
        x = self.linear(x)  # Pass through a linear layer
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)  # Apply sigmoid activation function
        return x

