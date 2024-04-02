import torch
import torch.nn as nn

import torch.nn as nn

def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class classifier_graph(nn.Module):

    def __init__(self, latent_size, device):
        super(classifier_graph,self).__init__()

        self.latent_size = 64
        self.out_channels = 64

        self.mlp = nn.Sequential(
            nn.Linear(self.out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):

        out = self.mlp(x)
        return out
    


class classifier(nn.Module):

    def __init__(self, latent_size, device):
        super(classifier,self).__init__()

        self.latent_size =  1024                           

        self.classifier = nn.Sequential(nn.Linear(self.latent_size,1024),
                                        nn.ReLU(),
                                        nn.Linear(self.latent_size,1),
                                        nn.Sigmoid(),)
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        out = self.classifier(x)
        #print(out,out.shape)
        #print(out,out.shape)
        return out
    
