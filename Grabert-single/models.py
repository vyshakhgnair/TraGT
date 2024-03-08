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


class classifier(nn.Module):

    def __init__(self, latent_size, device):
        super(classifier,self).__init__()

        self.latent_size = latent_size

        self.classifier = nn.Sequential(nn.Linear(self.latent_size, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 1),
                                        )
        
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):

        out = self.classifier(x)
        return out