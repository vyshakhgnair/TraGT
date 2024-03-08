import copy
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data

import pickle
import random
import numpy as np


import time
from tqdm import tqdm

import graph_models
import seq_models
import models


class Model(nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()

        self.device = args[1]
        self.train_data = args[2]
        self.test_data = args[3]
        self.learning_rate = args[4]
        self.input_dim_train = args[5]
        self.input_dim_test = args[6]
        self.latent_size = 1
        self.hidden_size = 64

        self.graph = args[0][0]
        self.sequence = args[0][1]
        self.use_fusion = args[0][2]

        self.graph_pretrain = graph_models.Graph_SageMLP(in_channels=self.input_dim_train, hidden_channels=64, out_channels=64)

        #self.VAE = seq_models.VAE(args)

        self.AtomEmbedding = nn.Embedding(self.input_dim_train,
                                          self.hidden_size).to(self.device)
        self.AtomEmbedding.weight.requires_grad = True

        self.output_layer = models.classifier(self.latent_size, self.device)

        self.label_criterion = nn.BCEWithLogitsLoss()


        self.optimizer  = optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=1e-8,
                                     amsgrad=True)



    def train(self,data, epoch):

        nodes_emb = self.AtomEmbedding(data.x.long())

        if self.graph:
            nodes_emb = self.graph_pretrain(data.x,data.edge_index)
            graph_emb = nodes_emb
            #print("graph",graph_emb)
            #print('Graph embeddings',graph_emb.shape)
            #print(nodes_emb)
            #print('Nodes format:',type(nodes_emb),"\n")

        if self.sequence:
            #recons_loss, nodes_emb = self.VAE(nodes_emb, epoch)
            seq_emb = nodes_emb
            #print("seq",seq_emb.shape)

        if self.use_fusion:
            molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)
        

        #print('Molecule embeddings:',molecule_emb.shape)

        pred = self.output_layer(molecule_emb)[0]
        

        label = torch.FloatTensor([int(data.y.view(-1, 1)[0])]).to(self.device)
        #print('Label:',label , 'Prediction:',pred)

        self.optimizer.zero_grad()

        loss = self.label_criterion(pred, label)

        loss.backward()
        self.optimizer.step()

        return loss, pred

    def test(self, data, epoch):

        nodes_emb = self.AtomEmbedding(data.x.long())

        if self.graph:
            nodes_emb = self.graph_pretrain(data.x,data.edge_index)
            graph_emb = nodes_emb

        if self.sequence:
            #nodes_emb = self.VAE.test_vae(nodes_emb)
            seq_emb = nodes_emb

        if self.use_fusion:
            molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)

        pred = self.output_layer(molecule_emb)

        return pred