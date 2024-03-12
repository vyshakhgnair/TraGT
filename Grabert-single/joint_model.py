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
from seq_models import Smiles_BERT, BERT_base
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
        self.Smiles_vocab = args[7]
        self.bert_params = args[8]

        self.graph = args[0][0]
        self.sequence = args[0][1]
        self.use_fusion = args[0][2]

        self.graph_pretrain = graph_models.Graph_SageMLP(in_channels=self.input_dim_train, hidden_channels=64, out_channels=64)

        self.seq_model = Smiles_BERT(len(self.Smiles_vocab), max_len=256, nhead=16, feature_dim=1024, feedforward_dim=1024, nlayers=8, dropout_rate=self.bert_params['dropout'],adj=True)
        self.seq_model.load_state_dict(torch.load('saved_models/pretrained.pt', map_location=self.device), strict=False)
        self.seq_pretrain = BERT_base(self.seq_model)  

        self.AtomEmbedding = nn.Embedding(self.input_dim_train,
                                          self.hidden_size).to(self.device)
        
        self.AtomEmbedding.weight.requires_grad = True

        self.output_layer = models.classifier(self.latent_size, self.device)

        self.label_criterion = nn.BCEWithLogitsLoss()


        self.optimizer  = optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=1e-8,
                                     amsgrad=True)



    def train(self,graph_data,seq_data, epoch):
        seq_data = {key:value.to(self.device) for key, value in seq_data.items()}
        nodes_emb = self.AtomEmbedding(graph_data.x.long())
        #print('Nodes embeddings:',nodes_emb)

        if self.graph:
            nodes_emb = self.graph_pretrain(graph_data.x,graph_data.edge_index)
            graph_emb = nodes_emb
            print("graph",graph_emb)
            print('Graph embeddings',graph_emb.shape)
            print('graphsum',torch.sum(graph_emb,dim=0,keepdim=True)/graph_emb.shape[0])
            #print(nodes_emb)
            #print('Nodes format:',type(nodes_emb),"\n")

        if self.sequence:
            position_num = torch.arange(256).repeat(seq_data["smiles_bert_input"].size(0),1).to(self.device)

            nodes_emb = self.seq_pretrain.forward(seq_data["smiles_bert_input"], position_num, adj_mask=seq_data["smiles_bert_adj_mask"], adj_mat=seq_data["smiles_bert_adjmat"])
            seq_emb = nodes_emb
            print("seq",seq_emb,seq_emb.shape)

        if self.use_fusion:
            molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)
        

        print('Molecule embeddings:',molecule_emb)
        print('Molecule embeddings shape:',molecule_emb.shape)

        pred = self.output_layer(molecule_emb)[0]
        print('Prediction:',pred.shape)
        

        label = torch.FloatTensor([int(graph_data.y.view(-1, 1)[0])]).to(self.device)
        #label=(seq_data["smiles_bert_label"].double()[0]).to(self.device)
        print(graph_data.y)
        print('Label:',label , 'Prediction:',pred)

        self.optimizer.zero_grad()

        loss = self.label_criterion(pred, label)

        loss.backward()
        self.optimizer.step()

        return loss, pred

    def test(self, graph_data,seq_data, epoch):

        nodes_emb = self.AtomEmbedding(graph_data.x.long())

        if self.graph:
            nodes_emb = self.graph_pretrain(graph_data.x,graph_data.edge_index)
            graph_emb = nodes_emb

        if self.sequence:
            position_num = torch.arange(256).repeat(seq_data["smiles_bert_input"].size(0),1).to(self.device)

            nodes_emb = self.seq_pretrain.forward(seq_data["smiles_bert_input"], position_num, adj_mask=seq_data["smiles_bert_adj_mask"], adj_mat=seq_data["smiles_bert_adjmat"])
            seq_emb = nodes_emb

        if self.use_fusion:
            molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)

        pred = self.output_layer(molecule_emb)[0]

        return pred