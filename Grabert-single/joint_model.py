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
        
        self.seq_model.load_state_dict(torch.load('saved_models/pretrained_model.pt', map_location=self.device), strict=False)
        
        self.seq_pretrain = BERT_base(self.seq_model)  

        self.AtomEmbedding = nn.Embedding(self.input_dim_train,
                                          self.hidden_size).to(self.device)
        
        self.AtomEmbedding.weight.requires_grad = True

        self.output_layer = models.classifier(self.latent_size, self.device)

        self.custom_output_layer = models.classifier_graph(self.latent_size, self.device)


        self.label_criterion = nn.BCEWithLogitsLoss()


        self.optimizer  = optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=1e-8,
                                     amsgrad=True)



    def train(self,graph_data,seq_data, epoch):
        seq_data = {key:value.to(self.device) for key, value in seq_data.items()}
        nodes_emb = self.AtomEmbedding(graph_data.x.long())
        

        if self.graph:
            nodes_emb = self.graph_pretrain(graph_data.x,graph_data.edge_index)
            graph_emb = nodes_emb


        if self.sequence:
            position_num = torch.arange(256).repeat(seq_data["smiles_bert_input"].size(0),1).to(self.device)

            nodes_emb = self.seq_pretrain.forward(seq_data["smiles_bert_input"], position_num, adj_mask=seq_data["smiles_bert_adj_mask"], adj_mat=seq_data["smiles_bert_adjmat"])
            seq_emb = nodes_emb
            
            

        if self.use_fusion:
            projection = nn.Linear(1024, 64)
            seq_emb = projection(seq_emb.squeeze(0)) 
            seq_emb = torch.mean(seq_emb, dim=0)
            graph_emb = torch.mean(graph_emb, dim=0)
            molecule_emb = torch.cat((seq_emb.unsqueeze(0),graph_emb.unsqueeze(0)), dim=0)


            #molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)
        


        if self.use_fusion==False and self.sequence==False:
            #print('Custom output layer')
            pred = self.custom_output_layer(molecule_emb)[0]
        else:
            #print('Output layer')
            pred = self.output_layer(molecule_emb)[0]
            pred = torch.mean(pred,dim=0,keepdim=True)[0]

        

        label = torch.FloatTensor([int(graph_data.y.view(-1, 1)[0])]).to(self.device)
        #pred = abs(torch.round(pred).detach())
        #print('Label:',label , 'Prediction:',pred)

        #print(pred,pred.shape)
        self.optimizer.zero_grad()
        

        loss = self.label_criterion(pred, label)

        loss.backward()
        self.optimizer.step()

        return loss, pred

    def test(self, graph_data,seq_data, epoch):
        seq_data = {key:value.to(self.device) for key, value in seq_data.items()}
        nodes_emb = self.AtomEmbedding(graph_data.x.long())
        

        if self.graph:
            nodes_emb = self.graph_pretrain(graph_data.x,graph_data.edge_index)
            graph_emb = nodes_emb


        if self.sequence:
            position_num = torch.arange(256).repeat(seq_data["smiles_bert_input"].size(0),1).to(self.device)

            nodes_emb = self.seq_pretrain.forward(seq_data["smiles_bert_input"], position_num, adj_mask=seq_data["smiles_bert_adj_mask"], adj_mat=seq_data["smiles_bert_adjmat"])
            seq_emb = nodes_emb
            
            

        if self.use_fusion:
            projection = nn.Linear(1024, 64)
            seq_emb = projection(seq_emb.squeeze(0)) 
            seq_emb = torch.mean(seq_emb, dim=0)
            graph_emb = torch.mean(graph_emb, dim=0)
            molecule_emb = torch.cat((seq_emb.unsqueeze(0),graph_emb.unsqueeze(0)), dim=0)


            #molecule_emb = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)
        else:
            molecule_emb = torch.mean(nodes_emb, dim=0, keepdim=True)
        


        if self.use_fusion==False and self.sequence==False:
            #print('Custom output layer')
            pred = self.custom_output_layer(molecule_emb)[0]
        else:
            #print('Output layer')
            pred = self.output_layer(molecule_emb)[0]
            pred = torch.mean(pred,dim=0,keepdim=True)

        return pred