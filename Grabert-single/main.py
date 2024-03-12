import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from rdkit.Chem import AllChem
from rdkit import Chem
from load_data import load_data, Vocab, FinetuningDataset
from torch_geometric.data import DataLoader,Data
from torch.utils.data import SubsetRandomSampler
from graph_models import Graph_SageMLP
import torch.nn as nn
from sklearn.metrics import accuracy_score
import json
from joint_model import Model


def adj_list_to_adj_matrix(adj_list):
    num_nodes = len(adj_list)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1.0
            adj_matrix[neighbor][node] = 1.0
    return adj_matrix


def main(option,d_name):

    # Sequence model parameters
    with open('input/input.json', 'r') as f:
        input_file = json.load(f)
    
    if torch.cuda.is_available() == False:
        torch.set_num_threads(24)

    bert_params = {'batch_size':1, 'dropout':0, 'learning_rate':0.00001, 'optimizer':'Adam', 'model':'Transformer'}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Smiles_vocab = Vocab()
    

    dataset = FinetuningDataset('input/dataset.csv', Smiles_vocab, seq_len=256)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split1= int(np.floor(input_file['split_ratio'] * len(dataset)))
    train_idx, test_idx = indices[:split1], indices[split1:]
	
    print(len(train_idx), len(test_idx))
	
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_dataloader = DataLoader(dataset, batch_size=bert_params['batch_size'], sampler=train_sampler, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(dataset, batch_size=bert_params['batch_size'], sampler=test_sampler, num_workers=2)





    # Graph model parameters
    train_data,train_labels,test_data,test_labels=load_data(d_name,device)
    
    learning_rate=0.001
    input_dim_train = train_data['features'][0].size(-1)
    input_dim_test = test_data['features'][0].size(-1)

    adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
    adj_matrices_test = [adj_list_to_adj_matrix(adj_list) for adj_list in test_data['adj_lists']]

    data_list_train = [Data(x=torch.tensor(features, dtype=torch.float),
                  edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                  y=torch.tensor(label, dtype=torch.float))
             for features, adj_matrix, label in zip(train_data['features'], adj_matrices_train, train_labels)]
    data_list_test= [Data(x=torch.tensor(features, dtype=torch.float),
                  edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                  y=torch.tensor(label, dtype=torch.float))
             for features, adj_matrix, label in zip(test_data['features'], adj_matrices_test, test_labels)]
    
    
    train_loader = DataLoader(data_list_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(data_list_test, batch_size=1, shuffle=True)

    args=[option,device,train_loader,test_loader,learning_rate,input_dim_train,input_dim_test,Smiles_vocab,bert_params]

    #model = Graph_SageMLP(in_channels=input_dim_train, hidden_channels=64, out_channels=64)
    model=Model(args)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs=1
    correct_predictions = 0
    total_predictions = 0
    losses = 0.0
    c=0

    for epoch in range(num_epochs):
        losses=0.0
        for graph_data,seq_data in zip(train_loader,train_dataloader):

            print('\n data:',c)
            c+=1
            loss, output = model.train(graph_data,seq_data, epoch)  
            print('output:',output)
            losses += loss.item()

            # Convert model output to predicted labels
            predicted_labels = torch.round(output).detach()
            target = graph_data.y.double()
            print('Prediction:',predicted_labels,'Target',target)
            # Compare with actual labels
            correct_predictions += (predicted_labels == target).sum().item()
            total_predictions += graph_data.y.size(0)
            print('\n correct_predictions:',correct_predictions)
            print('total_predictions:',total_predictions)   

        accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses:.4f}, Accuracy: {accuracy:.4f}')
        test_correct=0
        test_total=0
        print('Evaluating on test set... \n')
        for graph_data,seq_data in zip(test_loader,test_dataloader):
            with torch.no_grad():
                predictions=[]
                true_labels=[]
                output = model.test(graph_data,seq_data, epoch)
                predicted_labels = torch.round(output).detach()  # Assuming binary classification
                target = seq_data["smiles_bert_label"].double()
                test_correct+=(predicted_labels == target).sum().item()
                test_total+=1
        print(test_correct,test_total)
        epoch_accuracy=test_correct/test_total
        #epoch_accuracy = accuracy_score(np.round(predictions), true_labels)
        print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {epoch_accuracy:.4f}')




    

    # # Training loop
    # num_epochs = 1
    # for epoch in range(num_epochs):
    #     model.train()
    #     for data in train_loader:
    #         #print(data)
    #         #print(data.x)
    #         #print(data.edge_index)
    #         optimizer.zero_grad()
    #         output = model(data.x, data.edge_index)
    #         #print(torch.LongTensor([int(data.y.view(-1, 1)[0])]))
    #         print(output)
    #         loss = criterion(output, data.y.view(-1, 1))
    #         loss.backward()
    #         optimizer.step()

    #     # Evaluate accuracy on training set
    #     logp_sage_train_accuracy = []  

    #     model.eval()
    #     with torch.no_grad():
    #         predictions = []
    #         true_labels = []
    #         for data in train_loader:
    #             output = model(data.x, data.edge_index)
    #             predictions.extend(torch.sigmoid(output).cpu().numpy().flatten())
    #             true_labels.extend(data.y.cpu().numpy())

    #         epoch_accuracy = accuracy_score(np.round(predictions), true_labels)
    #         logp_sage_train_accuracy.append(epoch_accuracy)
    #         print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {epoch_accuracy:.4f}')




if __name__ == "__main__":
    option=[True,True,True]
    d_name='logp'
    main(option,d_name)