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
from load_data import load_data, load_data_long,SMILESDataset
from torch_geometric.data import DataLoader,Data
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn
from sklearn.metrics import accuracy_score
import json
from joint_model import Model
from sklearn.model_selection import train_test_split
import datetime


def adj_list_to_adj_matrix(adj_list):
    num_nodes = len(adj_list)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1.0
            adj_matrix[neighbor][node] = 1.0
    return adj_matrix

def pad_sequence_to_length(sequence, length):
    if len(sequence) < length:
        pad_size = length - len(sequence)
        padding = torch.zeros(pad_size, *sequence.size()[1:], dtype=sequence.dtype, device=sequence.device)
        return torch.cat((sequence, padding), dim=0)
    else:
        return sequence[:length]
    


def main(option,d_name):
    batch_size=1
    # Sequence model parameters
    with open('input/input.json', 'r') as f:
        input_file = json.load(f)
    
    if torch.cuda.is_available() == False:
        torch.set_num_threads(24)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
    train_data_te, train_labels_te,test_data_te,test_labels_te=load_data("logp", device="cuda" if torch.cuda.is_available() else "cpu")


    
    
    train_data_te['sequence'] = [torch.Tensor(seq) for seq in train_data_te['sequence']]
    test_data_te['sequence'] = [torch.Tensor(seq) for seq in test_data_te['sequence']]
    padded_train_sequence = [pad_sequence_to_length(tensor, length=100) for tensor in train_data_te['sequence']]
    padded_test_sequence = [pad_sequence_to_length(tensor, length=100) for tensor in test_data_te['sequence']]

    train_data_te=padded_train_sequence
    test_data_te=padded_test_sequence


    # Create datasets and data loaders for training and validation
    train_dataset = SMILESDataset(train_data_te, train_labels_te)
    test_dataset = SMILESDataset(test_data_te, test_labels_te)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


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
    
    
    train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_list_test, batch_size=batch_size, shuffle=True)

    args=[option,device,train_loader,test_loader,learning_rate,input_dim_train,input_dim_test,padded_train_sequence,padded_test_sequence]


    model=Model(args)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs=50
    correct_predictions = 0
    total_predictions = 0
    losses = 0.0
    c=0
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{output_dir}/train_accuracy_details_{current_time}.txt'

    with open(file_name, 'a') as file:
        for epoch in range(num_epochs):
            losses = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for graph_data, seq_data in zip(train_loader, train_dataloader):
                #print('Graph data:',graph_data.x.shape,graph_data.edge_index.shape)
                loss, output = model.train(graph_data, seq_data, epoch)
                losses += loss.item()
                
                # Convert model output to predicted labels
                predicted_labels = abs(torch.round(output).detach())
                target = graph_data.y.double()
                #print('output:',output)
                file.write(f'output:{output}\n')
                #print('Prediction:',predicted_labels,'Target',target)
                file.write(f'Prediction:{predicted_labels}, Target:{target}\n')
                
                # Compare with actual labels
                correct_predictions += (predicted_labels == target).sum().item()
                total_predictions += graph_data.y.size(0)
            
            train_accuracy = correct_predictions / total_predictions
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {losses:.4f}, Train Accuracy: {train_accuracy:.4f}')
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {losses:.4f}, Train Accuracy: {train_accuracy:.4f}\n')
        
    # Evaluation on test set
    test_correct = 0
    test_total = 0
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{output_dir}/test_accuracy_details_{current_time}.txt'

    with open(file_name, 'a') as file:
        for graph_data, seq_data in zip(test_loader, test_dataloader):
            with torch.no_grad():
                output = model.test(graph_data, seq_data, epoch)
                predicted_labels = torch.round(output).detach()
                target = graph_data.y.double()
                test_correct += (predicted_labels == target).sum().item()
                test_total += graph_data.y.size(0)
            
            test_accuracy = test_correct / test_total
            print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_accuracy:.4f}')
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_accuracy:.4f}\n')    







if __name__ == "__main__":
    option=[True,True,True]
    #option=[True,False,False]
    #option=[False,True,False]
    d_name='logp'
    main(option,d_name)