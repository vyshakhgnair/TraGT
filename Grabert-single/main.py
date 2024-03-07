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
from load_data import load_data
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from graph_model import Graph_SageMLP
import torch.nn as nn
from sklearn.metrics import accuracy_score


def adj_list_to_adj_matrix(adj_list):
    num_nodes = len(adj_list)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1.0
            adj_matrix[neighbor][node] = 1.0
    return adj_matrix


def main(option,d_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data,train_labels,test_data,test_labels=load_data(d_name,device)
    
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

    model = Graph_SageMLP(in_channels=input_dim_train, hidden_channels=64, out_channels=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    logp_sage_train_accuracy=[]

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = criterion(output, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

        # Evaluate accuracy on training set
        model.eval()
        with torch.no_grad():
            predictions = []
            true_labels = []
            for data in train_loader:
                output = model(data.x, data.edge_index)
                predictions.extend(torch.sigmoid(output).cpu().numpy().flatten())
                true_labels.extend(data.y.cpu().numpy())

        epoch_accuracy = accuracy_score(np.round(predictions), true_labels)
        logp_sage_train_accuracy.append(epoch_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {epoch_accuracy:.4f}')




if __name__ == "__main__":
    option=[True,False,False]
    d_name='logp'
    main(option,d_name)