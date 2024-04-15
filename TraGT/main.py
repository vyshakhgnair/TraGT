import os
import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from load_data import load_data
from torch_geometric.data import DataLoader,Data,Dataset
import torch.nn as nn
from sklearn.metrics import accuracy_score
import json
#from joint_model import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import torch.optim as optim
from seq_models import TransformerModel
from graph_models import Graph_Transformer
from fusion_model import FusionModel

def adj_list_to_adj_matrix(adj_list):
    num_nodes = len(adj_list)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1.0
            adj_matrix[neighbor][node] = 1.0
    return adj_matrix



    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, sequence_list):
        self.data_list = data_list
        self.sequence_list = sequence_list


    def __getitem__(self, index):
        data = self.data_list[index]
        sequence = self.sequence_list[index]
        return data, sequence

    def __len__(self):
        return len(self.data_list)





def main(option, d_name):
    batch_size = 1
    learning_rate = 0.001

    # if torch.cuda.is_available():  # Check GPU availability
    #     device = torch.device("cuda:0")  # Set device to GPU
    #     torch.cuda.set_device(device)
    # else:
    #     device = torch.device("cpu")

    device = torch.device("cpu")    
    
    #train_data, train_labels, test_data, test_labels=load_data("logp", device="cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, train_labels, test_data, test_labels=load_data("logp", device="cpu")

    # Move data tensors to GPU
    train_data['sequence'] = [torch.Tensor(seq).to(device) for seq in train_data['sequence']]
    test_data['sequence'] = [torch.Tensor(seq).to(device) for seq in test_data['sequence']]



    input_dim_train = train_data['features'][0].size(-1)
    input_dim_test = test_data['features'][0].size(-1)

    adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
    adj_matrices_test = [adj_list_to_adj_matrix(adj_list) for adj_list in test_data['adj_lists']]
    
     

    data_list_train = [
        Data(   
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
            y=torch.tensor(label, dtype=torch.float)
        )
        for features, adj_matrix, label in zip(train_data['features'], adj_matrices_train, train_labels)
    ]
    data_list_test = [
        Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
            y=torch.tensor(label, dtype=torch.float)
        )
        for features, adj_matrix, label in zip(test_data['features'], adj_matrices_test, test_labels)
    ]


    # Combine both graph and sequence data into a single dataset
    train_data = CustomDataset(data_list_train, train_data['sequence'])
    test_data = CustomDataset(data_list_test, test_data['sequence'])
    
    vocab_size = 100
    d_model = 100
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512

    args=[option,device,train_data,test_data,learning_rate,input_dim_train,input_dim_test]

    graph_model = Graph_Transformer(in_channels=input_dim_train, hidden_channels=64, out_channels=1, heads=4)
    sequence_model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward)
    
    model=FusionModel(graph_model, sequence_model)
    model.to(device)

    num_epochs=50
    losses = 0.0
    c=0

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{output_dir}/train_accuracy_details_{current_time}.txt'
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    with open(file_name, 'a') as file:
        for epoch in range(num_epochs):
            total_correct=0.
            total_samples=0
            losses = 0.0
            for data_batch in train_data:
                graph_data_batch=data_batch[0].to(device)
                seq_data_batch=data_batch[1].to(device)
                target=graph_data_batch.y.to(device)
                sequence_targets = graph_data_batch.y
                #sequence_targets = graph_data_batch.y.view(-1, 1)
            
                
                output=model(graph_data_batch,seq_data_batch)
                file.write(f'output:{output}\n')
                
                binary_predictions = (output > 0.5).float().to(device)
                sequence_targets = sequence_targets.to(binary_predictions.device)
                
                file.write(f'prediction{binary_predictions} target:{target}\n')
                batch_correct = (binary_predictions == sequence_targets).sum().item()
                total_correct += batch_correct
                total_samples += 1
                
                sequence_targets = sequence_targets.view(-1, 1)
                loss = criterion(output, sequence_targets)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                #losses += loss.item()
                
            epoch_accuracy = total_correct / total_samples
            print(f"Epoch {epoch+1}/{100}, Epoch Accuracy: {epoch_accuracy:.4f}")
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {losses:.4f}, Train Accuracy: {epoch_accuracy:.4f}\n')



if __name__ == "__main__":
    #option=[False,True,False]
    #option=[True,True,True]
    #option=[False,True,False]
    option = [False,False,True]
    d_name='logp'
    main(option,d_name)