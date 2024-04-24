import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from load_data import load_data,load_data_long
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_list, sequence_list):
        self.data_list = data_list
        self.sequence_list = sequence_list

    def __getitem__(self, index):
        data = self.data_list[index]
        sequence = self.sequence_list[index]
        return data, sequence

    def __len__(self):
        return len(self.data_list)


train_data, train_labels, test_data, test_labels = load_data_long("bbbp", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logp_input_dim_train = train_data['features'][0].size(-1)
logp_input_dim_test = test_data['features'][0].size(-1)

def adj_list_to_adj_matrix(adj_list):
    num_nodes = max(adj_list.keys()) + 1
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1.0
            adj_matrix[neighbor][node] = 1.0
    return adj_matrix

logp_adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
logp_adj_matrices_test = [adj_list_to_adj_matrix(adj_list) for adj_list in test_data['adj_lists']]

logp_data_sequence_train = torch.stack(train_data['sequence'])
logp_data_sequence_test = torch.stack(test_data['sequence'])

logp_data_list_train = [Data(x=torch.tensor(features, dtype=torch.float),
                              edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                              y=torch.tensor(label, dtype=torch.float))
                         for features, adj_matrix, label in zip(train_data['features'], logp_adj_matrices_train, train_labels)]
logp_data_list_test = [Data(x=torch.tensor(features, dtype=torch.float),
                             edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                             y=torch.tensor(label, dtype=torch.float))
                        for features, adj_matrix, label in zip(test_data['features'], logp_adj_matrices_test, test_labels)]

train_dataset = CustomDataset(logp_data_list_train, train_data['sequence'])
test_dataset = CustomDataset(logp_data_list_test, test_data['sequence'])



class GCNConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, edge_weight=None):
        super(GCNConvLayer, self).__init__()
        self.conv = TransformerConv(in_channels, out_channels, heads)
        self.edge_weight = edge_weight

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class GCN_2l(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout_rate=0.5):
        super(GCN_2l, self).__init__()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = GCNConvLayer(in_channels, hidden_channels, heads)
        self.conv2 = GCNConvLayer(hidden_channels * heads, out_channels, heads)
        self.linear = nn.Linear(heads, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x.to(self.device) , data.edge_index.to(self.device)
        x = self.conv1(x, edge_index).to(self.device)
        x = torch.relu(x).to(self.device)
        x = self.dropout(x).to(self.device)
        x = self.conv2(x, edge_index).to(self.device)
        x = self.linear(x).to(self.device)
        x = x.mean(dim=0, keepdim=True).to(self.device)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_length=100):
        super(TransformerModel, self).__init__()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward), num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x).to(self.device)
        x = torch.transpose(x, 0, 1).to(self.device)
        x = self.transformer_encoder(x).to(self.device)
        x = torch.mean(x, dim=0).to(self.device)
        x = self.fc(x).to(self.device)
        x = x.mean(dim=0, keepdim=True).to(self.device)
        return x

vocab_size = 100
d_model = 100
nhead = 4
num_encoder_layers = 3
dim_feedforward = 512
max_length = 100
batch_size = 1
num_epochs = 100

# Define the reconstruction loss function
reconstruction_criterion = nn.MSELoss()


# class FusionModel(nn.Module):
#     def __init__(self, graph_model, sequence_model):
#         super(FusionModel, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.graph_model = graph_model
#         self.sequence_model = sequence_model
#         self.attention = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)
#         self.fusion_linear = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)

#     def forward(self, graph_data, sequence_data):
#         # Move graph_data to device
#         graph_data = graph_data.to(self.device)
#         sequence_data=sequence_data.to(self.device)
#         graph_embedding = self.graph_model(graph_data).to(self.device)

#         # Move sequence_data to device
#         #sequence_data = sequence_data.to(self.sequence_model.device)
#         sequence_embedding = self.sequence_model(sequence_data).to(self.device)

#         # Move sequence_embedding to the same device as graph_embedding
#         #sequence_embedding = sequence_embedding.to(graph_embedding.device)

#         sequence_embedding = sequence_embedding.unsqueeze(0).to(self.device)
#         combined_embedding = torch.cat((graph_embedding, sequence_embedding), dim=1).to(self.device)
#         attention_weights = torch.sigmoid(self.attention(combined_embedding)).to(self.device)
#         weighted_sequence_embedding = (attention_weights * sequence_embedding).to(self.device)
#         # Fusion
#         fused_embedding = torch.cat((graph_embedding, weighted_sequence_embedding), dim=1).to(self.device)
#         # Apply fusion layer
#         output = self.fusion_linear(fused_embedding).to(self.device)
#         return output

import torch.nn.functional as F
class FusionModel(nn.Module):
    def __init__(self, graph_model, sequence_model):
        super(FusionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.graph_model = graph_model
        self.sequence_model = sequence_model
        self.attention = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)
        self.fusion_linear = nn.Linear(graph_model.linear.out_features + sequence_model.fc.out_features, 1)

    def forward(self, graph_data, sequence_data):
        # Move graph_data to device
        graph_data = graph_data.to(self.device)
        sequence_data=sequence_data.to(self.device)
        graph_embedding = self.graph_model(graph_data).to(self.device)
        

        # Move sequence_data to device
        #sequence_data = sequence_data.to(self.sequence_model.device)
        sequence_embedding = self.sequence_model(sequence_data).to(self.device)

        # Move sequence_embedding to the same device as graph_embedding
        #sequence_embedding = sequence_embedding.to(graph_embedding.device)

        sequence_embedding = sequence_embedding.unsqueeze(0).to(self.device)
        combined_embedding = torch.cat((graph_embedding, sequence_embedding), dim=1).to(self.device)
        attention_weights = torch.sigmoid(self.attention(combined_embedding)).to(self.device)
        weighted_sequence_embedding = (attention_weights * sequence_embedding).to(self.device)
        # Fusion
        fused_embedding = torch.cat((graph_embedding, weighted_sequence_embedding), dim=1).to(self.device)
        # Apply fusion layer
        output = self.fusion_linear(fused_embedding).to(self.device)
        return output
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

graph_model = GCN_2l(in_channels=logp_input_dim_train, hidden_channels=64, out_channels=1, heads=4).to(device)
sequence_model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward).to(device)
fusion_model = FusionModel(graph_model, sequence_model).to(device)

criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.0015)


for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    true_labels_train = []
    pred_probs_train = []
    
    for data_batch in train_dataset:
        graph_data_batch = data_batch[0]
        sequence_inputs = data_batch[1]
        sequence_targets = graph_data_batch.y
        
        output = fusion_model(graph_data_batch, sequence_inputs)
        
        # Compute reconstruction from the graph model
        reconstructed_seq = sequence_model(sequence_inputs.to(device))
        
        binary_predictions = (output >= 0.5).float()
        batch_correct = (binary_predictions == sequence_targets).sum().item()
        total_correct += batch_correct
        total_samples += 1
        
        output = output.to(device)
        sequence_targets = sequence_targets.view(-1, 1).to(device)
        
        true_labels_train.append(sequence_targets.cpu().numpy())
        pred_probs_train.append(output.detach().cpu().numpy())
        
        # Calculate the reconstruction loss
        reconstructed_seq = reconstructed_seq.squeeze(0).to(device)
        sequence_inputs = sequence_inputs.to(device)
        reconstruction_loss = reconstruction_criterion(reconstructed_seq, sequence_inputs)

        # Calculate the binary cross-entropy loss
        loss = criterion(output, sequence_targets) + reconstruction_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_accuracy = total_correct / total_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Epoch Training Accuracy: {epoch_accuracy:.4f} ",end='')
    true_labels_train = np.concatenate(true_labels_train)
    pred_probs_train = np.concatenate(pred_probs_train)
    auc_roc_train = roc_auc_score(true_labels_train, pred_probs_train)
    f1_train = f1_score(true_labels_train, (pred_probs_train >= 0.5).astype(int))
    print(f"Train AUC-ROC: {auc_roc_train:.4f}, Train F1 Score: {f1_train:.4f}")
    
    total_correct = 0
    total_samples = 0
    true_labels_test = []
    pred_probs_test = []
    for data_batch in test_dataset:
        graph_data_batch = data_batch[0]
        sequence_inputs = data_batch[1]
        sequence_targets = graph_data_batch.y
        
        output = fusion_model(graph_data_batch, sequence_inputs)
        binary_predictions = (output >= 0.5).float()
        batch_correct = (binary_predictions == sequence_targets).sum().item()
        total_correct += batch_correct
        total_samples += 1
        
        true_labels_test.append(sequence_targets.cpu().numpy().reshape(-1))
        pred_probs_test.append(output.detach().cpu().numpy())
    epoch_test_accuracy = total_correct/total_samples
    print(f"Epoch Testing Accuracy : {epoch_test_accuracy:.4f}")
    true_labels_test = np.concatenate(true_labels_test)
    pred_probs_test = np.concatenate(pred_probs_test)
    auc_roc_test = roc_auc_score(true_labels_test, pred_probs_test)
    f1_test = f1_score(true_labels_test, (pred_probs_test >= 0.5).astype(int))
    print(f"Test AUC-ROC: {auc_roc_test:.4f}, Test F1 Score: {f1_test:.4f}")



