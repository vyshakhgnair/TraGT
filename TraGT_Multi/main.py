from load_data import load_data, load_data_long,CustomDataset,adj_list_to_adj_matrix
import torch
from torch_geometric.data import Data
from seq_models import TransformerModel
from graph_models import Graph_Transformer
from fusion_model import FusionModel
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import os


def main(data_name,options):
    
    vocab_size = 100
    d_model = 100
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512
    max_length = 100
    batch_size = 1
    num_epochs = 100


    train_data, train_labels, test_data, test_labels = load_data_long(data_name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    unique_train_labels = torch.unique(train_labels)
    unique_test_labels = torch.unique(test_labels)
    print("Number of unique labels in train: ", len(unique_train_labels))
    print("Number of unique labels in test: ", len(unique_test_labels))
    return
    input_dim_train = train_data['features'][0].size(-1)
    input_dim_test = test_data['features'][0].size(-1)
    
    
    adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
    adj_matrices_test = [adj_list_to_adj_matrix(adj_list) for adj_list in test_data['adj_lists']]

    data_sequence_train = torch.stack(train_data['sequence'])
    data_sequence_test = torch.stack(test_data['sequence'])
    
    data_list_train = [Data(x=torch.tensor(features, dtype=torch.float),
                              edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                              y=torch.tensor(label, dtype=torch.float))
                         for features, adj_matrix, label in zip(train_data['features'], adj_matrices_train, train_labels)]
    data_list_test = [Data(x=torch.tensor(features, dtype=torch.float),
                                edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous(),
                                y=torch.tensor(label, dtype=torch.float))
                            for features, adj_matrix, label in zip(test_data['features'], adj_matrices_test, test_labels)]

    train_dataset = CustomDataset(data_list_train, train_data['sequence'])
    test_dataset = CustomDataset(data_list_test, test_data['sequence'])
    
    if options[0]:
        graph_model = Graph_Transformer(in_channels=input_dim_train, hidden_channels=64, out_channels=1, heads=4).to(device)
    if options[1]:
        sequence_model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward).to(device)
    
    if options[2]:
        fusion_model = FusionModel(graph_model, sequence_model).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001, weight_decay=5e-4)
    
    
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join('saved_models', formatted_datetime)
    os.makedirs(folder_path, exist_ok=True)


    best_train_accuracy = 0.0
    best_test_accuracy=0.0
    
    # Training loop
    for epoch in range(100):
        total_correct = 0
        total_samples = 0
        for data_batch in train_dataset:
            graph_data_batch = data_batch[0]
            sequence_inputs  = data_batch[1]
            sequence_targets=graph_data_batch.y

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = fusion_model(graph_data_batch, sequence_inputs)
            
            # Compute binary predictions
            binary_predictions = (output >= 0.5).float()

            # Compute batch accuracy
            batch_correct = (binary_predictions == sequence_targets).sum().item()
            total_correct += batch_correct
            total_samples += 1

            output = output.to(device)
            sequence_targets = sequence_targets.view(-1, 1).to(device)
            
            # Compute loss
            loss = criterion(output, sequence_targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()


        # Compute epoch accuracy
        epoch_train_accuracy = (total_correct / total_samples)*100
        print(f"Epoch {epoch+1}/{100}, Epoch Accuracy: {epoch_train_accuracy:.4f}")
        
        if epoch_train_accuracy >= best_train_accuracy:
            best_train_accuracy = epoch_train_accuracy
            model_path = os.path.join(folder_path, f'train_best_model_{best_train_accuracy:.3f}.pth')
            torch.save(fusion_model.state_dict(), model_path)
            print("Saved model with accuracy train model with accuracy{:.2f}% to {}".format(best_train_accuracy, model_path))

        total_correct = 0
        total_samples = 0

        for data_batch in test_dataset:
            graph_data_batch = data_batch[0]
            sequence_inputs = data_batch[1]
            sequence_targets = graph_data_batch.y
            
            
            output = fusion_model(graph_data_batch, sequence_inputs)
            binary_predictions = (output >= 0.5).float()
            
            
            batch_correct = (binary_predictions == sequence_targets).sum().item()
            total_correct += batch_correct
            total_samples += 1
        epoch_test_accuracy = (total_correct / total_samples)*100
        print(f"Epoch Testing Accuracy : {epoch_test_accuracy:.4f}")
        
        if epoch_test_accuracy >= best_test_accuracy:
            best_test_accuracy = epoch_test_accuracy
            model_path = os.path.join(folder_path, f'test_best_model_{best_test_accuracy:.3f}.pth')
            torch.save(fusion_model.state_dict(), model_path)
            print("Saved model with Test Model with accuracy {:.2f}% to {}".format(best_test_accuracy, model_path))
            
    



if __name__ == "__main__":
    options=[True,True,True]
    data_name='tox21'
    main(data_name,options)
    