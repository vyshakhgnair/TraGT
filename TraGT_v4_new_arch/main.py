from load_data import load_data, load_data_long, CustomDataset, adj_list_to_adj_matrix
import torch
from torch_geometric.data import Data
from seq_models import TransformerModel
from graph_models import Graph_Transformer
from fusion_model import FusionModel
from datetime import datetime
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from classifier import SimpleClassifier

import torch.optim as optim
import torch.nn as nn


def main(data_name, options):
    
    vocab_size = 100
    d_model = 100
    nhead = 2
    num_encoder_layers = 3
    dim_feedforward = 512
    max_length = 100
    num_epochs = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, train_labels, test_data, test_labels = load_data_long(data_name, device=device)
    
    input_dim_train = train_data['features'][0].size(-1)
    input_dim_test = test_data['features'][0].size(-1)

    adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
    adj_matrices_test = [adj_list_to_adj_matrix(adj_list) for adj_list in test_data['adj_lists']]
    
    data_list_train = [Data(x=torch.tensor(features, dtype=torch.float).to(device),
                            edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous().to(device),
                            y=torch.tensor(label, dtype=torch.float).to(device))
                       for features, adj_matrix, label in zip(train_data['features'], adj_matrices_train, train_labels)]
    
    data_list_test = [Data(x=torch.tensor(features, dtype=torch.float).to(device),
                           edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous().to(device),
                           y=torch.tensor(label, dtype=torch.float).to(device))
                      for features, adj_matrix, label in zip(test_data['features'], adj_matrices_test, test_labels)]

    train_dataset = CustomDataset(data_list_train, train_data['sequence'])
    test_dataset = CustomDataset(data_list_test, test_data['sequence'])
    
    # Initialize the values for the model
    in_channels = train_data['features'][0].size(-1)
    hidden_channels = 32
    num_heads = 4  # Number of attention heads
    num_layers = 6  # Number of transformer layers
    output_dim = 1  # Output for fusion
    num_hidden_channels = hidden_channels
    classifier = SimpleClassifier(128, output_size=1).to(device)

    if options[0]:
        graph_model = Graph_Transformer(in_channels, hidden_channels, 4).to(device)
    if options[1]:
        sequence_model = TransformerModel(128, 4, num_hidden_channels, num_layers).to(device)
    
    if options[2]:
        fusion_model = FusionModel(graph_model, sequence_model).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adam(fusion_model.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = optim.Adam(list(graph_model.parameters()) + list(sequence_model.parameters()), lr=0.001)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')
    session_name = f'{data_name}_{formatted_datetime}'
    folder_path = os.path.join('saved_models', session_name)
    os.makedirs(folder_path, exist_ok=True)
    
    output_dir_train = f'output/{data_name}/train'
    os.makedirs(output_dir_train, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name_train = f'{output_dir_train}/train_accuracy_details_{current_time}.txt'

    output_dir_test = f'output/{data_name}/test'
    os.makedirs(output_dir_test, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name_test = f'{output_dir_test}/test_accuracy_details_{current_time}.txt'

    best_train_accuracy = 0.0
    best_test_accuracy = 0.0


    recon = options[3]
    recon_weight = 1
    # Training loop
    with open(file_name_train, 'a') as file_train, open(file_name_test, 'a') as file_test:
        for epoch in range(100):
            total_correct = 0
            total_samples = 0
            true_labels_train = []
            pred_probs_train = []
            losses = 0.0

            for data_batch in train_dataset:
                graph_data_batch = data_batch[0].to(device)
                sequence_inputs = data_batch[1].to(device)
                sequence_targets = graph_data_batch.y

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output,sequence,reconstruction = fusion_model(graph_data_batch, sequence_inputs)
                output = classifier(output)
                

                sequence_targets = sequence_targets.view(1, 1)
                # Compute binary predictions
                binary_predictions = (output >= 0.5).float()
                
                # Compute batch accuracy
                batch_correct = (binary_predictions == sequence_targets).sum().item()
                total_correct += batch_correct
                total_samples += 1

                true_labels_train.append(sequence_targets.cpu().numpy().reshape(-1))
                pred_probs_train.append(output.detach().cpu().numpy())
                
                # Cast sequence_inputs to float
                sequence_inputs = sequence_inputs.float()

                # Compute loss
                #print(output.shape,output, sequence_targets.shape, sequence_targets)
                if recon:
                    reconstruction_loss = nn.MSELoss()(reconstruction, sequence)
                    loss = criterion(output, sequence_targets) + reconstruction_loss * recon_weight
                    #print(loss)
                else:
                    loss = criterion(output, sequence_targets)
                losses += loss.item()

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

            # Compute epoch accuracy
            epoch_train_accuracy = (total_correct / total_samples) * 100
            print(f"Epoch {epoch+1}/{100}, Epoch Accuracy: {epoch_train_accuracy:.4f}")
            
            if epoch_train_accuracy >= best_train_accuracy:
                best_train_accuracy = epoch_train_accuracy
                model_path = os.path.join(folder_path, f'train_best_model_{best_train_accuracy:.3f}.pth')
                torch.save(fusion_model.state_dict(), model_path)
                print("Saved model with accuracy train model with accuracy{:.2f}% to {}".format(best_train_accuracy, model_path))
            
            true_labels_train = np.concatenate(true_labels_train)
            pred_probs_train = np.concatenate(pred_probs_train)
            
            precision_train = precision_score(true_labels_train, (pred_probs_train >= 0.5).astype(int))
            recall_train = recall_score(true_labels_train, (pred_probs_train >= 0.5).astype(int))
            auc_roc_train = roc_auc_score(true_labels_train, pred_probs_train)
            f1_train = f1_score(true_labels_train, (pred_probs_train >= 0.5).astype(int))
            print(f"Train AUC-ROC: {auc_roc_train:.4f}, Train F1 Score: {f1_train:.4f}, Train Precision: {precision_train:.4f}, Train Recall: {recall_train:.4f}\n")
            file_train.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {losses:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Train AUC-ROC: {auc_roc_train:.4f}, Train F1 Score: {f1_train:.4f}, Train Precision: {precision_train:.4f}, Train Recall: {recall_train:.4f}\n')
            
            total_correct = 0
            total_samples = 0
            true_labels_test = []
            pred_probs_test = []
            
            for data_batch in test_dataset:
                graph_data_batch = data_batch[0].to(device)
                sequence_inputs = data_batch[1].to(device)
                sequence_targets = graph_data_batch.y
                    
                output,sequence,reconstruction = fusion_model(graph_data_batch, sequence_inputs)
                output = classifier(output)
                binary_predictions = (output >= 0.5).float()
                    
                batch_correct = (binary_predictions == sequence_targets).sum().item()
                total_correct += batch_correct
                total_samples += 1
                
                true_labels_test.append(sequence_targets.cpu().numpy().reshape(-1))
                pred_probs_test.append(output.detach().cpu().numpy()[0])
                
            epoch_test_accuracy = (total_correct / total_samples) * 100
            print(f"Epoch Testing Accuracy : {epoch_test_accuracy:.4f}")
                
            if epoch_test_accuracy >= best_test_accuracy:
                best_test_accuracy = epoch_test_accuracy
                model_path = os.path.join(folder_path, f'test_best_model_{best_test_accuracy:.3f}.pth')
                torch.save(fusion_model.state_dict(), model_path)
                print("Saved model with Test Model with accuracy {:.2f}% to {}".format(best_test_accuracy, model_path))
            
            true_labels_test = np.concatenate(true_labels_test)
            pred_probs_test = np.concatenate(pred_probs_test)
            
            precision_test = precision_score(true_labels_test, (pred_probs_test >= 0.5).astype(int))
            recall_test = recall_score(true_labels_test, (pred_probs_test >= 0.5).astype(int))
            auc_roc_test = roc_auc_score(true_labels_test, pred_probs_test)
            f1_test = f1_score(true_labels_test, (pred_probs_test >= 0.5).astype(int))
            
            print(f"Test AUC-ROC: {auc_roc_test:.4f}, Test F1 Score: {f1_test:.4f}, Test Precision: {precision_test:.4f}, Test Recall: {recall_test:.4f}\n")
            file_test.write(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {epoch_test_accuracy:.4f},Test AUC-ROC: {auc_roc_test:.4f}, Test F1 Score: {f1_test:.4f}, Test Precision: {precision_test:.4f}, Test Recall: {recall_test:.4f} \n')
    
    file_test.close()
    file_train.close()
    

if __name__ == "__main__":
    options = [True, True, True, True]
    data_name = 'logp'
    main(data_name, options)

