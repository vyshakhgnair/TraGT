from load_data import load_data_long, CustomDataset, adj_list_to_adj_matrix
import torch
from torch_geometric.data import Data
from seq_models import TransformerModel
from graph_models import Graph_Transformer
from fusion_model import FusionModel
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

def train_and_evaluate(train_indices, test_indices, all_data, all_labels, data_name, fold, options):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Extract fold-specific data
    train_data = {k: [v[i] for i in train_indices] for k, v in all_data.items()}
    test_data = {k: [v[i] for i in test_indices] for k, v in all_data.items()}
    train_labels = [all_labels[i] for i in train_indices]
    test_labels = [all_labels[i] for i in test_indices]

    input_dim = train_data['features'][0].size(-1)

    adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
    adj_matrices_test = [adj_list_to_adj_matrix(adj_list) for adj_list in test_data['adj_lists']]

    data_list_train = [Data(x=torch.tensor(f, dtype=torch.float),
                            edge_index=torch.nonzero(adj, as_tuple=False).t().contiguous(),
                            y=torch.tensor(label, dtype=torch.float))
                       for f, adj, label in zip(train_data['features'], adj_matrices_train, train_labels)]

    data_list_test = [Data(x=torch.tensor(f, dtype=torch.float),
                           edge_index=torch.nonzero(adj, as_tuple=False).t().contiguous(),
                           y=torch.tensor(label, dtype=torch.float))
                      for f, adj, label in zip(test_data['features'], adj_matrices_test, test_labels)]

    train_dataset = CustomDataset(data_list_train, train_data['sequence'])
    test_dataset = CustomDataset(data_list_test, test_data['sequence'])

    # Model
    if options[0]:
        graph_model = Graph_Transformer(in_channels=input_dim, hidden_channels=64, out_channels=1, heads=4).to(device)
    if options[1]:
        sequence_model = TransformerModel(vocab_size=100, d_model=100, nhead=2, num_encoder_layers=3, dim_feedforward=512).to(device)
    if options[2]:
        fusion_model = FusionModel(graph_model, sequence_model).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001, weight_decay=5e-4)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_folder = os.path.join('model/saved_models', f"{data_name}_fold{fold}_{timestamp}")
    os.makedirs(model_folder, exist_ok=True)

    output_train_file = f'output/{data_name}/train/fold{fold}_train_accuracy_{timestamp}.txt'
    output_test_file = f'output/{data_name}/test/fold{fold}_test_accuracy_{timestamp}.txt'
    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_file), exist_ok=True)

    best_test_acc = 0.0

    with open(output_train_file, 'a') as f_train, open(output_test_file, 'a') as f_test:
        for epoch in range(1, 101):
            fusion_model.train()
            total_correct = 0
            total_samples = 0
            true_train = []
            pred_train = []
            losses = 0.0

            for graph_data_batch, sequence_inputs in train_dataset:
                sequence_targets = graph_data_batch.y.to(device)
                optimizer.zero_grad()
                output = fusion_model(graph_data_batch.to(device), sequence_inputs.to(device))
                binary_preds = (output >= 0.5).float()
                total_correct += (binary_preds == sequence_targets).sum().item()
                total_samples += 1
                true_train.append(sequence_targets.cpu().numpy().reshape(-1))
                pred_train.append(output.detach().cpu().numpy())
                loss = criterion(output, sequence_targets.view(-1, 1))
                losses += loss.item()
                loss.backward()
                optimizer.step()

            epoch_train_acc = (total_correct / total_samples) * 100
            true_train = np.concatenate(true_train)
            pred_train = np.concatenate(pred_train)
            prec_train = precision_score(true_train, (pred_train >= 0.5).astype(int))
            rec_train = recall_score(true_train, (pred_train >= 0.5).astype(int))
            auc_train = roc_auc_score(true_train, pred_train)
            f1_train = f1_score(true_train, (pred_train >= 0.5).astype(int))
            print(f"[FOLD {fold}] Epoch {epoch} Train Acc: {epoch_train_acc:.2f} | AUC: {auc_train:.4f} | F1: {f1_train:.4f}")
            f_train.write(f"Epoch {epoch}, Loss: {losses:.4f}, Accuracy: {epoch_train_acc:.4f}, AUC: {auc_train:.4f}, F1: {f1_train:.4f}, Precision: {prec_train:.4f}, Recall: {rec_train:.4f}\n")

            # Evaluation
            fusion_model.eval()
            total_correct = 0
            total_samples = 0
            true_test = []
            pred_test = []

            with torch.no_grad():
                for graph_data_batch, sequence_inputs in test_dataset:
                    sequence_targets = graph_data_batch.y.to(device)
                    output = fusion_model(graph_data_batch.to(device), sequence_inputs.to(device))
                    binary_preds = (output >= 0.5).float()
                    total_correct += (binary_preds == sequence_targets).sum().item()
                    total_samples += 1
                    true_test.append(sequence_targets.cpu().numpy().reshape(-1))
                    pred_test.append(output.cpu().numpy())

            epoch_test_acc = (total_correct / total_samples) * 100
            true_test = np.concatenate(true_test)
            pred_test = np.concatenate(pred_test)
            prec_test = precision_score(true_test, (pred_test >= 0.5).astype(int))
            rec_test = recall_score(true_test, (pred_test >= 0.5).astype(int))
            auc_test = roc_auc_score(true_test, pred_test)
            f1_test = f1_score(true_test, (pred_test >= 0.5).astype(int))
            print(f"[FOLD {fold}] Epoch {epoch} Test Acc: {epoch_test_acc:.2f} | AUC: {auc_test:.4f} | F1: {f1_test:.4f}")
            f_test.write(f"Epoch {epoch}, Accuracy: {epoch_test_acc:.4f}, AUC: {auc_test:.4f}, F1: {f1_test:.4f}, Precision: {prec_test:.4f}, Recall: {rec_test:.4f}\n")

            if epoch_test_acc > best_test_acc:
                best_test_acc = epoch_test_acc
                model_path = os.path.join(model_folder, f"best_model_fold{fold}_acc{best_test_acc:.2f}.pth")
                torch.save(fusion_model.state_dict(), model_path)
                print(f"Saved best model for fold {fold} at {model_path}")

def main(data_name, options, k_folds=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_data, all_labels, _, _ = load_data_long(data_name, device=device)
    labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n========== Fold {fold + 1} / {k_folds} ==========\n")
        train_and_evaluate(train_idx, test_idx, all_data, all_labels, data_name, fold + 1, options)

if __name__ == "__main__":
    options = [True, True, True]  # Graph, Sequence, Fusion
    data_name = 'bbbp'
    main(data_name, options, k_folds=5)
