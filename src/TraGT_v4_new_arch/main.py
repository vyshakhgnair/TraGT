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

from sklearn.model_selection import KFold
from torch.utils.data import Subset
import copy

def train_and_evaluate(fusion_model, classifier, train_subset, val_subset, device, optimizer, criterion, recon, recon_weight, num_epochs, fold, file_train, file_test):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training
        fusion_model.train()
        classifier.train()
        total_correct = 0
        total_samples = 0
        true_labels_train = []
        pred_probs_train = []
        losses = 0.0

        for data_batch in train_subset:
            graph_data_batch = data_batch[0].to(device)
            sequence_inputs = data_batch[1].to(device)
            sequence_targets = graph_data_batch.y.view(1, 1)

            optimizer.zero_grad()
            output, sequence, reconstruction = fusion_model(graph_data_batch, sequence_inputs)
            output = classifier(output)

            binary_predictions = (output >= 0.5).float()
            total_correct += (binary_predictions == sequence_targets).sum().item()
            total_samples += 1

            true_labels_train.append(sequence_targets.cpu().numpy().reshape(-1))
            pred_probs_train.append(output.detach().cpu().numpy())

            sequence_inputs = sequence_inputs.float()
            if recon:
                reconstruction_loss = nn.MSELoss()(reconstruction, sequence)
                loss = criterion(output, sequence_targets) + reconstruction_loss * recon_weight
            else:
                loss = criterion(output, sequence_targets)

            losses += loss.item()
            loss.backward()
            optimizer.step()

        epoch_train_accuracy = (total_correct / total_samples) * 100
        true_labels_train = np.concatenate(true_labels_train)
        pred_probs_train = np.concatenate(pred_probs_train)
        precision_train = precision_score(true_labels_train, (pred_probs_train >= 0.5).astype(int))
        recall_train = recall_score(true_labels_train, (pred_probs_train >= 0.5).astype(int))
        auc_roc_train = roc_auc_score(true_labels_train, pred_probs_train)
        f1_train = f1_score(true_labels_train, (pred_probs_train >= 0.5).astype(int))
        file_train.write(f'[Fold {fold}] Epoch {epoch + 1}/{num_epochs}, Loss: {losses:.4f}, Acc: {epoch_train_accuracy:.2f}, AUC: {auc_roc_train:.4f}, F1: {f1_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}\n')

        # Validation
        fusion_model.eval()
        classifier.eval()
        total_correct = 0
        total_samples = 0
        true_labels_val = []
        pred_probs_val = []

        for data_batch in val_subset:
            graph_data_batch = data_batch[0].to(device)
            sequence_inputs = data_batch[1].to(device)
            sequence_targets = graph_data_batch.y.view(1, 1)

            with torch.no_grad():
                output, sequence, reconstruction = fusion_model(graph_data_batch, sequence_inputs)
                output = classifier(output)

            binary_predictions = (output >= 0.5).float()
            total_correct += (binary_predictions == sequence_targets).sum().item()
            total_samples += 1

            true_labels_val.append(sequence_targets.cpu().numpy().reshape(-1))
            pred_probs_val.append(output.cpu().numpy())

        epoch_val_accuracy = (total_correct / total_samples) * 100
        true_labels_val = np.concatenate(true_labels_val)
        pred_probs_val = np.concatenate(pred_probs_val)
        precision_val = precision_score(true_labels_val, (pred_probs_val >= 0.5).astype(int))
        recall_val = recall_score(true_labels_val, (pred_probs_val >= 0.5).astype(int))
        auc_roc_val = roc_auc_score(true_labels_val, pred_probs_val)
        f1_val = f1_score(true_labels_val, (pred_probs_val >= 0.5).astype(int))
        file_test.write(f'[Fold {fold}] Epoch {epoch + 1}/{num_epochs}, Acc: {epoch_val_accuracy:.2f}, AUC: {auc_roc_val:.4f}, F1: {f1_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}\n')

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            print(f'[Fold {fold}] New best model at epoch {epoch + 1} with accuracy: {best_val_accuracy:.2f}%')

    return best_val_accuracy


def main_cv(data_name, options, k_folds=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data, train_labels, test_data, test_labels = load_data_long(data_name, device=device)
    adj_matrices_train = [adj_list_to_adj_matrix(adj_list) for adj_list in train_data['adj_lists']]
    data_list_train = [Data(x=torch.tensor(features, dtype=torch.float).to(device),
                            edge_index=torch.nonzero(adj_matrix, as_tuple=False).t().contiguous().to(device),
                            y=torch.tensor(label, dtype=torch.float).to(device))
                       for features, adj_matrix, label in zip(train_data['features'], adj_matrices_train, train_labels)]
    full_dataset = CustomDataset(data_list_train, train_data['sequence'])

    output_dir_train = f'output/{data_name}/train'
    output_dir_test = f'output/{data_name}/test'
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name_train = os.path.join(output_dir_train, f'train_cv_{timestamp}.txt')
    file_name_test = os.path.join(output_dir_test, f'test_cv_{timestamp}.txt')

    num_epochs = 50
    recon_weight = 1
    recon = options[3]

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_fold_accuracies = []

    with open(file_name_train, 'w') as file_train, open(file_name_test, 'w') as file_test:
        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
            print(f'--- Fold {fold + 1}/{k_folds} ---')

            train_subset = [full_dataset[i] for i in train_ids]
            val_subset = [full_dataset[i] for i in val_ids]

            # Reinitialize models per fold
            graph_model = Graph_Transformer(in_channels=train_data['features'][0].size(-1), hidden_channels=32, heads=4).to(device) if options[0] else None
            sequence_model = TransformerModel(128, 4, 32, 6).to(device) if options[1] else None
            fusion_model = FusionModel(graph_model, sequence_model).to(device) if options[2] else None
            classifier = SimpleClassifier(128, output_size=1).to(device)

            optimizer = optim.Adam(list(fusion_model.parameters()) + list(classifier.parameters()), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()

            best_val_accuracy = train_and_evaluate(
                fusion_model, classifier, train_subset, val_subset,
                device, optimizer, criterion, recon, recon_weight,
                num_epochs, fold + 1, file_train, file_test
            )
            all_fold_accuracies.append(best_val_accuracy)

    print(f"\nCross-validation results: {all_fold_accuracies}")
    print(f"Average Accuracy: {np.mean(all_fold_accuracies):.2f}%")

if __name__ == "__main__":
    import os
    print("Current working directory:", os.getcwd())
    options = [True, True, True, True]
    data_name = 'logp'
    main_cv(data_name, options, k_folds=5)
