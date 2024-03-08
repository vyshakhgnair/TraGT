import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import defaultdict


def load_data(dataset, device):

    data_file = f"./original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    node_types = set()
    label_types = set()
    tr_len = 0
    for line in file:
        tr_len += 1
        smiles = line.split("\t")[1]
        s = []
        mol = AllChem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            s.append(atom.GetAtomicNum())
        node_types |= set(s)
        label = line.split("\t")[2][:-1]
        label_types.add(label)
    file.close()

    te_len = 0
    data_file = f"./original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    for line in file:
        te_len += 1
        smiles = line.split("\t")[1]
        s = []
        mol = AllChem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            s.append(atom.GetAtomicNum())
        node_types |= set(s)
        label = line.split("\t")[2][:-1]
        label_types.add(label)
    file.close()

    print(tr_len)
    print(te_len)

    node2index = {n: i for i, n in enumerate(node_types)}
    label2index = {l: i for i, l in enumerate(label_types)}

    #print(node2index)
    #print(label2index)

    data_file = f"./original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    train_adjlists = []
    train_features = []
    train_sequence = []
    train_labels = torch.zeros(tr_len)
    for line in file:
        smiles = line.split("\t")[1]
        label = line.split("\t")[2][:-1]
        mol = AllChem.MolFromSmiles(smiles)
        feature = torch.zeros(len(mol.GetAtoms()), len(node_types))

        l = 0
        smiles_seq = []
        for atom in mol.GetAtoms():
            feature[l, node2index[atom.GetAtomicNum()]] = 1
            smiles_seq.append(node2index[atom.GetAtomicNum()])
            l += 1
        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            typ = bond.GetBondType()
            adj_list[i].append(j)
            adj_list[j].append(i)
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
                adj_list[i].append(j)
                adj_list[j].append(i)

        train_labels[len(train_adjlists)]= int(label2index[label])
        train_adjlists.append(adj_list)
        train_features.append(torch.FloatTensor(feature).to(device))
        train_sequence.append(torch.tensor(smiles_seq))
    file.close()

    data_file = f"./original_datasets/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    test_adjlists = []
    test_features = []
    test_sequence = []
    test_labels = np.zeros(te_len)
    for line in file:
        smiles = line.split("\t")[1]
        # print(smiles)
        label = line.split("\t")[2][:-1]
        mol = AllChem.MolFromSmiles(smiles)
        feature = torch.zeros(len(mol.GetAtoms()), len(node_types))
        l = 0
        smiles_seq = []
        for atom in mol.GetAtoms():
            feature[l, node2index[atom.GetAtomicNum()]] = 1
            smiles_seq.append(node2index[atom.GetAtomicNum()])
            l += 1
        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            typ = bond.GetBondType()
            adj_list[i].append(j)
            adj_list[j].append(i)
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
                adj_list[i].append(j)
                adj_list[j].append(i)

        test_labels[len(test_adjlists)] = int(label2index[label])
        test_adjlists.append(adj_list)
        test_features.append(torch.FloatTensor(feature).to(device))
        test_sequence.append(torch.tensor(smiles_seq))
    file.close()

    train_data = {}
    train_data['adj_lists'] = train_adjlists
    train_data['features'] = train_features
    train_data['sequence'] = train_sequence

    test_data = {}
    test_data['adj_lists'] = test_adjlists
    test_data['features'] = test_features
    test_data['sequence'] = test_sequence
    return train_data, train_labels, test_data, test_labels