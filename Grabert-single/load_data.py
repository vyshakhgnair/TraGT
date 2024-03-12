import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import defaultdict
import re
import glob
import torch
import os,time
import pandas as pd
import random
import json
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, auc
from seq_models import Smiles_BERT, BERT_base





class Vocab(object):
	def __init__(self):
		self.pad_index = 0
		self.mask_index = 1
		self.unk_index = 2
		self.start_index = 3
		self.end_index = 4

		# check 'Na' later
		self.voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', \
							'(', 'n', '2', 'c', 'F', ')', '=', 'N', '3', 'S', '/', 's', '-', '+', 'o', 'P', \
							 'R', '\\', 'L', '#', 'X', '6', 'B', '7', '4', 'I', '5', 'i', 'p', '8', '9', '%', '0', '.', ':', 'A']

		self.dict = {s: i for i, s in enumerate(self.voca_list)}

	def __len__(self):
		return len(self.voca_list)

class FinetuningDataset(Dataset):
	def __init__(self, datapath, vocab, seq_len):
		self.vocab = vocab
		self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
		self.smiles_dataset = []
		self.adj_dataset = []
		
		self.seq_len = seq_len

		smiles_data = glob.glob(datapath)
		text = pd.read_csv(smiles_data[0])

		csv_columns = text.columns
		if len(csv_columns) == 2:
			try:
				if Chem.MolFromSmiles(text[csv_columns[0]][0]) != None and type(text[csv_columns[0]][0]) == str:
					smiles_list = np.asarray(text[csv_columns[0]])
					label_list = np.asarray(text[csv_columns[1]])
				else:
					smiles_list = np.asarray(text[csv_columns[1]])
					label_list = np.asarray(text[csv_columns[0]])
			except:
				smiles_list = np.asarray(text[csv_columns[1]])
				label_list = np.asarray(text[csv_columns[0]])
		else:
			raise NameError("The number of columns should be two (smiles and y_label).")
			print("The number of columns should be two. (smiles and y)")
			exit(1)
		'''
		try:
			smiles_list = np.asarray(text['smiles'])
			label_list = np.asarray(text['y'])
		except:
			print("Header should include smiles and y")
			exit(1)
		'''

		self.label = label_list.reshape(-1,1)
		for i in smiles_list:
			self.adj_dataset.append(i)
			self.smiles_dataset.append(self.replace_halogen(i))

	def __len__(self):
		return len(self.smiles_dataset)

	def __getitem__(self, idx):
		item = self.smiles_dataset[idx]
		label = self.label[idx]

		input_token, input_adj_masking = self.CharToNum(item)

		input_data = [self.vocab.start_index] + input_token + [self.vocab.end_index]
		input_adj_masking = [0] + input_adj_masking + [0]

		smiles_bert_input = input_data[:self.seq_len]
		smiles_bert_adj_mask = input_adj_masking[:self.seq_len]

		padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
		smiles_bert_input.extend(padding)
		smiles_bert_adj_mask.extend(padding)

		mol = Chem.MolFromSmiles(self.adj_dataset[idx])
		if mol != None:
			adj_mat = GetAdjacencyMatrix(mol)
			smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))
		else:
			smiles_bert_adjmat = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)

		output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": label,  \
					"smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat}
		

		return {key:torch.tensor(value) for key, value in output.items()}

	def CharToNum(self, smiles):
		tokens = [i for i in smiles]
		adj_masking = []

		for i, token in enumerate(tokens):
			if token in self.atom_vocab:
				adj_masking.append(1)
			else:
				adj_masking.append(0)

			tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

		return tokens, adj_masking


	def replace_halogen(self, string):
		"""Regex to replace Br and Cl with single letters"""
		br = re.compile('Br')
		cl = re.compile('Cl')
		sn = re.compile('Sn')
		na = re.compile('Na')
		string = br.sub('R', string)
		string = cl.sub('L', string)
		string = sn.sub('X', string)
		string = na.sub('A', string)
		return string

	def zero_padding(self, array, shape):
		if array.shape[0] > shape[0]:
			array = array[:shape[0],:shape[1]]
		padded = np.zeros(shape, dtype=np.float32)
		padded[:array.shape[0], :array.shape[1]] = array
		return padded


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
    label2index = {l: int(l) for i, l in enumerate(label_types)}

    #print(node2index)
    print(label2index)

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

        train_labels[len(train_adjlists)] = int(label2index[label])
        #print(train_labels)
        train_adjlists.append(adj_list)
        train_features.append(torch.FloatTensor(feature).to(device))
        train_sequence.append(torch.tensor(smiles_seq))
    file.close()

    data_file = f"./original_datasets/{dataset}/{dataset}_test"
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



