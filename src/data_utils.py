import os
import sys
import pandas as pd
import numpy as np
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import to_dense_batch

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score
sys.path.append('./Model/pretrained_model/RNA-FM')
import fm
_, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()

sys.path.append('./Model/pretrained_model/Mole-BERT')
from model import GNN, GNN_graphpred
from loader import mol_to_graph_data_obj_simple, allowable_features

class RNADataset(Dataset):
    def __init__(self, root):
        input_df = pd.read_csv(f'{root}/raw/interactions.csv', sep=',')
        self.rna_sequences = input_df['sequence'].tolist()
        self.binding_sites = input_df['binding_site_index'].tolist()

        seqs = [(s, self.rna_sequences[s]) for s in range(len(self.rna_sequences))]
        _, _, self.seq_tokens = batch_converter(seqs)

    def __len__(self):
        return len(self.rna_sequences)

    def __getitem__(self, idx):
        rna_sequence_tokens = self.seq_tokens[idx]
        rna_seq_len = torch.tensor(len(eval(self.binding_sites[idx])))
        rna_binding_sites = torch.tensor(eval(self.binding_sites[idx]))
        rna_binding_sites = torch.cat(
            [torch.zeros(1), rna_binding_sites, torch.zeros(len(rna_sequence_tokens) - rna_seq_len - 1)])

        return rna_sequence_tokens, rna_binding_sites, rna_seq_len


class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.interaction_data_path = root
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):
        input_df = pd.read_csv(f'{self.interaction_data_path}/raw/interactions.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
        rdkit_mol_objs = [AllChem.MolFromSmiles(s) for s in smiles_list]
        data_list = []
        data_smiles_list = []

        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            data = mol_to_graph_data_obj_simple(rdkit_mol)
            data_list.append(data)
            data_smiles_list.append(smiles_list[i])

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphDataset_test(Dataset):
    def __init__(self, root, file_name='interactions.csv'):
        self.interaction_data_path = root
        self.file_name = file_name
        input_df = pd.read_csv(f'{self.interaction_data_path}/raw/{self.file_name}', sep=',')
        self.smiles_list = input_df['smiles'].tolist()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES : {smiles}')

        data = mol_to_graph_data_obj_simple(mol)

        return data

class InteractionDataset:
    def __init__(self, root, graph_dataset, rna_dataset):
        input_df = pd.read_csv(f'{root}/raw/interactions.csv', sep=',')
        self.labels = input_df['interactions']
        self.graph_dataset = graph_dataset
        self.rna_dataset = rna_dataset


    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        compound_graph_data = self.graph_dataset[idx]
        rna_sequence_tokens, rna_binding_sites, rna_seq_len = self.rna_dataset[idx]
        interaction = torch.tensor(self.labels[idx])

        return compound_graph_data, rna_sequence_tokens, rna_binding_sites, rna_seq_len, interaction



class CombinedDataLoader:
    def __init__(self, bs_loader, dti_loader):
        self.bs_loader = bs_loader
        self.dti_loader = dti_loader

        self.iter_bs = iter(self.bs_loader)
        self.iter_dti = iter(self.dti_loader)

        self.bs_exhausted = False
        self.extra_iters = 0

    def __iter__(self):
        self.iter_dti = iter(self.dti_loader)
        self.iter_bs = iter(self.bs_loader)
        self.bs_exhausted = False
        self.extra_iters = 0
        return self

    def __next__(self):

        try:
            batch_bs = next(self.iter_bs)
        except StopIteration:
            self.iter_bs = iter(self.bs_loader)
            batch_bs = next(self.iter_bs)


        try:
            batch_dti = next(self.iter_dti)
        except StopIteration:
            raise StopIteration

        if self.bs_exhausted:
            self.extra_iters += 1

        return batch_bs, batch_dti


class CombinedDataLoader_rawdata:
    def __init__(self, bs_loader, dti_loader):
        self.bs_loader = bs_loader
        self.dti_loader = dti_loader

        self.iter_bs = iter(self.bs_loader)
        self.iter_dti = iter(self.dti_loader)

        self.bs_exhausted = False
        self.extra_iters = 0

    def __iter__(self):
        self.iter_dti = iter(self.dti_loader)
        self.iter_bs = iter(self.bs_loader)
        self.bs_exhausted = False
        self.extra_iters = 0
        return self

    def __next__(self):

        try:
            if self.bs_exhausted:
                batch_bs = None
            else:
                batch_bs = next(self.iter_bs)
        except StopIteration:
            self.iter_bs = None
            batch_bs = None
            if not self.bs_exhausted:
                self.bs_exhausted = True


        try:
            batch_dti = next(self.iter_dti)
        except StopIteration:
            raise StopIteration

        if self.bs_exhausted:
            self.extra_iters += 1

        return batch_bs, batch_dti