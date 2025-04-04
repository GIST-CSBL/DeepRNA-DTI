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

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score, \
    roc_auc_score


class DeepRNA_DTI(nn.Module):
    def __init__(self, RNAmodel, CompoundModel):
        super(DeepRNA_DTI, self).__init__()
        # RNA embedding
        self.rna_embedding = RNAmodel
        self.rna_linear = nn.Linear(640, 128)
        self.rna_global_linear = nn.Linear(128, 32)

        # compound embedding
        self.compound_embedding = CompoundModel
        self.compound_linear = nn.Linear(300, 128)
        self.compound_linear2 = nn.Linear(128, 128)

        # shared network
        self.shared_linear0 = nn.Linear(128, 64)
        self.shared_linear1 = nn.Linear(64, 32)

        # bs attention weight
        self.att_weight1 = nn.Parameter(torch.randn(128, 8))
        self.att_weight2 = nn.Parameter(torch.randn(8, 128))

        # bs prediction
        self.bs_linear1 = nn.Linear(32, 16)
        self.bs_linear2 = nn.Linear(16, 1)

        # dti prediction
        self.dti_linear1 = nn.Linear(32, 16)
        self.dti_linear2 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def bs_classifier(self, att):

        dti_pred = self.dti_classifier(att)
        dti_pred = self.sigmoid(dti_pred)
        dti_pred = dti_pred.unsqueeze(-1).unsqueeze(-1)
        att = att[:, 1:, 1:, :]

        att = dti_pred * att

        bs_pred = self.relu(self.bs_linear1(att))
        bs_pred = self.bs_linear2(bs_pred)
        bs_pred = torch.mean(bs_pred, 2)
        bs_pred = self.sigmoid(bs_pred)
        return bs_pred

    def dti_classifier(self, att):

        att = torch.mean(att, (1, 2))
        dti_pred = self.relu(self.dti_linear1(att))
        dti_pred = self.dti_linear2(dti_pred)
        return dti_pred

    def forward(self, rna_input, compound_input, task):
        # rna processing
        rna_mask = (rna_input >= 4) & (rna_input <= 7)
        rna_mask = rna_mask.unsqueeze(-1)

        # rna shared weight
        rna = self.rna_embedding(rna_input, repr_layers=[12])
        rna = rna["representations"][12]
        rna = self.relu(self.rna_linear(rna))
        rna = rna_mask * rna
        self.rna_global_token = torch.mean(rna, 1, keepdim=True)  # [32,N,128] -> [32, 1, 128]

        # compound shared weight
        batch = compound_input[3]
        _, compound = self.compound_embedding(compound_input[0], compound_input[1], compound_input[2],
                                                 compound_input[3])
        compound = self.relu(self.compound_linear(compound))
        compound = self.relu(self.compound_linear2(compound))
        compound, mask = to_dense_batch(compound, batch)
        self.compound_global_token = torch.mean(compound, 1, keepdim=True)

        rna = torch.cat([self.rna_global_token.expand(rna.shape[0], -1, -1), rna], dim=1)
        compound = torch.cat([self.compound_global_token.expand(compound.shape[0], -1, -1), compound], dim=1)
        compound = torch.transpose(compound, 1, 2)

        # att calculation
        att_weight = torch.matmul(self.att_weight1, self.att_weight2)
        att = torch.matmul(rna, att_weight)
        att = torch.matmul(att, compound)

        att_expanded = att.unsqueeze(-1)
        rna_expanded = rna.unsqueeze(2)
        compound = torch.transpose(compound, 1, 2)
        compound_expanded = compound.unsqueeze(1)

        rna_compound = att_expanded * rna_expanded * compound_expanded

        rc = self.relu(self.shared_linear0(rna_compound))
        rc = self.relu(self.shared_linear1(rc))

        # binding site prediction
        if task == 'bs':
            bs_pred = self.bs_classifier(rc)
            return bs_pred

        # dti prediction
        elif task == 'dti':
            dti_pred = self.dti_classifier(rc)
            return dti_pred