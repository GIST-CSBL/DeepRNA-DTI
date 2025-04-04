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

def batch_processing(batch, device):
    batch[0] = batch[0].to(device)
    compound_input = batch[0].x, batch[0].edge_index, batch[0].edge_attr, batch[0].batch
    rna_input = batch[1].to(device)
    bs_label = batch[2].unsqueeze(2).to(device)
    dti_label = batch[4].to(device)

    return rna_input, compound_input, bs_label, dti_label\



def dti_evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        rna_input, compound_input, _, dti_label = batch_processing(batch, device)
        with torch.no_grad():
            dti_pred = model(rna_input, compound_input, task='dti')

        y_true.append(dti_label.view(dti_pred.shape).to(torch.float32))
        y_pred.append(dti_pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    dti_aupr_value = average_precision_score(y_true, y_pred)
    dti_auc_value = auc(fpr, tpr)

    return dti_auc_value, dti_aupr_value


def bs_evaluate(model, loader, device):
    model.eval()

    bs_labels = []
    bs_preds = []
    rna_per_aucs = []
    rna_per_auprs = []
    for step, batch in enumerate(loader):
        rna_input, compound_input, bs_label, dti_label = batch_processing(batch, device)

        with torch.no_grad():
            bs_pred = model(rna_input, compound_input, task='bs')
        bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(
            -1)
        bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1],
                                                                       1)
        bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

        filtered_predictions = bs_pred[bs_label_mask_true_seq].squeeze(-1)
        filtered_labels = bs_label[bs_label_mask_true_seq].squeeze(-1) 

        bs_labels.append(filtered_labels)
        bs_preds.append(filtered_predictions)

        for sample in range(bs_pred.size(0)):
            predictions = bs_pred[sample].squeeze(-1)
            labels = bs_label[sample].squeeze(-1)

            non_special_mask = bs_label_mask_true_seq[sample].squeeze(-1)
            filtered_predictions = predictions[non_special_mask]
            filtered_labels = labels[non_special_mask]

            if sum(filtered_labels) == 0:
                pass
            else:
                auc = roc_auc_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy())
                aupr = average_precision_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy())
                rna_per_aucs.append(auc)
                rna_per_auprs.append(aupr)

    bs_labels = torch.cat(bs_labels, dim=0).view(-1, 1).cpu().numpy()
    bs_preds = torch.cat(bs_preds, dim=0).view(-1, 1).cpu().numpy()

    macro_auc = roc_auc_score(bs_labels, bs_preds)
    macro_aupr = average_precision_score(bs_labels, bs_preds)

    micro_auc = np.mean(rna_per_aucs)
    micro_aupr = np.mean(rna_per_auprs)

    return macro_auc, macro_aupr, micro_auc, micro_aupr


class FocalLoss_per_sample(nn.Module):
    def __init__(self,alpha=0.25, gamma=2):
        super(FocalLoss_per_sample, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-10

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        F_loss_sum = F_loss.sum(axis=1)
        seq_len = targets.sum(axis=1)
        F_loss_sample = F_loss_sum / (seq_len + self.epsilon)

        return F_loss_sample.mean()

