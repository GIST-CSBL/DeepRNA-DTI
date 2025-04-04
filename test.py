import os
import sys
import pandas as pd
import numpy as np
from itertools import repeat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder_path', default='./Model/trained_weight')
parser.add_argument('--data_folder_path', default='./Dataset/processed')
parser.add_argument('--molebert_path', default='./Model/pretrained_model/Mole-BERT')
parser.add_argument('--rnafm_path', default='./Model/pretrained_model/RNA-FM')
parser.add_argument('--device', default='0')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--test_category', default='total')


args = parser.parse_args()

model_folder_path = args.model_folder_path
data_folder_path = args.data_folder_path
molebert_path = args.molebert_path
rnafm_path = args.rnafm_path
device = args.device
batch_size = args.batch_size
test_category = args.test_category

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

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

from src.utils import FocalLoss_per_sample, dti_evaluate, bs_evaluate, batch_processing
from src.model import DeepRNA_DTI
from src.data_utils import RNADataset, GraphDataset, InteractionDataset, CombinedDataLoader


if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

seed = 42
runseed = 0
torch.manual_seed(runseed)
np.random.seed(runseed)

sys.path.append(molebert_path)
from model import GNN, GNN_graphpred
from loader import mol_to_graph_data_obj_simple, allowable_features

sys.path.append(rnafm_path)
import fm


def test_total():
    dti_test_path = f'{data_folder_path}/total_data/test_fold'
    dti_test_dataset = InteractionDataset(dti_test_path, GraphDataset(dti_test_path), RNADataset(dti_test_path))
    dti_test_loader = DataLoader(dti_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    bs_test_path = f'{data_folder_path}/bs_data/test_fold'
    bs_test_dataset = InteractionDataset(bs_test_path, GraphDataset(bs_test_path), RNADataset(bs_test_path))
    bs_test_loader = DataLoader(bs_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print('Data load successfully')

    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    molebert_model = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1)
    molebert_model.from_pretrained(f'{molebert_path}/model_gin/Mole-BERT.pth')

    print('pre-trained model load successfully')

    dti_aucs = []
    dti_auprs = []
    bs_micro_aucs = []
    bs_micro_auprs = []

    for fold in range(5):
        model = DeepRNA_DTI(rna_fm_model, molebert_model)
        if model_folder_path != './Model/trained_weight':
            model.load_state_dict(torch.load(f'{model_folder_path}/model_fold{fold}.pt', map_location=device)['model_state_dict'])
        if model_folder_path == './Model/trained_weight':
            model.load_state_dict(torch.load(f'{model_folder_path}/model_fold{fold}.pt', map_location=device))

        model = model.to(device)
        model.eval()
        val_auc, val_aupr = dti_evaluate(model, dti_test_loader, device)
        dti_aucs.append(val_auc);dti_auprs.append(val_aupr)
        bs_macro_auc, bs_macro_aupr, bs_micro_auc, bs_micro_aupr = bs_evaluate(model, bs_test_loader, device)
        bs_micro_aucs.append(bs_micro_auc); bs_micro_auprs.append(bs_micro_aupr)

    print('DTI AUC : ', round(np.mean(dti_aucs), 3))
    print('DTI AUPR : ', round(np.mean(dti_auprs), 3))
    print('BS per sample AUC : ', round(np.mean(bs_micro_aucs), 3))
    print('BS per sample AUPR : ', round(np.mean(bs_micro_auprs), 3))

def test_type():
    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    molebert_model = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1)
    molebert_model.from_pretrained(f'{molebert_path}/model_gin/Mole-BERT.pth')
    print('pre-trained model load successfully')

    types = ['aptamer','miRNA','repeats','ribosomal','riboswitch','viral']

    type_loaders=[]
    for rna_type in types:
        dti_test_path = f'{data_folder_path}/type_{rna_type}/test_fold'
        dti_test_dataset = InteractionDataset(dti_test_path, GraphDataset(dti_test_path), RNADataset(dti_test_path))
        dti_test_loader = DataLoader(dti_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        type_loaders.append(dti_test_loader)
        dti_aucs = []
        dti_auprs = []

        for fold in range(5):
            model = DeepRNA_DTI(rna_fm_model, molebert_model)
            model.load_state_dict(torch.load(f'{model_folder_path}/model_fold{fold}.pt', map_location=device))

            model = model.to(device)
            model.eval()
            val_auc, val_aupr = dti_evaluate(model, dti_test_loader, device)
            dti_aucs.append(val_auc); dti_auprs.append(val_aupr)
        print(f'{rna_type} DTI AUC : {round(np.mean(dti_aucs), 3)}, AUPR : {round(np.mean(dti_auprs), 3)}' )


def test_deeprsma():
    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    molebert_model = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1)
    molebert_model.from_pretrained(f'{molebert_path}/model_gin/Mole-BERT.pth')

    print('pre-trained model load successfully')

    dti_aucs = []
    dti_auprs = []

    for fold in range(10):
        dti_test_path = f'{data_folder_path}/DeepRSMA/test_fold{fold}'
        dti_test_dataset = InteractionDataset(dti_test_path, GraphDataset(dti_test_path), RNADataset(dti_test_path))
        dti_test_loader = DataLoader(dti_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        model = DeepRNA_DTI(rna_fm_model, molebert_model)
        model.load_state_dict(torch.load(f'{model_folder_path}/DeepRSMA_data/model_fold{fold}.pt', map_location=device))

        model = model.to(device)
        model.eval()
        val_auc, val_aupr = dti_evaluate(model, dti_test_loader, device)
        dti_aucs.append(val_auc);
        dti_auprs.append(val_aupr)

    print('DTI AUC : ', round(np.mean(dti_aucs), 3))
    print('DTI AUPR : ', round(np.mean(dti_auprs), 3))




if test_category=='total':
    test_total()

if test_category=='type':
    test_type()

if test_category=='DeepRSMA':
    test_deeprsma()



print()