import os
import sys
import pandas as pd
import numpy as np
from itertools import repeat

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder_path', default='./Model')
parser.add_argument('--data_folder_path', default='./Dataset/processed')
parser.add_argument('--molebert_path', default='./Model/pretrained_model/Mole-BERT')
parser.add_argument('--rnafm_path', default='./Model/pretrained_model/RNA-FM')
parser.add_argument('--device', default='0')
parser.add_argument('--batch_size', default=32)
parser.add_argument('--num_epochs', default=100)
parser.add_argument('--training_strategy',default='oversampled_pooled')
parser.add_argument('--data',default='DeepRNA-DTI')


args = parser.parse_args()

model_folder_path = args.model_folder_path
data_folder_path = args.data_folder_path
molebert_path = args.molebert_path
rnafm_path = args.rnafm_path
device = args.device
batch_size = args.batch_size
num_epochs = args.num_epochs
training_strategy = args.training_strategy
data = args.data

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

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score

from src.utils import FocalLoss_per_sample, dti_evaluate, bs_evaluate, batch_processing
from src.model import DeepRNA_DTI
from src.data_utils import RNADataset, GraphDataset, GraphDataset_test, InteractionDataset, CombinedDataLoader, CombinedDataLoader_rawdata
from src.training_strategies import select_training_strategies


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

_, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()

bs_loss_function = FocalLoss_per_sample()
dti_loss_function = nn.BCEWithLogitsLoss()


def get_dataloader(training_strategy,data_folder_path, fold):
    bs_train_path = f'{data_folder_path}/bs_data/train_fold{fold}'
    bs_val_path = f'{data_folder_path}/bs_data/val_fold{fold}'
    bs_train_dataset = InteractionDataset(bs_train_path, GraphDataset(bs_train_path), RNADataset(bs_train_path))
    bs_val_dataset = InteractionDataset(bs_val_path, GraphDataset(bs_val_path), RNADataset(bs_val_path))

    dti_train_path = f'{data_folder_path}/total_data/train_fold{fold}'
    dti_train_dataset = InteractionDataset(dti_train_path, GraphDataset(dti_train_path), RNADataset(dti_train_path))
    dti_val_path = f'{data_folder_path}/total_data/val_fold{fold}'
    dti_val_dataset = InteractionDataset(dti_val_path, GraphDataset(dti_val_path), RNADataset(dti_val_path))

    bs_train_loader = DataLoader(bs_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bs_val_loader = DataLoader(bs_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    dti_train_loader = DataLoader(dti_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dti_val_loader = DataLoader(dti_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    if 'oversampled' in training_strategy:
        combined_dataloader = CombinedDataLoader(bs_train_loader, dti_train_loader)
    elif 'raw' in training_strategy:
        combined_dataloader = CombinedDataLoader_rawdata(bs_train_loader, dti_train_loader)

    return bs_train_loader, bs_val_loader, dti_train_loader, dti_val_loader, combined_dataloader


def get_dataloader_deeprsma(data_folder_path, fold):

    dti_train_path = f'{data_folder_path}/DeepRSMA/train_fold{fold}'
    dti_train_dataset = InteractionDataset(dti_train_path, GraphDataset(dti_train_path), RNADataset(dti_train_path))
    dti_val_path = f'{data_folder_path}/DeepRSMA/test_fold{fold}'
    dti_val_dataset = InteractionDataset(dti_val_path, GraphDataset(dti_val_path), RNADataset(dti_val_path))

    dti_train_loader = DataLoader(dti_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dti_val_loader = DataLoader(dti_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return dti_train_loader, dti_val_loader

if data=='DeepRNA-DTI':
    for fold in range(5):
        bs_train_loader, bs_val_loader, dti_train_loader, dti_val_loader, combined_dataloader = get_dataloader(training_strategy, data_folder_path, fold)
        print('Data load successfully')

        # get pre-trained compound model
        molebert_model = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1, JK='last', drop_ratio=0.5,
                                       graph_pooling='mean', gnn_type='gin')
        molebert_model.from_pretrained(f'{molebert_path}/model_gin/Mole-BERT.pth')
        for name, param in molebert_model.named_parameters():
            param.requires_grad = False
        molebert_model = molebert_model.to(device)

        # get pre-trained RNA model
        rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
        for name, param in rna_fm_model.named_parameters():
            param.requires_grad = False
        rna_fm_model = rna_fm_model.to(device)
        print('pre-trained model load successfully')

        model = DeepRNA_DTI(rna_fm_model, molebert_model)
        model = model.to(device)
        for name, param in model.named_parameters():
            if 'rna_embedding' in name or 'compound_embedding' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


        if training_strategy !='oversampled_pooled':
            select_training_strategies(fold, training_strategy, num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                         dti_val_loader, bs_loss_function, dti_loss_function,device)


        elif training_strategy=='oversampled_pooled':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            best_set = {'epoch': 0, 'val_aupr': 0}

            bs_epoch_val_auc = []
            bs_epoch_val_aupr = []
            total_dti_epoch_val_auc = []
            total_dti_epoch_val_aupr = []

            for epoch in range(num_epochs):
                model.train()

                bs_epoch_losses = []
                dti_epoch_losses = []
                total_losses = []

                for step, (batch_bs, batch_dti) in enumerate(combined_dataloader):
                    optimizer.zero_grad()

                    # bs train
                    rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
                    bs_pred = model(rna_input, compound_input, task='bs')
                    bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
                    bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
                    bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

                    bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
                    bs_epoch_losses.append(bs_loss.item())

                    # dti train
                    rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
                    dti_pred = model(rna_input, compound_input, task='dti')
                    dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
                    dti_epoch_losses.append(dti_loss.item())

                    total_loss = bs_loss + dti_loss

                    total_losses.append(total_loss.item())
                    total_loss.backward()
                    optimizer.step()


                dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
                bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                                       device)

                if dti_val_aupr > best_set['val_aupr']:
                    best_set['val_aupr'] = dti_val_aupr
                    best_set['epoch'] = epoch
                    torch.save(model.state_dict(), f'{model_folder_path}/model_fold{fold}.pt')

                print(f'[Fold {fold}, Epoch {epoch}]'
                      f'Loss - BS:{np.mean(bs_epoch_losses):.3f},  DTI:{np.mean(dti_epoch_losses):.3f} | '
                      f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

elif data=='DeepRSMA':
    for fold in range(10):
        os.makedirs('./Model/trained_weight/DeepRSMA_data', exist_ok=True)
        dti_train_loader, dti_val_loader = get_dataloader_deeprsma(data_folder_path,fold)
        print('Data load successfully')
        # get pre-trained compound model
        molebert_model = GNN_graphpred(num_layer=5, emb_dim=300, num_tasks=1, JK='last', drop_ratio=0.5,
                                       graph_pooling='mean', gnn_type='gin')
        molebert_model.from_pretrained(f'{molebert_path}/model_gin/Mole-BERT.pth')
        for name, param in molebert_model.named_parameters():
            param.requires_grad = False
        molebert_model = molebert_model.to(device)

        # get pre-trained RNA model
        rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
        for name, param in rna_fm_model.named_parameters():
            param.requires_grad = False
        rna_fm_model = rna_fm_model.to(device)
        print('pre-trained model load successfully')

        model = DeepRNA_DTI(rna_fm_model, molebert_model)
        model = model.to(device)
        for name, param in model.named_parameters():
            if 'rna_embedding' in name or 'compound_embedding' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_set = {'epoch': 0, 'val_aupr': 0}

        bs_epoch_val_auc = []
        bs_epoch_val_aupr = []

        total_dti_epoch_val_auc = []
        total_dti_epoch_val_aupr = []

        for epoch in range(num_epochs):
            model.train()

            bs_epoch_losses = []
            dti_epoch_losses = []
            total_losses = []

            for step, batch_dti in enumerate(dti_train_loader):
                optimizer.zero_grad()

                # dti train
                rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
                dti_pred = model(rna_input, compound_input, task='dti')
                dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
                dti_epoch_losses.append(dti_loss.item())

                total_loss = dti_loss

                total_losses.append(total_loss.item())
                total_loss.backward()
                optimizer.step()

            dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)

            if dti_val_aupr > best_set['val_aupr']:
                best_set['val_aupr'] = dti_val_aupr
                best_set['epoch'] = epoch
                torch.save(model.state_dict(), f'./Model/trained_weight/DeepRSMA_data/model_fold{fold}.pt')

            print(f'[Fold {fold}, Epoch {epoch}]'
                  f'Loss -  DTI:{np.mean(dti_epoch_losses):.3f} | '
                  f'Val(AUC/AUPR)- DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')