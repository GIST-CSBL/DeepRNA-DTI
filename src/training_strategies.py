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

import argparse

from src.model import DeepRNA_DTI
from src.data_utils import RNADataset, GraphDataset, InteractionDataset, CombinedDataLoader, CombinedDataLoader_rawdata
from src.utils import FocalLoss_per_sample, dti_evaluate, bs_evaluate, batch_processing



def select_training_strategies(fold,training_strategy, num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader, bs_loss_function, dti_loss_function, device):
    if training_strategy == 'raw_sequential_DTI_BS':
        raw_sequential_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader, bs_loss_function, dti_loss_function, device)
    if training_strategy == 'raw_sequential_BS_DTI':
        raw_sequential_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)
    if training_strategy == 'raw_alternating_BS_DTI':
        raw_alternating_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)
    if training_strategy == 'raw_alternating_DTI_BS':
        raw_alternating_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)
    if training_strategy == 'raw_pooled':
        raw_pooled(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,combined_dataloader, bs_loss_function, dti_loss_function, device)


    if training_strategy == 'oversampled_sequential_DTI_BS':
        oversampled_sequential_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)
    if training_strategy == 'oversampled_sequential_BS_DTI':
        oversampled_sequential_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)
    if training_strategy == 'oversampled_alternating_BS_DTI':
        oversampled_alternating_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)
    if training_strategy == 'oversampled_alternating_DTI_BS':
        oversampled_alternating_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)
    if training_strategy == 'oversampled_pooled':
        oversampled_pooled(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                              dti_val_loader,bs_loss_function, dti_loss_function, device)


def raw_sequential_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader, bs_loss_function, dti_loss_function,device='cuda', model_folder_path='./Model/trained_weight/raw_sequential_DTI_BS'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}
    dti_epoch = int(num_epochs/2)
    bs_epoch = int(num_epochs/2)


    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    # dti first
    for epoch in range(dti_epoch):
        model.train()
        dti_epoch_losses = []
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)
        dti_epoch_aucs.append(dti_val_auc);
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc);
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    # bs second
    for epoch in range(bs_epoch):
        model.train()
        bs_epoch_losses = []
        for step, batch_bs in enumerate(bs_train_loader):
            optimizer.zero_grad()
            # bs train
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
            bs_pred = model(rna_input, compound_input, task='bs')
            bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
            bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
            bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

            bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
            bs_epoch_losses.append(bs_loss.item())
            bs_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': bs_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f}, | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')

def raw_sequential_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,bs_loss_function, dti_loss_function, device='cuda', model_folder_path='./Model/trained_weight/raw_sequential_BS_DTI',):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}

    # sequential training
    dti_epoch = int(num_epochs/2)
    bs_epoch = int(num_epochs/2)

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    # bs first
    for epoch in range(bs_epoch):
        model.train()
        bs_epoch_losses = []
        for step, batch_bs in enumerate(bs_train_loader):
            optimizer.zero_grad()
            # bs train
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
            bs_pred = model(rna_input, compound_input, task='bs')
            bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
            bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
            bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

            bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
            bs_epoch_losses.append(bs_loss.item())
            bs_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': bs_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f}, | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    # dti second
    for epoch in range(dti_epoch):
        model.train()
        dti_epoch_losses = []
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')

def raw_alternating_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,bs_loss_function, dti_loss_function, device='cuda', model_folder_path='./Model/trained_weight/raw_alternating_BS_DTI'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    for epoch in range(num_epochs):
        model.train()

        bs_epoch_losses = []
        dti_epoch_losses = []
        total_losses = []
        # bs train
        for step, batch_bs in enumerate(bs_train_loader):
            optimizer.zero_grad()

            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
            bs_pred = model(rna_input, compound_input, task='bs')
            bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
            bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
            bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

            bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
            bs_epoch_losses.append(bs_loss.item())
            bs_loss.backward()
            optimizer.step()
        #dti train
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f},  DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')
    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')

def raw_alternating_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,bs_loss_function, dti_loss_function, device='cuda',model_folder_path='./Model/trained_weight/raw_alternating_DTI_BS'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    for epoch in range(num_epochs):
        model.train()

        bs_epoch_losses = []
        dti_epoch_losses = []
        total_losses = []

        # DTI train
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        # BS train
        for step, batch_bs in enumerate(bs_train_loader):
            optimizer.zero_grad()
            # bs train
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
            bs_pred = model(rna_input, compound_input, task='bs')
            bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
            bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
            bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

            bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
            bs_epoch_losses.append(bs_loss.item())
            bs_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f},  DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')
    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')
def raw_pooled(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,combined_dataloader, bs_loss_function, dti_loss_function, device='cuda',model_folder_path='./Model/trained_weight/raw_pooled'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    for epoch in range(num_epochs):
        model.train()

        bs_epoch_losses = []
        dti_epoch_losses = []
        total_losses = []

        for step, (batch_bs, batch_dti) in enumerate(combined_dataloader):
            optimizer.zero_grad()

            if batch_bs is not None:
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

            if batch_bs is not None:
                total_loss = bs_loss + dti_loss
            else:
                total_loss = dti_loss

            total_losses.append(total_loss.item())
            total_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f},  DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')
    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')





def oversampled_sequential_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,bs_loss_function, dti_loss_function, device='cuda',model_folder_path='./Model/trained_weight/oversampled_sequential_DTI_BS'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}


    # sequential training
    dti_epoch = int(num_epochs/2)
    bs_epoch = int(2*num_epochs)

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    # dti first
    for epoch in range(dti_epoch):
        model.train()
        dti_epoch_losses = []
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    # bs second
    for epoch in range(bs_epoch):
        model.train()
        bs_epoch_losses = []
        for step, batch_bs in enumerate(bs_train_loader):
            optimizer.zero_grad()
            # bs train
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
            bs_pred = model(rna_input, compound_input, task='bs')
            bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
            bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
            bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

            bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
            bs_epoch_losses.append(bs_loss.item())
            bs_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': bs_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f}, | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')

def oversampled_sequential_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,bs_loss_function, dti_loss_function, device='cuda',model_folder_path='./Model/trained_weight/oversampled_sequential_BS_DTI'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}

    # sequential training
    dti_epoch = int(num_epochs/2)
    bs_epoch = int(2*num_epochs)

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    # bs first
    for epoch in range(bs_epoch):
        model.train()
        bs_epoch_losses = []
        for step, batch_bs in enumerate(bs_train_loader):
            optimizer.zero_grad()
            # bs train
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
            bs_pred = model(rna_input, compound_input, task='bs')
            bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
            bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
            bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

            bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
            bs_epoch_losses.append(bs_loss.item())
            bs_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': bs_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f}, | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    # dti second
    for epoch in range(dti_epoch):
        model.train()
        dti_epoch_losses = []
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')

    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')


def oversampled_alternating_BS_DTI(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,bs_loss_function, dti_loss_function, device='cuda',model_folder_path='./Model/trained_weight/oversampled_alternating_BS_DTI'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    for epoch in range(num_epochs):
        model.train()

        bs_epoch_losses = []
        dti_epoch_losses = []
        total_losses = []

        # BS train
        for k in range(5):
            for step, batch_bs in enumerate(bs_train_loader):
                optimizer.zero_grad()
                # bs train
                rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
                bs_pred = model(rna_input, compound_input, task='bs')
                bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
                bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
                bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

                bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
                bs_epoch_losses.append(bs_loss.item())
                bs_loss.backward()
                optimizer.step()

        # DTI train
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f},  DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')
    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')

def oversampled_alternating_DTI_BS(fold,num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
                     dti_val_loader,bs_loss_function, dti_loss_function, device='cuda',model_folder_path='./Model/trained_weight/oversampled_sequential_DTI_BS'):
    os.makedirs(model_folder_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_set = {'epoch': 0, 'val_aupr': 0}

    dti_epoch_aucs = []
    dti_epoch_auprs = []
    bs_epoch_aucs = []
    bs_epoch_auprs = []
    dti_losses = []
    bs_losses = []

    for epoch in range(num_epochs):
        model.train()

        bs_epoch_losses = []
        dti_epoch_losses = []
        total_losses = []

        # DTI train
        for step, batch_dti in enumerate(dti_train_loader):
            optimizer.zero_grad()
            rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
            dti_pred = model(rna_input, compound_input, task='dti')
            dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
            dti_epoch_losses.append(dti_loss.item())
            dti_loss.backward()
            optimizer.step()

        for k in range(5):
            # BS train
            for step, batch_bs in enumerate(bs_train_loader):
                optimizer.zero_grad()
                rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
                bs_pred = model(rna_input, compound_input, task='bs')
                bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
                bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
                bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

                bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
                bs_epoch_losses.append(bs_loss.item())
                bs_loss.backward()
                optimizer.step()

        dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
        bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = bs_evaluate(model, bs_val_loader,
                                                                                               device)

        dti_epoch_aucs.append(dti_val_auc)
        dti_epoch_auprs.append(dti_val_aupr)
        bs_epoch_aucs.append(bs_val_micro_auc)
        bs_epoch_auprs.append(bs_val_micro_aupr)
        dti_losses.append(np.mean(dti_epoch_losses))
        bs_losses.append(np.mean(bs_epoch_losses))

        if dti_val_aupr > best_set['val_aupr']:
            best_set['val_aupr'] = dti_val_aupr
            best_set['epoch'] = epoch
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': dti_epoch_losses,
                'dti_auc': dti_epoch_aucs,
                'dti_aupr': dti_epoch_auprs,
                'bs_auc': bs_epoch_aucs,
                'bs_aupr': bs_epoch_auprs
            }
            torch.save(save_dict, f'{model_folder_path}/model_fold{fold}.pt')

        print(f'[Fold {fold}, Epoch {epoch}]'
              f'Loss - BS:{np.mean(bs_epoch_losses):.3f},  DTI:{np.mean(dti_epoch_losses):.3f} | '
              f'Val(AUC/AUPR)- BS :({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), DTI:({dti_val_auc:.3f}/{dti_val_aupr:.3f}), ')
    save_dict = {
        'epoch': epoch,
        'dti_loss': dti_losses,
        'bs_loss': bs_losses,
        'dti_auc': dti_epoch_aucs,
        'dti_aupr': dti_epoch_auprs,
        'bs_auc': bs_epoch_aucs,
        'bs_aupr': bs_epoch_auprs
    }

    torch.save(save_dict, f'{model_folder_path}/model_fold{fold}_final_epoch.pt')

# def oversampled_pooled(num_epochs, model, bs_train_loader, bs_val_loader, dti_train_loader,
#                      dti_val_loader,model_folder_path='./Model/trained_weight/oversampled_pooled'):
#     os.makedirs(model_folder_path, exist_ok=True)
