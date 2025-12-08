import os

"""Setup CUDA device."""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

from src.utils import FocalLoss_per_sample, dti_evaluate, bs_evaluate, batch_processing
from src.model import DeepRNA_DTI
from src.data_utils import RNADataset, GraphDataset, InteractionDataset, CombinedDataLoader


def parse_args():
    """Parse command line arguments for training"""
    parser = argparse.ArgumentParser(description='Train DeepRNA-DTI train')
    parser.add_argument('--train_type', type=str, default='unseen_pair',
                        help='Type of training: unseen_pair, unseen_rna, unseen_compound, unseen_both')
    parser.add_argument('--model_folder_path', type=str, default='./Model/trained_weight',
                        help='Path to save trained models')
    parser.add_argument('--data_folder_path', type=str, default='./Dataset/',
                        help='Path to dataset folder')
    parser.add_argument('--molebert_path', type=str, default='./Model/pretrained_model/Mole-BERT',
                        help='Path to Mole-BERT pretrained model')
    parser.add_argument('--rnafm_path', type=str, default='./Model/pretrained_model/RNA-FM',
                        help='Path to RNA-FM pretrained model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--init', type=bool, default=True,
                        help='Whether to initialize weights')
    parser.add_argument('--init_method', type=str, default='he',
                        help='Weight initialization method: xavier, he ')
    return parser.parse_args()


def load_pretrained_models(molebert_path, rnafm_path):
    """Load pretrained Mole-BERT and RNA-FM models."""
    # Load Mole-BERT
    sys.path.append(molebert_path)
    from model import GNN_graphpred
    
    molebert_model = GNN_graphpred(
        num_layer=5, 
        emb_dim=300, 
        num_tasks=1, 
        JK='last', 
        drop_ratio=0.5,
        graph_pooling='mean', 
        gnn_type='gin'
    )
    molebert_model.from_pretrained(f'{molebert_path}/model_gin/Mole-BERT.pth')
    for param in molebert_model.parameters():
        param.requires_grad = False
    molebert_model.eval()
    
    # Load RNA-FM
    sys.path.append(rnafm_path)
    import fm
    
    rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
    for param in rna_fm_model.parameters():
        param.requires_grad = False
    rna_fm_model.eval()
    
    return molebert_model, rna_fm_model


def get_dataloader(data_folder_path, train_type, fold, batch_size):
    """Create dataloaders for binding site (BS) and drug-target interaction (DTI) tasks."""
    # Binding site data
    bs_train_path = f'{data_folder_path}/{train_type}/bs_data/train_fold{fold}'
    bs_val_path = f'{data_folder_path}/{train_type}/bs_data/val_fold{fold}'
    
    bs_train_dataset = InteractionDataset(
        bs_train_path, 
        GraphDataset(bs_train_path), 
        RNADataset(bs_train_path)
    )
    bs_val_dataset = InteractionDataset(
        bs_val_path, 
        GraphDataset(bs_val_path), 
        RNADataset(bs_val_path)
    )
    
    # Drug-target interaction data
    dti_train_path = f'{data_folder_path}/{train_type}/dti_data/train_fold{fold}'
    dti_val_path = f'{data_folder_path}/{train_type}/dti_data/val_fold{fold}'
    
    dti_train_dataset = InteractionDataset(
        dti_train_path, 
        GraphDataset(dti_train_path), 
        RNADataset(dti_train_path)
    )
    dti_val_dataset = InteractionDataset(
        dti_val_path, 
        GraphDataset(dti_val_path), 
        RNADataset(dti_val_path)
    )
    
    # Create dataloaders
    bs_train_loader = DataLoader(bs_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bs_val_loader = DataLoader(bs_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    dti_train_loader = DataLoader(dti_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dti_val_loader = DataLoader(dti_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    combined_dataloader = CombinedDataLoader(bs_train_loader, dti_train_loader)
    
    return bs_train_loader, bs_val_loader, dti_train_loader, dti_val_loader, combined_dataloader


def train_epoch(model, combined_dataloader, bs_loss_function, dti_loss_function, optimizer, device):
    """Train model for one epoch."""
    model.train()
    
    bs_epoch_losses = []
    dti_epoch_losses = []
    
    for batch_bs, batch_dti in combined_dataloader:
        optimizer.zero_grad()
        
        # Binding site training
        rna_input, compound_input, bs_label, dti_label = batch_processing(batch_bs, device)
        bs_pred = model(rna_input, compound_input, task='bs')
        
        # Apply masks
        bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
        bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(-1, bs_label.shape[1], 1)
        bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti
        
        bs_loss = bs_loss_function(bs_pred.float(), bs_label.float())
        bs_epoch_losses.append(bs_loss.item())
        
        # Drug-target interaction training
        rna_input, compound_input, bs_label, dti_label = batch_processing(batch_dti, device)
        dti_pred = model(rna_input, compound_input, task='dti')
        dti_loss = dti_loss_function(dti_pred, dti_label.view(dti_pred.shape).to(torch.float32))
        dti_epoch_losses.append(dti_loss.item())
        
        # Backward pass
        total_loss = bs_loss + dti_loss
        total_loss.backward()
        optimizer.step()
    
    return np.mean(bs_epoch_losses), np.mean(dti_epoch_losses)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load pretrained models
    molebert_model, rna_fm_model = load_pretrained_models(args.molebert_path, args.rnafm_path)
    molebert_model = molebert_model.to(device)
    rna_fm_model = rna_fm_model.to(device)
    print('Pretrained models loaded successfully')
    
    # Loss functions
    bs_loss_function = FocalLoss_per_sample()
    dti_loss_function = nn.BCEWithLogitsLoss()
    
    # Training loop for each fold
    for fold in range(5):
        print(f'\n{"="*60}')
        print(f'Starting training for Fold {fold}')
        print(f'{"="*60}')
        os.makedirs(f'{args.model_folder_path}/{args.train_type}', exist_ok=True)
        # Load data
        bs_train_loader, bs_val_loader, dti_train_loader, dti_val_loader, combined_dataloader = \
            get_dataloader(args.data_folder_path, args.train_type, fold, args.batch_size)
        print('Data loaded successfully')
        
        # Initialize model
        model = DeepRNA_DTI(
            rna_fm_model, 
            molebert_model, 
            init=args.init,
            init_method=args.init_method
        )
        model = model.to(device)
        
        # Freeze embedding layers
        for name, param in model.named_parameters():
            if 'rna_embedding' in name or 'compound_embedding' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # Training tracking
        best_set = {'epoch': 0, 'val_auc': 0}
        
        # Training loop
        for epoch in range(args.num_epochs):
            # Train
            bs_train_loss, dti_train_loss = train_epoch(
                model, combined_dataloader, bs_loss_function, 
                dti_loss_function, optimizer, device
            )
            
            # Validation
            model.eval()
            dti_val_auc, dti_val_aupr = dti_evaluate(model, dti_val_loader, device)
            bs_val_macro_auc, bs_val_macro_aupr, bs_val_micro_auc, bs_val_micro_aupr = \
                bs_evaluate(model, bs_val_loader, device)
            
            # Save best model
            if dti_val_auc > best_set['val_auc']:
                best_set['val_auc'] = dti_val_auc
                best_set['epoch'] = epoch
                torch.save(
                    model.state_dict(), 
                    f'{args.model_folder_path}/{args.train_type}/model_fold{fold}.pt'
                )
            
            # Print progress
            print(f'[Fold {fold}, Epoch {epoch}] '
                  f'Loss - BS: {bs_train_loss:.3f}, DTI: {dti_train_loss:.3f} | '
                  f'Val(AUC/AUPR) - BS: ({bs_val_micro_auc:.3f}/{bs_val_micro_aupr:.3f}), '
                  f'DTI: ({dti_val_auc:.3f}/{dti_val_aupr:.3f})')
        
        print(f'\nBest model for Fold {fold} saved at epoch {best_set["epoch"]} '
              f'with AUC: {best_set["val_auc"]:.3f}')


if __name__ == '__main__':
    main()