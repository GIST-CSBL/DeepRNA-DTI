import os

"""Setup CUDA device."""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)


import sys
import argparse
import numpy as np
import torch
from torch_geometric.data import DataLoader
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

from src.utils import dti_evaluate, bs_evaluate
from src.model import DeepRNA_DTI
from src.data_utils import RNADataset, GraphDataset, InteractionDataset


def parse_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description='DeepRNA-DTI test')
    parser.add_argument('--test_type', type=str, default='unseen_pair',
                        help='Dataset type: unseen_pair / unseen_rna / unseen_compound')
    parser.add_argument('--model_folder_path', type=str, default='./Model/trained_weight',
                        help='Path to trained model weights')
    parser.add_argument('--data_folder_path', type=str, default='./Dataset/',
                        help='Root path to dataset')
    parser.add_argument('--molebert_path', type=str, default='./Model/pretrained_model/Mole-BERT',
                        help='Path to Mole-BERT pretrained weights')
    parser.add_argument('--rnafm_path', type=str, default='./Model/pretrained_model/RNA-FM',
                        help='Path to RNA-FM pretrained weights')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
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
    molebert_model.eval()

    # Load RNA-FM
    sys.path.append(rnafm_path)
    import fm  
    rna_fm_model, _ = fm.pretrained.rna_fm_t12()
    rna_fm_model.eval()

    return molebert_model, rna_fm_model


def get_test_loaders(data_folder_path, test_type, batch_size):
    """Create DTI/BS test dataloaders."""
    dti_test_path = f'{data_folder_path}/{test_type}/dti_data/test_fold'
    bs_test_path = f'{data_folder_path}/{test_type}/bs_data/test_fold'

    dti_test_dataset = InteractionDataset(
        dti_test_path, GraphDataset(dti_test_path), RNADataset(dti_test_path)
    )
    bs_test_dataset = InteractionDataset(
        bs_test_path, GraphDataset(bs_test_path), RNADataset(bs_test_path)
    )

    dti_test_loader = DataLoader(dti_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    bs_test_loader = DataLoader(bs_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return dti_test_loader, bs_test_loader



def main():
    args = parse_args()


    dti_test_loader, bs_test_loader = get_test_loaders(
        args.data_folder_path, args.test_type, args.batch_size
    )
    print('Data loaded successfully')

    molebert_model, rna_fm_model = load_pretrained_models(args.molebert_path, args.rnafm_path)
    molebert_model = molebert_model.to(device)
    rna_fm_model = rna_fm_model.to(device)
    print('Pre-trained models loaded successfully')

    dti_aucs, dti_auprs = [], []
    bs_micro_aucs, bs_micro_auprs = [], []

    for fold in range(5):
        model = DeepRNA_DTI(rna_fm_model, molebert_model).to(device)
        state = torch.load(
            f'{args.model_folder_path}/{args.test_type}/model_fold{fold}.pt',
            map_location=device
        )
        model.load_state_dict(state)
        model.eval()

        val_auc, val_aupr = dti_evaluate(model, dti_test_loader, device)
        dti_aucs.append(val_auc)
        dti_auprs.append(val_aupr)

        _, _, bs_micro_auc, bs_micro_aupr = bs_evaluate(
            model, bs_test_loader, device
        )
        bs_micro_aucs.append(bs_micro_auc)
        bs_micro_auprs.append(bs_micro_aupr)

    print('DTI AUC : ', round(np.mean(dti_aucs), 3))
    print('DTI AUPR : ', round(np.mean(dti_auprs), 3))
    print('BS per sample AUC : ', round(np.mean(bs_micro_aucs), 3))
    print('BS per sample AUPR : ', round(np.mean(bs_micro_auprs), 3))



if __name__ == '__main__':
    main()