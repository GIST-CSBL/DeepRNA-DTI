import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
import warnings
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, roc_curve
from rdkit import Chem  
import dgl
from dgllife.utils import (
    BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer,
    atom_degree_one_hot, atom_implicit_valence_one_hot,
    atom_is_aromatic_one_hot, atom_total_num_H_one_hot, atom_type_one_hot,
    bond_is_conjugated_one_hot, bond_is_in_ring_one_hot,
    bond_stereo_one_hot, bond_type_one_hot, smiles_to_bigraph,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import MaxPooling
import dgl.backend as F

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'
    
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------
# Featurisers and data processing
# ----------------------------------------------------------------------------
MAX_RNA_LEN = 200
MAX_SMI_LEN = 100

RNA_VOC = "AUGC"
RNA_DICT = {v: (i + 1) for i, v in enumerate(RNA_VOC)}

SMI_DICT = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
}

def seq_cat(seq: str, max_len: int = MAX_RNA_LEN) -> np.ndarray:
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq[:max_len]):
        x[i] = RNA_DICT[ch]
    return x
def smi_cat(smi: str, max_len: int = MAX_SMI_LEN) -> np.ndarray:
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(smi[:max_len]):
        x[i] = SMI_DICT[ch]
    return x
def _canonicalize_smiles(smiles_iter) -> List[str]:
    from rdkit import Chem

    out = []
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi)
        out.append(Chem.MolToSmiles(mol, canonical=True) if mol is not None else smi)
    return out
def process_split(model: str, unseen_type: str, dataset_root: str):
    """Build DeepDTA and GraphDTA pickles for all 5 folds of one unseen split."""
    path = f"../data_{model}"
    os.makedirs(path, exist_ok=True)
    
    def _load(split: str, fold: int):
        if split == "test":
            csv = f"{dataset_root}/unseen_{unseen_type}/dti_data/test_fold/raw/interactions.csv"
        else:
            csv = f"{dataset_root}/unseen_{unseen_type}/dti_data/{split}_fold{fold}/raw/interactions.csv"
        return pd.read_csv(csv)
        
    if model=='DeepDTA':
        for fold in range(5):
            for split in ("train", "val", "test"):
                df = _load(split, fold)
                df["can_smi"] = _canonicalize_smiles(df["smiles"])
    
                smi_tok = [smi_cat(s) for s in df["can_smi"]]
                seq_tok = [seq_cat(p) for p in df["sequence"]]
                seq_lens = [len(p) for p in df["sequence"]]
                ys = list(df["interactions"])
                
                deepdta = list(zip(smi_tok, seq_tok, ys, seq_lens))
                if split == "test":
                    with open(f"{path}/unseen_{unseen_type}_test.pickle", "wb") as fw:
                        pickle.dump(deepdta, fw)
                else:
                    with open(f"{path}/unseen_{unseen_type}_{split}_fold{fold}.pickle", "wb") as fw:
                        pickle.dump(deepdta, fw)
                    
    else:
        for fold in range(5):
            for split in ("train", "val", "test"):
                df = _load(split, fold)
                df["can_smi"] = _canonicalize_smiles(df["smiles"])
    
                smi_tok = [smi_cat(s) for s in df["can_smi"]]
                seq_tok = [seq_cat(p) for p in df["sequence"]]
                seq_lens = [len(p) for p in df["sequence"]]
                ys = list(df["interactions"])
                
                atom_featurizer = BaseAtomFeaturizer(
                    featurizer_funcs={"h": ConcatFeaturizer([
                        atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot,
                        atom_implicit_valence_one_hot, atom_is_aromatic_one_hot,
                    ])}
                )
                bond_featurizer = BaseBondFeaturizer({"e": ConcatFeaturizer([
                    bond_type_one_hot, bond_is_conjugated_one_hot,
                    bond_is_in_ring_one_hot, bond_stereo_one_hot,
                ])})
        
                graphs = [
                    dgl.add_self_loop(smiles_to_bigraph(
                        s, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer
                    ))
                    for s in df["can_smi"]
                ]
                graph_pkl = list(zip(graphs, seq_tok, ys, seq_lens))
                if split == "test":
                    with open(f"{path}/unseen_{unseen_type}_test.pickle", "wb") as fw:
                        pickle.dump(graph_pkl, fw)
                else:
                    with open(f"{path}/unseen_{unseen_type}_{split}_fold{fold}.pickle", "wb") as fw:
                        pickle.dump(graph_pkl, fw)
       


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def _build_models():
    class DeepDTA(nn.Module):
        def __init__(self, embed_dim=128, n_filters=32, dropout=0.1, num_features_xt=4):
            super().__init__()
            self.embedding_xd = nn.Embedding(64 + 1, embed_dim, padding_idx=0)
            self.conv_xd_1 = nn.Conv1d(embed_dim, n_filters, kernel_size=4)
            self.conv_xd_2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=4)
            self.conv_xd_3 = nn.Conv1d(n_filters * 2, 128, kernel_size=4)

            self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0)
            self.conv_xt_1 = nn.Conv1d(embed_dim, n_filters, kernel_size=8)
            self.conv_xt_2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=8)
            self.conv_xt_3 = nn.Conv1d(n_filters * 2, 128, kernel_size=8)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 64)
            self.out = nn.Linear(64, 1)
            self.gmp = nn.AdaptiveMaxPool1d(1)

        def forward(self, drug, target):
            d = self.embedding_xd(drug).permute(0, 2, 1)
            d = self.relu(self.conv_xd_1(d))
            d = self.relu(self.conv_xd_2(d))
            d = self.relu(self.conv_xd_3(d))
            d = self.gmp(d).squeeze(-1)

            t = self.embedding_xt(target).permute(0, 2, 1)
            t = self.relu(self.conv_xt_1(t))
            t = self.relu(self.conv_xt_2(t))
            t = self.relu(self.conv_xt_3(t))
            t = self.gmp(t).squeeze(-1)

            x = torch.cat([t, d], dim=1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            return self.out(x)

    class GraphDTA(nn.Module):
        def __init__(self, n_filters=32, embed_dim=128, num_features_xd=68,num_features_xt=25,
                     output_dim=128, dropout=0.1, k_size_xt=8):
            super().__init__()
            self.conv1 = GraphConv(num_features_xd, embed_dim)
            self.conv2 = GraphConv(embed_dim, embed_dim * 2)
            self.conv3 = GraphConv(embed_dim * 2, output_dim)
            self.gmp_graph = MaxPooling()

            self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0)
            self.conv_xt_1 = nn.Conv1d(embed_dim, n_filters, kernel_size=k_size_xt)
            self.conv_xt_2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=k_size_xt)
            self.conv_xt_3 = nn.Conv1d(n_filters * 2, output_dim, kernel_size=k_size_xt)
            self.gmp_seq = nn.AdaptiveMaxPool1d(1)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(2 * output_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.out = nn.Linear(64, 1)
            

        def forward(self, graph, atom_feats, target, seq_lens, device):
            h = self.relu(self.conv1(graph, atom_feats))
            h = self.relu(self.conv2(graph, h))
            h = self.relu(self.conv3(graph, h))
            h = self.gmp_graph(graph, h)

            t = self.embedding_xt(target).permute(0, 2, 1)
            t = self.relu(self.conv_xt_1(t))
            t = self.relu(self.conv_xt_2(t))
            t = self.relu(self.conv_xt_3(t))
            t = self.gmp_seq(t).squeeze(-1)

            x = torch.cat([t, h], dim=1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            return self.out(x)

    class GraphATT_DTA(nn.Module):
        def __init__(self, n_filters=32, embed_dim=128, num_features_xd=68,num_features_xt=25,
                     output_dim=128, dropout=0.1, k_size_xt=8, infinity=-5.0e10):
            super().__init__()
            self.inf = infinity
            self.conv1 = GraphConv(num_features_xd, embed_dim)
            self.conv2 = GraphConv(embed_dim, embed_dim * 2)
            self.conv3 = GraphConv(embed_dim * 2, output_dim)

            self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0)
            self.conv_xt_1 = nn.Conv1d(embed_dim, n_filters, kernel_size=k_size_xt)
            self.conv_xt_2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=k_size_xt)
            self.conv_xt_3 = nn.Conv1d(n_filters * 2, output_dim, kernel_size=k_size_xt)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(2 * output_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.out = nn.Linear(64, 1)
            self.k_size_xt = k_size_xt

        def _graph_in_batch(self, batch_num_objs, h):
            return F.pad_packed_tensor(h, batch_num_objs, 0)

        def _padding_zero_to_inf(self, sim, batch_num_objs, seq_lens, device):
            atom_mask = (
                torch.arange(sim.size(1), device=device).expand(len(batch_num_objs), -1)
                >= batch_num_objs.unsqueeze(1)
            )

            eff_seq_lens = torch.clamp(seq_lens, max=MAX_RNA_LEN) - 3 * (self.k_size_xt - 1)
            eff_seq_lens = torch.clamp(eff_seq_lens, min=1)
            seq_mask = (
                torch.arange(sim.size(2), device=device).expand(len(eff_seq_lens), -1)
                >= eff_seq_lens.unsqueeze(1)
            )
            sim.masked_fill_(atom_mask.unsqueeze(2), self.inf)
            sim.masked_fill_(seq_mask.unsqueeze(1), self.inf)
            return sim

        def _attention(self, xd, xt, batch_num_objs, seq_lens, device):
            sim = torch.bmm(xd, xt)
            sim = self._padding_zero_to_inf(sim, batch_num_objs, seq_lens, device)
            s_a = nn.Softmax(dim=1)(sim).permute(0, 2, 1)
            a_s = nn.Softmax(dim=-1)(sim)
            f1 = torch.sum(torch.bmm(s_a, xd), 1)
            f2 = torch.sum(torch.bmm(a_s, xt.permute(0, 2, 1)), 1)
            return torch.cat([f1, f2], 1)

        def forward(self, graph, atom_feats, target, seq_lens, device):
            batch_num_objs = graph.batch_num_nodes()
            h = self.relu(self.conv1(graph, atom_feats))
            h = self.relu(self.conv2(graph, h))
            h = self.relu(self.conv3(graph, h))
            hs = self._graph_in_batch(batch_num_objs, h)

            t = self.embedding_xt(target).permute(0, 2, 1)
            t = self.relu(self.conv_xt_1(t))
            t = self.relu(self.conv_xt_2(t))
            t = self.relu(self.conv_xt_3(t))

            att = self._attention(hs, t, batch_num_objs, seq_lens, device)
            x = self.dropout(self.relu(self.fc1(att)))
            x = self.dropout(self.relu(self.fc2(x)))
            return self.out(x)

    return DeepDTA, GraphDTA, GraphATT_DTA
# ----------------------------------------------------------------------------
# Collate functions
# ----------------------------------------------------------------------------
def _collate_seq(sample):
    drugs, proteins, labels, seq_lens = map(list, zip(*sample))
    return (torch.tensor(drugs), torch.tensor(proteins),
            torch.tensor(labels), torch.tensor(seq_lens))
def _collate_graph(sample):
    graphs, proteins, labels, seq_lens = map(list, zip(*sample))
    return (dgl.batch(graphs), torch.tensor(proteins),
            torch.tensor(labels), torch.tensor(seq_lens))
# ----------------------------------------------------------------------------
# Train / eval loop
# ----------------------------------------------------------------------------

def _forward(model_name, model, batch, device):
    drugs, targets, labels, seq_lens = batch
    targets = targets.to(device)
    labels = labels.to(device)
    seq_lens = seq_lens.to(device)
    if model_name == "DeepDTA":
        drugs = drugs.to(device)
        out = model(drugs, targets)
    else:
        atom_feats = drugs.ndata["h"].to(device)
        drugs = drugs.to(device)
        out = model(drugs, atom_feats, targets, seq_lens, device)
    return out, labels
def evaluation(model_name, model, loader, device):
    import torch
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            out, labels = _forward(model_name, model, batch, device)
            y_true.append(labels.view(out.shape).float())
            y_pred.append(out)
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    return auc(fpr, tpr), average_precision_score(y_true, y_pred)
def _build_model(model_name, models_tuple):
    DeepDTA, GraphDTA, GraphATT_DTA = models_tuple
    if model_name == "DeepDTA":
        return DeepDTA()
    if model_name == "GraphDTA":
        return GraphDTA()
    if model_name == "GraphATT_DTA":
        return GraphATT_DTA()
    raise ValueError(model_name)
def _paths_for(model_name):
    """Return (data_dir, model_dir, collate_fn_picker)."""
    if model_name == "DeepDTA":
        return "data_DeepDTA", "model_DeepDTA", _collate_seq
    if model_name == "GraphDTA":
        return "data_GraphDTA", "model_GraphDTA", _collate_graph
    if model_name == "GraphATT_DTA":
        return "data_GraphDTA", "model_GraphATT_DTA", _collate_graph
    raise ValueError(model_name)
def train(model_name, unseen_type, args, models_tuple, device):
    data_dir, model_dir, collate_fn = _paths_for(model_name)
    os.makedirs(model_dir, exist_ok=True)
    if model_name =='DeepDTA':
        lr=0.001
        patience=100 #DeepDTA does not have early stopping
    else:
        lr=0.0001
        patience=30

    for fold in range(5):
        with open(f"{data_dir}/unseen_{unseen_type}_train_fold{fold}.pickle", "rb") as fr:
            train_data = pickle.load(fr)
        with open(f"{data_dir}/unseen_{unseen_type}_val_fold{fold}.pickle", "rb") as fr:
            val_data = pickle.load(fr)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=False)

        model = _build_model(model_name, models_tuple).to(device)
        model_path = f"{model_dir}/unseen_{unseen_type}_model_fold{fold}"

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        best_auc = 0.0
        counter = 0
        avg_train_losses, avg_val_aucs = [], []

        for epoch in range(100):
            model.train()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                out, labels = _forward(model_name, model, batch, device)
                loss = loss_fn(out, labels.view(-1, 1).float())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            train_loss = float(np.mean(losses))
            val_auc, val_aupr = evaluation(model_name, model, val_loader, device)
            avg_train_losses.append(train_loss)
            avg_val_aucs.append(val_auc)

            print(f"[{model_name}|unseen_{unseen_type}|fold{fold}] "
                  f"epoch {epoch}/{100}  train_loss={train_loss:.5f}  "
                  f"val_auc={val_auc:.5f}  val_aupr={val_aupr:.5f}")

            if val_auc > best_auc:
                print(f"  best AUC: {best_auc:.5f} -> {val_auc:.5f}; saving")
                best_auc = val_auc
                counter = 0
                torch.save({
                    "Epoch": epoch,
                    "State_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_aupr": val_aupr,
                    "avg_train_losses": avg_train_losses,
                    "avg_val_aucs": avg_val_aucs,
                }, model_path)
            else:
                counter += 1
                print(f"  EarlyStop counter: {counter}/{patience}")
                if counter > patience:
                    print("  early stopping")
                    break
def test(model_name, unseen_type, args, models_tuple, device):
    data_dir, model_dir, collate_fn = _paths_for(model_name)
    with open(f"{data_dir}/unseen_{unseen_type}_test.pickle", "rb") as fr:
        test_data = pickle.load(fr)

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=False)

    aucs, auprs = [], []
    for fold in range(5):
        model_path = f"{model_dir}/unseen_{unseen_type}_model_fold{fold}"
        info = torch.load(model_path, map_location=device)
        model = _build_model(model_name, models_tuple).to(device)
        model.load_state_dict(info["State_dict"])

        auc_val, aupr_val = evaluation(model_name, model, test_loader, device)
        print(f"[{model_name}|{unseen_type}|fold{fold}] test AUC={auc_val:.4f}  AUPR={aupr_val:.4f}")
        aucs.append(auc_val)
        auprs.append(aupr_val)

    if aucs:
        print(f"\n===== {model_name} / unseen_{unseen_type} (n={len(aucs)} folds) =====")
        print(f"DTI AUC  : {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
        print(f"DTI AUPR : {np.mean(auprs):.3f} +/- {np.std(auprs):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", required=True, choices=["DeepDTA", "GraphDTA", "GraphATT_DTA"])
    parser.add_argument("--unseen-type", required=True, choices=["compound", "rna", "pair", "both"])
    parser.add_argument("--dataset-root", default="../Dataset")
    args = parser.parse_args()

    print(f"Data processing for {args.model}, {args.unseen_type}")
    process_split(args.model, args.unseen_type, args.dataset_root)
    print(f"Build model for {args.model}, {args.unseen_type}")
    models_tuple = _build_models()
    train(args.model, args.unseen_type, args, models_tuple, device)
    test(args.model, args.unseen_type, args, models_tuple, device)


if __name__ == "__main__":
    main()
