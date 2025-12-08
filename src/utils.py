import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    roc_auc_score,
    roc_curve
)

def batch_processing(batch, device):
    """
    Process batch data and move to device.
    
    Args:
        batch: Tuple containing (graph_data, rna_tokens, bs_labels, rna_seq_len, dti_labels)
        device: Target device (cuda or cpu)
        
    Returns:
        rna_input: RNA sequence tokens [B, L_rna]
        compound_input: Tuple of (x, edge_index, edge_attr, batch) for compound graph
        bs_label: Binding site labels [B, L_rna, 1]
        dti_label: Drug-target interaction labels [B]
    """
    graph_data = batch[0].to(device)
    compound_input = (
        graph_data.x,
        graph_data.edge_index,
        graph_data.edge_attr,
        graph_data.batch
    )
    rna_input = batch[1].to(device)
    bs_label = batch[2].unsqueeze(2).to(device)
    dti_label = batch[4].to(device)

    return rna_input, compound_input, bs_label, dti_label



def dti_evaluate(model, loader, device):
    """
    Evaluate model performance on drug-target interaction (DTI) prediction task.
    
    Args:
        model: Trained DeepRNA_DTI model
        loader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        dti_auc: Area Under ROC Curve for DTI prediction
        dti_aupr: Area Under Precision-Recall Curve for DTI prediction
    """
    model.eval()

    y_true = []
    y_pred = []

    for batch in loader:
        rna_input, compound_input, _, dti_label = batch_processing(batch, device)
        
        with torch.no_grad():
            dti_pred = model(rna_input, compound_input, task='dti')

        y_true.append(dti_label.view(dti_pred.shape).to(torch.float32))
        y_pred.append(dti_pred)

    # Concatenate all predictions
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    dti_auc = auc(fpr, tpr)
    dti_aupr = average_precision_score(y_true, y_pred)

    return dti_auc, dti_aupr


def bs_evaluate(model, loader, device):
    """
    Evaluate model performance on binding site (BS) prediction task.
    
    Computes both macro-averaged (over all positions) and micro-averaged 
    (per-RNA sample) metrics.
    
    Args:
        model: Trained DeepRNA_DTI model
        loader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        macro_auc: Macro-averaged AUC (over all binding site positions)
        macro_aupr: Macro-averaged AUPR (over all binding site positions)
        micro_auc: Micro-averaged AUC (per-RNA sample average)
        micro_aupr: Micro-averaged AUPR (per-RNA sample average)
    """
    model.eval()

    bs_labels = []
    bs_preds = []
    rna_per_aucs = []
    rna_per_auprs = []
    
    for batch in loader:
        rna_input, compound_input, bs_label, dti_label = batch_processing(batch, device)

        with torch.no_grad():
            bs_pred = model(rna_input, compound_input, task='bs')
        
        # Create masks
        # Mask for actual sequence positions (exclude special tokens)
        bs_label_mask_true_seq = ((rna_input >= 4) & (rna_input <= 7)).unsqueeze(-1)
        
        # Mask for positive DTI (only evaluate binding sites for positive DTI pairs)
        bs_label_mask_dti = dti_label.unsqueeze(1).unsqueeze(2).expand(
            -1, bs_label.shape[1], 1
        )
        
        # Apply masks: only evaluate on actual sequence positions with positive DTI
        bs_pred = bs_pred * bs_label_mask_true_seq * bs_label_mask_dti

        # Collect predictions and labels for macro-averaged metrics
        filtered_predictions = bs_pred[bs_label_mask_true_seq].squeeze(-1)
        filtered_labels = bs_label[bs_label_mask_true_seq].squeeze(-1)

        bs_labels.append(filtered_labels)
        bs_preds.append(filtered_predictions)

        # Compute per-RNA sample metrics (micro-averaged)
        for sample_idx in range(bs_pred.size(0)):
            predictions = bs_pred[sample_idx].squeeze(-1)
            labels = bs_label[sample_idx].squeeze(-1)
            non_special_mask = bs_label_mask_true_seq[sample_idx].squeeze(-1)
            
            filtered_predictions = predictions[non_special_mask]
            filtered_labels = labels[non_special_mask]

            # Skip samples with no positive binding sites (negative DTI)
            if filtered_labels.sum() > 0:
                auc_score = roc_auc_score(
                    filtered_labels.cpu().numpy(),
                    filtered_predictions.cpu().numpy()
                )
                aupr_score = average_precision_score(
                    filtered_labels.cpu().numpy(),
                    filtered_predictions.cpu().numpy()
                )
                rna_per_aucs.append(auc_score)
                rna_per_auprs.append(aupr_score)

    # Compute macro-averaged metrics
    bs_labels = torch.cat(bs_labels, dim=0).view(-1, 1).cpu().numpy()
    bs_preds = torch.cat(bs_preds, dim=0).view(-1, 1).cpu().numpy()

    macro_auc = roc_auc_score(bs_labels, bs_preds)
    macro_aupr = average_precision_score(bs_labels, bs_preds)

    # Compute micro-averaged metrics (mean of per-RNA metrics)
    micro_auc = np.mean(rna_per_aucs) if rna_per_aucs else 0.0
    micro_aupr = np.mean(rna_per_auprs) if rna_per_auprs else 0.0

    return macro_auc, macro_aupr, micro_auc, micro_aupr


class FocalLoss_per_sample(nn.Module):
    """
    Focal Loss with per-sample normalization.   
    """
    
    def __init__(self, alpha=0.25, gamma=2):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super(FocalLoss_per_sample, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-10

    def forward(self, inputs, targets):
        """
        Compute focal loss with per-sample normalization.
        
        Args:
            inputs: Predicted probabilities [B, L, 1]
            targets: Ground truth labels [B, L, 1]
            
        Returns:
            loss: Mean focal loss across samples
        """
        # Compute binary cross-entropy loss for each position
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        
        # Compute focal loss: alpha * (1 - pt)^gamma * BCE
        pt = torch.exp(-bce_loss)  # pt is the probability of true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        
        # Sum loss over sequence positions
        focal_loss_sum = focal_loss.sum(dim=1)  # [B, 1]
        
        # Normalize by number of positive labels per sample
        seq_len = targets.sum(dim=1)  # [B, 1] - number of positive labels
        focal_loss_sample = focal_loss_sum / (seq_len + self.epsilon)  # [B, 1]

        return focal_loss_sample.mean()

