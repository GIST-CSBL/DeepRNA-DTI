import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class DeepRNA_DTI(nn.Module):
    """DeepRNA-DTI model for drug-target interaction and binding site prediction."""
    
    def __init__(self, rna_model, compound_model, init=True):
        """
        Initialize DeepRNA-DTI model.
        
        Args:
            rna_model: Pretrained RNA-FM model for RNA embedding
            compound_model: Pretrained Mole-BERT model for compound embedding
            init: Whether to initialize weights (default: True)
        """
        super(DeepRNA_DTI, self).__init__()
        
        # RNA embedding layers
        self.rna_embedding = rna_model
        self.rna_linear = nn.Linear(640, 128)
        self.rna_linear2 = nn.Linear(128, 128)

        # Compound embedding layers
        self.compound_embedding = compound_model
        self.compound_linear = nn.Linear(300, 128)
        self.compound_linear2 = nn.Linear(128, 128)

        # Shared network
        self.shared_linear0 = nn.Linear(128, 64)
        self.shared_linear1 = nn.Linear(64, 32)

        # Attention weights
        self.att_weight1 = nn.Parameter(torch.randn(128, 8))
        self.att_weight2 = nn.Parameter(torch.randn(8, 128))

        # Binding site prediction layers
        self.bs_linear1 = nn.Linear(32, 16)
        self.bs_linear2 = nn.Linear(16, 1)

        # Drug-target interaction prediction layers
        self.dti_linear1 = nn.Linear(32, 16)
        self.dti_linear2 = nn.Linear(16, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        if init:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights.
        
        """
        
        nn.init.xavier_uniform_(self.att_weight1)
        nn.init.xavier_uniform_(self.att_weight2)
        
        linear_layers = [
            self.rna_linear, self.rna_linear2,
            self.compound_linear, self.compound_linear2,
            self.shared_linear0, self.shared_linear1,
            self.bs_linear1, self.bs_linear2,
            self.dti_linear1, self.dti_linear2
        ]
        
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


    def bs_classifier(self, att, rna_mask, compound_mask):
        """
        Predict binding sites using attention features and DTI prediction as guidance.
        
        Args:
            att: Attention features [B, L_rna+1, L_compound+1, D]
            rna_mask: RNA sequence mask [B, L_rna]
            compound_mask: Compound mask [B, L_compound]
            
        Returns:
            bs_pred: Binding site predictions [B, L_rna]
        """
        # Use DTI prediction as guidance
        dti_pred = self.dti_classifier(att, rna_mask, compound_mask)
        dti_pred = self.sigmoid(dti_pred)
        dti_pred = dti_pred.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        
        # Remove guide tokens (first row and column)
        att = att[:, 1:, 1:, :]  # [B, L_rna, L_compound, D]
        
        # Apply DTI guidance
        att = dti_pred * att

        # Predict binding sites
        bs_pred = self.relu(self.bs_linear1(att))  # [B, L_rna, L_compound, 1]
        bs_pred = self.bs_linear2(bs_pred)  # [B, L_rna, L_compound, 1]
        bs_pred = torch.mean(bs_pred, dim=2)  # [B, L_rna, 1] - average over compound
        bs_pred = self.sigmoid(bs_pred)  # [B, L_rna, 1]
        
        return bs_pred

    def dti_classifier(self, att, rna_mask, compound_mask):
        """
        Predict drug-target interaction using attention features.
        
        Args:
            att: Attention features [B, L_rna+1, L_compound+1, D]
            rna_mask: RNA sequence mask [B, L_rna]
            compound_mask: Compound mask [B, L_compound]
            
        Returns:
            dti_pred: Drug-target interaction prediction [B, 1]
        """
        # Create masks for guide tokens
        batch_size = rna_mask.shape[0]
        guide_rna_mask = torch.ones(
            (batch_size, 1), 
            dtype=rna_mask.dtype, 
            device=rna_mask.device
        )  # [B, 1]
        guide_compound_mask = torch.ones(
            (batch_size, 1), 
            dtype=compound_mask.dtype, 
            device=compound_mask.device
        )  # [B, 1]
        
        # Extend masks to include guide tokens
        rna_mask_ext = torch.cat([guide_rna_mask, rna_mask], dim=1)  # [B, L_rna+1]
        compound_mask_ext = torch.cat([guide_compound_mask, compound_mask], dim=1)  # [B, L_compound+1]

        # Create attention mask
        att_mask = rna_mask_ext.unsqueeze(2) * compound_mask_ext.unsqueeze(1)  # [B, L_rna+1, L_compound+1]

        # Compute masked attention mean
        att_sum = (att * att_mask.unsqueeze(-1)).sum(dim=(1, 2))  # [B, D]
        valid_count = att_mask.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)  # [B, 1]
        valid_count = valid_count.to(att_sum.dtype)
        att_mean = att_sum / valid_count  # [B, D]

        # Predict DTI
        dti_pred = self.relu(self.dti_linear1(att_mean))  # [B, 16]
        dti_pred = self.dti_linear2(dti_pred)  # [B, 1]

        return dti_pred

    def forward(self, rna_input, compound_input, task):
        """
        Forward pass of the model.
        
        Args:
            rna_input: RNA sequence tokens [B, L_rna]
            compound_input: Tuple of (x, edge_index, edge_attr, batch) for compound graph
            task: Task type ('bs' for binding site prediction, 'dti' for drug-target interaction)
            
        Returns:
            Prediction based on task type
        """
        # RNA processing
        rna_mask = (rna_input >= 4) & (rna_input <= 7)  # [B, L_rna]

        # Get RNA embeddings
        rna = self.rna_embedding(rna_input, repr_layers=[12])
        rna = rna["representations"][12]  # [B, L_rna, 640]
        rna = self.relu(self.rna_linear(rna))  # [B, L_rna, 128]
        rna = self.relu(self.rna_linear2(rna))  # [B, L_rna, 128]
        rna = rna * rna_mask.unsqueeze(-1).float()  # Apply mask
        
        # Compute RNA guide token (mean of valid positions)
        mask_sum = rna_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        rna_guide_token = torch.sum(rna, dim=1, keepdim=True) / mask_sum.unsqueeze(-1)  # [B, 1, 128]

        # Compound processing
        batch = compound_input[3]
        _, compound = self.compound_embedding(
            compound_input[0],  # x
            compound_input[1],  # edge_index
            compound_input[2],  # edge_attr
            compound_input[3]   # batch
        )  # [N, 300] where N is total number of nodes
        compound = self.relu(self.compound_linear(compound))  # [N, 128]
        compound = self.relu(self.compound_linear2(compound))  # [N, 128]
        compound, compound_mask = to_dense_batch(compound, batch)  # [B, L_compound, 128], [B, L_compound]
        
        # Compute compound guide token (mean of valid positions)
        mask_sum = compound_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        compound_guide_token = torch.sum(
            compound * compound_mask.unsqueeze(-1), 
            dim=1, 
            keepdim=True
        ) / mask_sum.unsqueeze(-1)  # [B, 1, 128]

        # Concatenate guide tokens
        rna = torch.cat([rna_guide_token, rna], dim=1)  # [B, L_rna+1, 128]
        compound = torch.cat([compound_guide_token, compound], dim=1)  # [B, L_compound+1, 128]
        compound = torch.transpose(compound, 1, 2)  # [B, 128, L_compound+1]

        # Compute attention
        att_weight = torch.matmul(self.att_weight1, self.att_weight2)  # [128, 128]
        att = torch.matmul(rna, att_weight)  # [B, L_rna+1, 128]
        att = torch.matmul(att, compound)  # [B, L_rna+1, L_compound+1]

        # Expand for element-wise multiplication
        att_expanded = att.unsqueeze(-1)  # [B, L_rna+1, L_compound+1, 1]
        rna_expanded = rna.unsqueeze(2)  # [B, L_rna+1, 1, 128]
        compound = torch.transpose(compound, 1, 2)  # [B, L_compound+1, 128]
        compound_expanded = compound.unsqueeze(1)  # [B, 1, L_compound+1, 128]

        # Combine RNA and compound features
        rna_compound = att_expanded * rna_expanded * compound_expanded  # [B, L_rna+1, L_compound+1, 128]
        
        # Shared network
        rc = self.relu(self.shared_linear0(rna_compound))  # [B, L_rna+1, L_compound+1, 64]
        rc = self.relu(self.shared_linear1(rc))  # [B, L_rna+1, L_compound+1, 32]

        # Task-specific prediction
        if task == 'bs':
            bs_pred = self.bs_classifier(rc, rna_mask, compound_mask)
            return bs_pred
        elif task == 'dti':
            dti_pred = self.dti_classifier(rc, rna_mask, compound_mask)
            return dti_pred
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'bs' or 'dti'.")