import os
import sys
import ast
from itertools import repeat

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from rdkit.Chem import AllChem

# Setup paths for pretrained models
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
RNA_FM_PATH = os.path.join(PROJECT_ROOT, 'Model', 'pretrained_model', 'RNA-FM')
MOLE_BERT_PATH = os.path.join(PROJECT_ROOT, 'Model', 'pretrained_model', 'Mole-BERT')

# Add paths to sys.path for imports
if RNA_FM_PATH not in sys.path:
    sys.path.append(RNA_FM_PATH)
if MOLE_BERT_PATH not in sys.path:
    sys.path.append(MOLE_BERT_PATH)

# Import pretrained model modules
import fm
from loader import mol_to_graph_data_obj_simple

# Initialize RNA-FM batch converter
_, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()

class RNADataset(Dataset):
    """Dataset for RNA sequences and binding sites."""
    
    def __init__(self, root):
        """
        Initialize RNA dataset.
        
        Args:
            root: Path to dataset directory containing raw/interactions.csv
        """
        input_df = pd.read_csv(f'{root}/raw/interactions.csv', sep=',')
        self.rna_sequences = input_df['sequence'].tolist()
        self.binding_sites = input_df['binding_site_index'].tolist()

        # Tokenize all sequences
        seqs = [(i, seq) for i, seq in enumerate(self.rna_sequences)]
        _, _, self.seq_tokens = batch_converter(seqs)

    def __len__(self):
        return len(self.rna_sequences)

    def __getitem__(self, idx):
        """
        Get RNA sequence tokens, binding sites, and sequence length.
        
        Args:
            idx: Index of the sample
            
        Returns:
            rna_sequence_tokens: Tokenized RNA sequence
            rna_binding_sites: Binding site labels with padding
            rna_seq_len: Length of the actual sequence
        """
        rna_sequence_tokens = self.seq_tokens[idx]
        
        # Parse binding sites (stored as string representation of list)
        binding_sites_list = ast.literal_eval(self.binding_sites[idx])
        rna_seq_len = torch.tensor(len(binding_sites_list))
        rna_binding_sites = torch.tensor(binding_sites_list, dtype=torch.float32)
        
        # Pad binding sites: [0] + binding_sites + [0, 0, ...]
        padding_size = len(rna_sequence_tokens) - rna_seq_len - 1
        rna_binding_sites = torch.cat([
            torch.zeros(1),
            rna_binding_sites,
            torch.zeros(max(0, padding_size))
        ])

        return rna_sequence_tokens, rna_binding_sites, rna_seq_len


class GraphDataset(InMemoryDataset):
    """Dataset for molecular graphs from SMILES strings."""
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, empty=False):
        """
        Initialize graph dataset.
        
        Args:
            root: Path to dataset directory
            transform: Optional transform to apply to data
            pre_transform: Optional transform to apply before saving
            pre_filter: Optional filter function
            empty: If True, create empty dataset (for processing)
        """
        self.interaction_data_path = root
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        """Get graph data at index."""
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        """Return list of raw file names."""
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        """Return processed file name."""
        return 'geometric_data_processed.pt'

    def download(self):
        """Download is not supported."""
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):
        """Process SMILES strings into graph data objects."""
        input_df = pd.read_csv(f'{self.interaction_data_path}/raw/interactions.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
        
        data_list = []
        for smiles in smiles_list:
            mol = AllChem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f'Invalid SMILES: {smiles}')
            data = mol_to_graph_data_obj_simple(mol)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphDataset_test(Dataset):
    """Test dataset for molecular graphs (on-the-fly processing)."""
    
    def __init__(self, root, file_name='interactions.csv'):
        """
        Initialize test graph dataset.
        
        Args:
            root: Path to dataset directory
            file_name: Name of the CSV file containing SMILES
        """
        self.interaction_data_path = root
        self.file_name = file_name
        input_df = pd.read_csv(f'{self.interaction_data_path}/raw/{self.file_name}', sep=',')
        self.smiles_list = input_df['smiles'].tolist()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        Get graph data for a single molecule.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Graph data object
        """
        smiles = self.smiles_list[idx]
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES: {smiles}')
        
        data = mol_to_graph_data_obj_simple(mol)
        return data


class InteractionDataset:
    """Dataset combining compound graphs, RNA sequences, and interaction labels."""
    
    def __init__(self, root, graph_dataset, rna_dataset):
        """
        Initialize interaction dataset.
        
        Args:
            root: Path to dataset directory
            graph_dataset: GraphDataset instance for compounds
            rna_dataset: RNADataset instance for RNA sequences
        """
        input_df = pd.read_csv(f'{root}/raw/interactions.csv', sep=',')
        self.labels = input_df['interactions'].values
        self.graph_dataset = graph_dataset
        self.rna_dataset = rna_dataset

    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        """
        Get interaction data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            compound_graph_data: Graph data for compound
            rna_sequence_tokens: Tokenized RNA sequence
            rna_binding_sites: Binding site labels
            rna_seq_len: RNA sequence length
            interaction: Interaction label (0 or 1)
        """
        compound_graph_data = self.graph_dataset[idx]
        rna_sequence_tokens, rna_binding_sites, rna_seq_len = self.rna_dataset[idx]
        interaction = torch.tensor(self.labels[idx], dtype=torch.float32)

        return compound_graph_data, rna_sequence_tokens, rna_binding_sites, rna_seq_len, interaction



class CombinedDataLoader:
    """Combined dataloader for binding site (BS) and drug-target interaction (DTI) tasks.
    
    Cycles through BS loader while iterating through DTI loader once.
    BS loader restarts when exhausted.
    """
    
    def __init__(self, bs_loader, dti_loader):
        """
        Initialize combined dataloader.
        
        Args:
            bs_loader: DataLoader for binding site data
            dti_loader: DataLoader for drug-target interaction data
        """
        self.bs_loader = bs_loader
        self.dti_loader = dti_loader
        self._reset_iterators()

    def _reset_iterators(self):
        """Reset iterators and state."""
        self.iter_bs = iter(self.bs_loader)
        self.iter_dti = iter(self.dti_loader)
        self.bs_exhausted = False

    def __iter__(self):
        """Reset iterators for new epoch."""
        self._reset_iterators()
        return self

    def __next__(self):
        """Get next batch pair (BS, DTI)."""
        # Get BS batch (restart if exhausted)
        try:
            batch_bs = next(self.iter_bs)
        except StopIteration:
            self.iter_bs = iter(self.bs_loader)
            batch_bs = next(self.iter_bs)

        # Get DTI batch (raise StopIteration when exhausted)
        try:
            batch_dti = next(self.iter_dti)
        except StopIteration:
            raise StopIteration

        return batch_bs, batch_dti


class CombinedDataLoader_rawdata:
    """Combined dataloader for raw data processing.
    
    Similar to CombinedDataLoader but allows BS loader to be None after exhaustion.
    """
    
    def __init__(self, bs_loader, dti_loader):
        """
        Initialize combined dataloader for raw data.
        
        Args:
            bs_loader: DataLoader for binding site data
            dti_loader: DataLoader for drug-target interaction data
        """
        self.bs_loader = bs_loader
        self.dti_loader = dti_loader
        self._reset_iterators()

    def _reset_iterators(self):
        """Reset iterators and state."""
        self.iter_bs = iter(self.bs_loader) if self.bs_loader is not None else None
        self.iter_dti = iter(self.dti_loader)
        self.bs_exhausted = False

    def __iter__(self):
        """Reset iterators for new epoch."""
        self._reset_iterators()
        return self

    def __next__(self):
        """Get next batch pair (BS, DTI)."""
        # Get BS batch (can be None after exhaustion)
        if self.bs_exhausted:
            batch_bs = None
        else:
            try:
                batch_bs = next(self.iter_bs)
            except StopIteration:
                self.iter_bs = None
                batch_bs = None
                self.bs_exhausted = True

        # Get DTI batch (raise StopIteration when exhausted)
        try:
            batch_dti = next(self.iter_dti)
        except StopIteration:
            raise StopIteration


        return batch_bs, batch_dti