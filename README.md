# DeepRNA-DTI: A Deep Learning Approach for RNA-Compound Interaction Prediction with Binding Site Interpretability

## Abstract

RNA-targeted therapeutics represent a promising frontier for expanding the druggable genome beyond conventional protein targets. However, computational prediction of RNA-compound interactions remains challenging due to limited experimental data and the inherent complexity of RNA structures. Here, we present DeepRNA-DTI, a novel sequence-based deep learning approach for RNA-compound interaction prediction with binding site interpretability. Our model leverages transfer learning from pretrained embeddings, RNA-FM for RNA sequences and Mole-BERT for compounds, and employs a multitask learning framework that simultaneously predicts both presence of interactions and nucleotide-level binding sites. This dual prediction strategy provides mechanistic insights into RNA-compound recognition patterns. Trained on a comprehensive dataset integrating resources from the Protein Data Bank and literature sources, DeepRNA-DTI demonstrates superior performance compared to existing methods. The model shows consistent effectiveness across diverse RNA subtypes, highlighting its robust generalization capabilities. Application to high-throughput virtual screening of over 48 million compounds against oncogenic pre-miR-21 successfully identified known binders and novel chemical scaffolds with RNA-specific physicochemical properties. By combining sequence-based predictions with binding site interpretability, DeepRNA-DTI advances our ability to identify promising RNA-targeting compounds and offers new opportunities for RNA-directed drug discovery.

## Overview

!Overview.png

## Setup

### 1. Environment Setup

Create a conda environment using the provided `environment.yml`:

```bash
cd DeepRNA-DTI
conda env create --file environment.yml -n DeepRNA_DTI
conda activate DeepRNA_DTI
```

### 2. Pretrained Model Setup

DeepRNA-DTI uses pretrained weights from RNA-FM and Mole-BERT. Download the model code and weights:

```bash
mkdir -p Model/pretrained_model Model/trained_weight
cd Model/pretrained_model

# Clone RNA-FM
git clone https://github.com/ml4bio/RNA-FM.git

# Clone Mole-BERT
git clone https://github.com/junxia97/Mole-BERT.git
```

**Model Sources:**
- RNA-FM: https://github.com/ml4bio/RNA-FM
- Mole-BERT: https://github.com/junxia97/Mole-BERT

### 3. Download Trained Weights

We provide pretrained weights for all four generalization scenarios. Download them using gdown:

```bash
pip install gdown
cd Model/trained_weight
gdown "https:// ... --folder"
```

#### Available Pretrained Weights

| Scenario | Description | Download |
|----------|-------------|----------|
| `unseen_pair` | Pair unseen at test time | `gdown "https://drive.google.com/drive/folders/109jdGX0yKuC7AgD50y4ZPs1rAa2h0Q9f" --folder` |
| `unseen_rna` | RNA unseen, compounds from training set | `gdown "https://drive.google.com/drive/folders/1UxpOBIyFzw90H4Gy5wyffYSgSVDUvMLi" --folder` |
| `unseen_compound` | Compounds unseen, RNA from training set | `gdown "https://drive.google.com/drive/folders/1uWNXatSLZ1qxhCVkhAV1Y4yiPt3NCW2V" --folder` |
| `unseen_both` | Both RNA and compound unseen at test time | `gdown "https://drive.google.com/drive/folders/1d0UIwn9sYiAeBgHkqXco4NQDOPnrk28S" --folder` |

After downloading, organize the weights as follows:
```
Model/trained_weight/
├── unseen_pair/
│   ├── model_fold0.pt
│   ├── model_fold1.pt
│   ├── model_fold2.pt
│   ├── model_fold3.pt
│   └── model_fold4.pt
├── unseen_rna/
│   └── ...
├── unseen_compound/
│   └── ...
└── unseen_both/
    └── ...
```

## Usage

### Testing (Inference)

Evaluate the model on test data:

```bash
python test.py --test_type unseen_pair
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--test_type` | `unseen_pair` | Dataset split: `unseen_pair`, `unseen_rna`, `unseen_compound`, `unseen_both` |
| `--model_folder_path` | `./Model/trained_weight` | Path to trained model weights |
| `--data_folder_path` | `./Dataset/` | Path to dataset |
| `--batch_size` | `32` | Batch size for evaluation |

### Training

Train the model from scratch:

```bash
python train.py --train_type unseen_pair --num_epochs 100
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--train_type` | `unseen_pair` | Training split type: `unseen_pair`, `unseen_rna`, `unseen_compound`, `unseen_both` |
| `--batch_size` | `32` | Batch size |
| `--num_epochs` | `100` | Number of training epochs |
| `--learning_rate` | `0.001` | Learning rate |
| `--weight_decay` | `1e-4` | L2 regularization |

## Dataset Structure

```
Dataset/
├── unseen_pair/          # Pair unseen at test time
├── unseen_rna/           # RNA unseen, compounds from training
├── unseen_compound/      # Compounds unseen, RNA from training
└── unseen_both/          # Both RNA and compound unseen at test time
    ├── dti_data/         # Drug-target interaction data
    │   ├── train_fold0/
    │   │   └── raw/
    │   │       └── interactions.csv
    │   ├── train_fold1/
    │   ├── ...
    │   ├── val_fold0/
    │   └── test_fold/
    └── bs_data/          # Binding site data
        ├── train_fold0/
        ├── ...
        └── test_fold/
```

### Data Format (`interactions.csv`)

| Column | Description |
|--------|-------------|
| `sequence` | RNA sequence |
| `smiles` | Compound SMILES string |
| `interactions` | Binary label (1 = interacts, 0 = no interaction) |
| `binding_site_index` | List of nucleotide indices that bind to the compound |

## Evaluation Metrics

### Drug-Target Interaction (DTI)
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **AUPR**: Area under the precision-recall curve

### Binding Site (BS)
- **Micro-averaged AUC/AUPR**: Per-RNA sample average (only for positive interactions)


## Project Structure

```
DeepRNA_DTI_github/
├── train.py              # Training script
├── test.py               # Evaluation script
├── environment.yml       # Conda environment
├── README.md             # This file
├── src/
│   ├── model.py          # DeepRNA_DTI model architecture
│   ├── data_utils.py     # Dataset classes and data loaders
│   └── utils.py          # Loss functions and evaluation metrics
├── Model/
│   ├── pretrained_model/
│   │   ├── RNA-FM/       # RNA-FM pretrained model
│   │   └── Mole-BERT/    # Mole-BERT pretrained model
│   └── trained_weight/   # Trained DeepRNA-DTI weights
└── Dataset/
    ├── unseen_pair/
    ├── unseen_rna/
    ├── unseen_compound/
    └── unseen_both/
```


## Contact

- **Haelee Bae**: haeleeeeleah@gm.gist.ac.kr
- **Hojung Nam** (Corresponding Author): hjnam@gist.ac.kr
