# DeepRNA-DTI: A Deep Learning Approach for RNA-Compound Interaction Prediction with Binding Site Interpretability
## Abstract
RNA-targeted therapeutics represent a promising frontier for expanding the druggable genome beyond conventional protein targets. However, computational prediction of RNA-compound interactions remains challenging due to limited experimental data and the inherent complexity of RNA structures. Here, we present DeepRNA-DTI, a novel sequence-based deep learning approach for RNA-compound interaction prediction with binding site interpretability. Our model leverages transfer learning from pretrained embeddings, RNA-FM for RNA sequences and Mole-BERT for compounds, and employs a multitask learning framework that simultaneously predicts both presence of interactions and nucleotide-level binding sites. This dual prediction strategy provides mechanistic insights into RNA-compound recognition patterns. Trained on a comprehensive dataset integrating resources from the Protein Data Bank and literature sources, DeepRNA-DTI demonstrates superior performance compared to existing methods. The model shows consistent effectiveness across diverse RNA subtypes, highlighting its robust generalization capabilities. Application to high-throughput virtual screening of over 48 million compounds against oncogenic pre-miR-21 successfully identified known binders and novel chemical scaffolds with RNA-specific physicochemical properties. By combining sequence-based predictions with binding site interpretability, DeepRNA-DTI advances our ability to identify promising RNA-targeting compounds and offers new opportunities for RNA-directed drug discovery.


## Setup
### 1. Anaconda Environment
To set up the environment, create anaconda environment with the provided environment.yml file
```bash
cd DeepRNA-DTI
conda env create --file environment.yml -n DeepRNA_DTI
conda activate DeepRNA_DTI
```

### 2. Pre-trained Model setting
DeepRNA-DTI utilizes pre-trained weights from RNA-FM and MoleBERT. You need to download the model codes and the corresponding pre-trained weights from each repository and place them in the Model/pretrained_model/ directory.

- RNA-FM: https://github.com/ml4bio/RNA-FM/tree/main
- MoleBERT : https://github.com/junxia97/Mole-BERT/tree/main

```bash
mkdir Model
mkdir Model/pretrained_model
mkdir Model/trained_weight
cd Model/pretrained_model
git clone https://github.com/ml4bio/RNA-FM.git
git clone https://github.com/junxia97/Mole-BERT.git
```

### 3. Prediction
The pre-trained weight files are too large to store on GitHub. Please download the pre-trained weights from the Google Drive link and place them in the `Model/trained_weight/` directory:

Or You can download files with gdown.
```bash
pip install gdown
cd Model/trained_weight

gdown https://drive.google.com/uc?id=1YW7kMy9O6L8fvubqDtN_TASrZakIJ0Q3
gdown https://drive.google.com/uc?id=11Jf8I5TpsupC2_Dd8zTWgVyPXbgM5fXm
gdown https://drive.google.com/uc?id=1PFk1IqjQIYDrSGXYHRm_9FVlKjI-YfWH
gdown https://drive.google.com/uc?id=1Sa7Stz9POe8BsyCHfOZUpCk3tUgeUxyS
gdown https://drive.google.com/uc?id=1A1Kq0B5-t9IDbEFUVbPHJocMzXP7O7W8

To perform interaction prediction using the pre-trained weights, run the following command:
```bash
python test.py
```

### 4. Train
If you'd like to train the model from scratch or fine-tune it, use the following command:
```bash
python train.py
```


## Contact
Haelee Bae, haeleeeeleah@gm.gist.ac.kr

Hojung Nam (Corresponding Author), hjnam@gist.ac.kr
