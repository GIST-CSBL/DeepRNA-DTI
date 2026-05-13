# Baseline Comparison

Re-implementations of three Drug–Protein DTA baselines (**DeepDTA**,
**GraphDTA**, **GraphATT-DTA**) adapted to the Drug–RNA DTI task, used as
comparison baselines for DeepRNA-DTI.

DeepRNA-DTI is additionally benchmarked against three recently published
RNA-targeted models (**DeepRSMA**, **RNAsmol**, **RSApred**); those were
run with the authors' published code on the DeepRNA-DTI dataset and are
not included here.

## Files

```
comparison.py     # data processing + training + testing for the three baselines
environment.yaml  # conda environment specification
```

## Setup
```bash
conda env create -f environment.yaml
conda activate python_dgl
```

## Usage

One run = data processing + 5-fold training + 5-fold testing, for one
(model, split) combination.

```bash
python comparison.py --model model_name --unseen-type type --dataset-root ../Dataset
```

| Flag | Choices | Notes |
|---|---|---|
| `--model` | `DeepDTA` / `GraphDTA` / `GraphATT_DTA` | required |
| `--unseen-type` | `compound` / `rna` / `pair` / `both` | required |
| `--dataset-root` | path | default `../Dataset` |

