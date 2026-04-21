# scGPT Embedding Engine

scGPT embedding computation tool, fully self-contained, no external dependencies beyond PyTorch and scanpy.

## Scripts

| File | Input | Output |
|------|-------|--------|
| `scgpt_embedding.py` | `.h5ad` file + model dir (`args.json`, `best_model.pt`, `vocab.json`) | Cell embeddings → `h5ad.obsm["X_scGPT"]` (+ optional `.npy`) |
| `extract_gene_embeddings.py` | Model dir + gene list file (TSV/CSV with `gene_name` column) | Gene embedding matrix `[n_genes, 512]` → `.npy` file |
| `node_embedding.py` | `.h5ad` file + model dir | Cell embeddings + per-patient KNN graph structures |

## Quick Start

### Compute cell embeddings

```bash
python scgpt_embedding.py data.h5ad --model-dir . --batch-size 32
```

Options:
- `--output-npy path.npy` — save embeddings as numpy file
- `--no-save-h5ad` — skip writing back to h5ad
- `--fast-transformer` — enable flash-attention (requires `flash-attn`)

### Extract gene embeddings by gene list

```bash
python extract_gene_embeddings.py . gene_list.tsv gene_embeddings.npy
```

Gene list must be TSV/CSV with a `gene_name` column.

## Installation

```bash
pip install torch scanpy numpy scipy pandas tqdm
```

Download pre-trained model files from [scGPT Model Zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo) and place `args.json`, `best_model.pt`, `vocab.json` in the working directory.

## Device

Auto-detects: CUDA > Apple Silicon MPS > CPU.

## Citation

```
@article{cui2023scGPT,
  title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
  journal={bioRxiv},
  year={2023}
}
```
