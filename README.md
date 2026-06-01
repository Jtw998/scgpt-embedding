# scGPT Embedding Engine

scGPT embedding computation tool, fully self-contained, no external dependencies beyond PyTorch and scanpy.

## Scripts

| File | Input | Output |
|------|-------|--------|
| `scgpt_embedding.py` | `.h5ad` file + model dir (`args.json`, `best_model.pt`, `vocab.json`) | Cell embeddings → `h5ad.obsm["X_scGPT"]` (+ optional `.npy`) |
| `scgpt_finetune.py` | `.h5ad` file + model dir | Fine-tuned model (`finetuned_model.pt`) + RNA projection head (`rna_head.pt`) |
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

### Contrastive fine-tune the model

```bash
python scgpt_finetune.py --h5ad data.h5ad --model-dir . --output-dir ./finetuned --epochs 30
```

Options:
- `--lr 1e-4` — learning rate (default: 1e-4)
- `--temperature 0.1` — SimCLR temperature (default: 0.1)
- `--proj-dim 128` — projection head output dimension (default: 128)
- `--dropout-rate 0.3` — fraction of gene tokens dropped per view (default: 0.3)
- `--batch-size 64` — training batch size (default: 64)
- `--max-seq-len 1200` — max gene tokens per cell (default: 1200)
- `--freeze-layers 0` — freeze first N transformer layers (default: 0 = unfreeze all)
- `--fast-transformer` — enable flash-attention (requires `flash-attn`)

### Extract gene embeddings by gene list

```bash
python extract_gene_embeddings.py . gene_list.tsv gene_embeddings.npy
```

Gene list must be TSV/CSV with a `gene_name` column.

## Contrastive Fine-tuning

`scgpt_finetune.py` fine-tunes a pre-trained scGPT model on custom scRNA-seq data using a SimCLR-style contrastive objective. This adapts the model to the target dataset's expression patterns without requiring labeled data.

**How it works:**

1. Each cell passes through the pre-trained scGPT transformer, producing a sequence of gene-token representations.
2. Two independent gene-dropout masks are applied to the gene tokens, creating two augmented views of the same cell.
3. Masked gene tokens are mean-pooled and projected through a 2-layer MLP projection head.
4. InfoNCE loss pulls views of the same cell together while pushing views of different cells apart.
5. After training, the fine-tuned model and an `RNAHead` (mean-pool + Linear projection, no L2 norm) are saved for downstream embedding.

**Output files:**
- `finetuned_model.pt` — state dict of the fine-tuned TransformerModel
- `rna_head.pt` — state dict of the RNAHead, which can be loaded later for producing L2-normalized cell embeddings via `scgpt_embedding.py`

**When to use:** Fine-tune when your target dataset comes from a different tissue, species, or protocol than the pre-training data. Contrastive fine-tuning typically improves clustering, visualization, and downstream prediction quality on the target data.

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
