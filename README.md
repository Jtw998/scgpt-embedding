# scGPT Embedding Engine

scGPT cell embedding + contrastive fine-tuning. Two backends: **MLX** (Apple Silicon Metal, 5–10× faster) and **PyTorch** (CUDA/MPS/CPU).

## Scripts

| File | Purpose | Backend |
|------|---------|---------|
| `scgpt_embedding.py` | Shared library — model, data, metrics, loss | PyTorch |
| `scgpt_finetune.py` | Contrastive fine-tune → saves model + RNA head | PyTorch |
| `benchmark_finetune.py` | Before/after clustering benchmark | PyTorch |
| `scgpt_mlx.py` | Inference + SimCLR fine-tune | **MLX** (Metal) |
| `benchmark_mlx.py` | Before/after benchmark | **MLX** (Metal) |
| `extract_gene_embeddings.py` | Extract gene embedding matrix from checkpoint | PyTorch |

**Shared logic** (`GeneVocab`, `SparseGeneDataset`, `DataCollator`, `TransformerModel`, `simclr_loss`, `RNAHead`, `clustering_metrics`, `embedding_stats`) lives in `scgpt_embedding.py`. Finetune and benchmark scripts import from it — no code duplication.

## Quick Start

### MLX (recommended on Apple Silicon)

```bash
# Inference only — 5× faster than PyTorch MPS
python scgpt_mlx.py data.h5ad --model-dir . --batch-size 16

# Fine-tune + save weights
python scgpt_mlx.py data.h5ad --model-dir . --batch-size 16 --finetune --epochs 10

# Before/after benchmark
python benchmark_mlx.py --subset 5000 --epochs 10 --batch-size 16
```

### PyTorch (CUDA / MPS / CPU)

```bash
# Inference
python scgpt_embedding.py data.h5ad --model-dir . --batch-size 32

# Fine-tune
python scgpt_finetune.py --h5ad data.h5ad --model-dir . --output-dir ./finetuned --epochs 30

# Benchmark
python benchmark_finetune.py --subset 5000 --epochs 10 --batch-size 32
```

## Performance (Apple Silicon)

| | PyTorch MPS | MLX Metal | Speedup |
|---|---|---|---|
| Embedding (inference) | ~3.7 it/s | **~36 it/s** | ~10× |
| Training (SimCLR) | ~2.7 it/s | **~22 it/s** | ~8× |
| 63万 cells embedding | ~30h | **~10 min** | — |

MLX uses `mx.fast.scaled_dot_product_attention` (Metal-accelerated flash attention) — no CPU fallback.

## Contrastive Fine-tuning

SimCLR InfoNCE loss with gene dropout augmentation. Two masked views per cell are pushed together; different cells pushed apart.

**How it works:**
1. Cell → scGPT transformer → gene-token sequence
2. Two independent gene-dropout masks create two augmented views
3. Masked tokens → mean pool → 2-layer MLP projection head
4. InfoNCE loss pulls same-cell views together, different cells apart
5. `RNAHead` (mean-pool + Linear) saved for downstream embedding

**When to use:** different tissue, species, or protocol from pre-training data. Improves clustering, visualization, and downstream prediction.

## MLX Fine-tuning Output

```bash
python scgpt_mlx.py data.h5ad --finetune --epochs 10
```

Saves to `./finetuned_mlx/`:
- `finetuned_model.npz` — scGPT weights
- `rna_head.npz` — projection head weights

## MLX Benchmark Output

```bash
python benchmark_mlx.py --subset 5000 --epochs 10
```

Prints per-epoch clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) and a final SUMMARY table showing Δ before vs. after fine-tuning. Saves `benchmark_mlx_output/benchmark_log.json`.

## Installation

```bash
# PyTorch
pip install torch scanpy numpy scipy pandas tqdm scikit-learn

# MLX (Apple Silicon only)
pip install mlx mlx-metal scanpy numpy pandas tqdm scikit-learn
```

Download pre-trained model files from [scGPT Model Zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo):
- `args.json`
- `best_model.pt`
- `vocab.json`

## Device

| Backend | Auto-detection |
|---------|---------------|
| MLX | Always uses Metal GPU |
| PyTorch | CUDA > MPS > CPU |

## Citation

```
@article{cui2023scGPT,
  title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
  journal={bioRxiv},
  year={2023}
}
```
