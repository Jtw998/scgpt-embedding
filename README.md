# scGPT Embedding Engine
> scGPT embedding computation tool fully compatible with official pre-trained models

## 📖 Overview
A lightweight, easy-to-use tool for extracting high-quality pre-trained embeddings from single-cell RNA-seq data using the scGPT foundation model. Implements the exact preprocessing and inference logic from the official scGPT pipeline with automatic hardware acceleration.

## ✨ Key Features
- 100% compatible with all official scGPT pre-trained models
- Auto-detects and uses CUDA, MPS (Apple Silicon), or CPU for acceleration
- Supports FlashAttention for 2-4x faster inference
- Extracts 512D cell-level embeddings, gene-level embeddings, and full sequence embeddings
- Complete built-in preprocessing pipeline aligned with scGPT pre-training
- Both Python API and command-line interface available
- Production-ready with robust error handling and validation

## 🚀 Quick Start
### Installation
```bash
# Install dependencies (requires scGPT installed first)
pip install scanpy torch numpy scipy
```
Download pre-trained model files from [scGPT Model Zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo) (vocab.json, best_model.pt) and place them in the working directory.

### Command Line
```bash
python scgpt_embedding.py --input data.h5ad --output data_with_embeddings.h5ad
```

### Python API
```python
import scanpy as sc
from scgpt_embedding import scGPTEmbeddingEngine

engine = scGPTEmbeddingEngine()
adata = sc.read("data.h5ad")
adata = engine.compute_embeddings(adata)

# Access embeddings: adata.obsm["X_scGPT"] (shape: n_cells × 512)
```

## 📝 Output
- Cell embeddings stored in `adata.obsm["X_scGPT"]`
- Optional gene embeddings stored in `adata.varm["scGPT_gene_emb"]`

## 📄 Citation
```
@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
```

## 📄 License
MIT
