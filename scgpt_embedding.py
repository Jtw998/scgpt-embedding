
"""
scGPT Cell Embedding Computation Script
Standalone script to compute 512-dimensional L2-normalized cell embeddings from scRNA-seq data
using pre-trained scGPT model.

Key features:
- Process all genes present in the vocabulary (no filtering based on gene presence)
- Preserve zero expression values (no filtering of genes with zero expression)
- Automatic device detection (CUDA > Apple Silicon MPS > CPU)
- Save embeddings back to h5ad file or as separate numpy file
- Easy-to-use command line interface
"""
# Enable MPS fallback for unsupported operations (fix for PyTorch MPS compatibility)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import numpy as np
import scanpy as sc
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from typing import Optional, Union, Dict, Tuple

# Import reusable components from existing node_embedding.py
from node_embedding import (
    GeneVocab,
    TransformerModel,
    DataCollator,
    load_scgpt_model
)

def get_device() -> torch.device:
    """
    Automatically detect and return the best available device for computation.
    Priority order: CUDA > Apple Silicon MPS > CPU.

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def validate_inputs(h5ad_path: Path, model_dir: Path) -> None:
    """
    Validate that all required input files exist and are accessible.

    Args:
        h5ad_path: Path to input h5ad file
        model_dir: Path to directory containing scGPT model files

    Raises:
        FileNotFoundError: If any required file is missing
        PermissionError: If any required file is not readable
    """
    # Validate h5ad file
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad file not found: {h5ad_path}")
    if not h5ad_path.is_file():
        raise ValueError(f"Path is not a file: {h5ad_path}")

    # Validate model directory
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Path is not a directory: {model_dir}")

    # Validate required model files
    required_files = ["args.json", "best_model.pt", "vocab.json"]
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required model file: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

class FullGeneDataset(Dataset):
    """
    Dataset class that includes all vocabulary genes for each cell,
    preserving zero expression values.
    """
    def __init__(
        self,
        count_matrix: np.ndarray,
        gene_ids: np.ndarray,
        vocab: GeneVocab,
        model_configs: Dict,
        max_seq_len: int = 1200
    ):
        """
        Args:
            count_matrix: Gene expression matrix with shape (n_cells, n_genes)
            gene_ids: Array of vocabulary IDs for each gene in the count matrix
            vocab: GeneVocab object
            model_configs: Model configuration dictionary
            max_seq_len: Maximum sequence length for model input
        """
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.vocab = vocab
        self.pad_token_id = vocab["<pad>"]
        self.cls_token_id = vocab["<cls>"]
        self.pad_value = model_configs.get("pad_value", 0)
        self.max_seq_len = max_seq_len
        # Indices of valid genes (present in vocabulary)
        self.valid_gene_mask = gene_ids >= 0
        self.valid_gene_ids = gene_ids[self.valid_gene_mask]
        print(f"Using {len(self.valid_gene_ids)} genes from vocabulary for embedding computation")

    def __len__(self) -> int:
        return len(self.count_matrix)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single cell's input sequence, including all valid vocabulary genes
        (preserving zero expression values), sampled to max sequence length if needed.

        Args:
            idx: Cell index

        Returns:
            Dictionary containing gene IDs and expression values for the cell
        """
        # Get full expression profile for the cell
        full_expression = self.count_matrix[idx]
        # Get expression values for all valid genes (including zeros)
        valid_expressions = full_expression[self.valid_gene_mask]

        # We have more genes than max sequence length allows, so we need to sample
        if len(self.valid_gene_ids) > (self.max_seq_len - 1):
            # Random sample without replacement to select genes for this cell
            # This ensures zero expression genes have equal chance to be included
            sample_indices = np.random.choice(
                len(self.valid_gene_ids),
                size=self.max_seq_len - 1,
                replace=False
            )
            selected_gene_ids = self.valid_gene_ids[sample_indices]
            selected_expressions = valid_expressions[sample_indices]
        else:
            # All genes fit within max sequence length, use all
            selected_gene_ids = self.valid_gene_ids
            selected_expressions = valid_expressions

        # Add CLS token at the beginning
        genes = np.concatenate([[self.cls_token_id], selected_gene_ids])
        expressions = np.concatenate([[self.pad_value], selected_expressions])

        # Convert to PyTorch tensors
        genes = torch.from_numpy(genes).long()
        expressions = torch.from_numpy(expressions).float()

        return {
            "genes": genes,
            "expressions": expressions,
        }

def compute_full_embeddings(
    adata: sc.AnnData,
    model: TransformerModel,
    vocab: GeneVocab,
    model_configs: Dict,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute cell embeddings using all vocabulary genes, preserving zero expression values.

    Args:
        adata: AnnData object containing single-cell RNA-seq data
        model: Loaded pre-trained scGPT model
        vocab: GeneVocab object
        model_configs: Model configuration dictionary
        batch_size: Batch size for embedding computation
        device: Device to use for computation (auto-detected if None)

    Returns:
        np.ndarray: L2-normalized cell embeddings with shape (n_cells, 512)
    """
    if device is None:
        device = get_device()

    # Convert count matrix to dense numpy array if it's sparse
    count_matrix = adata.X
    if not isinstance(count_matrix, np.ndarray):
        count_matrix = count_matrix.toarray()

    # Map gene names to vocabulary IDs
    if "id_in_vocab" not in adata.var:
        gene_names = adata.var_names
        gene_ids = np.array([vocab[gene] if gene in vocab else -1 for gene in gene_names])
        adata.var["id_in_vocab"] = gene_ids
    else:
        gene_ids = np.array(adata.var["id_in_vocab"])

    # Create dataset and dataloader
    max_seq_len = model_configs.get("max_seq_len", 1200)
    dataset = FullGeneDataset(
        count_matrix=count_matrix,
        gene_ids=gene_ids,
        vocab=vocab,
        model_configs=model_configs,
        max_seq_len=max_seq_len
    )

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab["<pad>"],
        pad_value=model_configs.get("pad_value", 0),
        do_mlm=False,
        do_binning=True,
        max_length=max_seq_len,
        sampling=False,  # Disable sampling, we already sampled in dataset
        keep_first_n_tokens=1
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,  # Disable multi-processing to avoid pickling issues
        pin_memory=False
    )

    # Compute embeddings
    embedding_dim = model_configs.get("embsize", 512)
    cell_embeddings = np.zeros((len(dataset), embedding_dim), dtype=np.float32)

    model.eval()
    model.to(device)

    with torch.no_grad():
        current_idx = 0
        for batch in tqdm(dataloader, desc="Computing cell embeddings"):
            # Move batch to device
            input_gene_ids = batch["gene"].to(device)
            input_values = batch["expr"].to(device)

            # Forward pass through model
            outputs = model(input_gene_ids, input_values)

            # Extract CLS token embedding as cell embedding
            batch_embeddings = outputs["cell_emb"].cpu().numpy()

            # Store embeddings
            batch_size = len(batch_embeddings)
            cell_embeddings[current_idx:current_idx + batch_size] = batch_embeddings
            current_idx += batch_size

    # L2 normalize embeddings
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )

    return cell_embeddings

def run_embedding_pipeline(
    h5ad_path: Union[str, Path],
    model_dir: Union[str, Path] = ".",
    output_npy: Optional[Union[str, Path]] = None,
    save_h5ad: bool = True,
    batch_size: int = 32,
    use_fast_transformer: bool = False,
) -> None:
    """
    End-to-end pipeline for scGPT cell embedding computation.

    Args:
        h5ad_path: Path to input h5ad file
        model_dir: Path to directory containing scGPT model files
        output_npy: Optional path to save embeddings as numpy file
        save_h5ad: Whether to save embeddings back to the original h5ad file
        batch_size: Batch size for embedding computation
        use_fast_transformer: Whether to use flash-attention fast transformer
    """
    # Convert paths to Path objects
    h5ad_path = Path(h5ad_path)
    model_dir = Path(model_dir)

    # Validate inputs
    print("Validating input files...")
    validate_inputs(h5ad_path, model_dir)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load scGPT model
    print("Loading pre-trained scGPT model...")
    model, vocab, model_configs = load_scgpt_model(
        model_dir=model_dir,
        device=device,
        use_fast_transformer=use_fast_transformer,
        trainable=False
    )

    # Load h5ad data
    print(f"Loading single-cell data from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # Compute embeddings
    print("Computing cell embeddings (including all vocabulary genes and zero expression values)...")
    cell_embeddings = compute_full_embeddings(
        adata=adata,
        model=model,
        vocab=vocab,
        model_configs=model_configs,
        batch_size=batch_size,
        device=device
    )

    # Save embeddings
    if save_h5ad:
        print(f"Saving embeddings back to h5ad file: {h5ad_path}")
        adata.obsm["X_scGPT"] = cell_embeddings
        adata.write_h5ad(h5ad_path, compression="gzip")

    if output_npy is not None:
        output_npy = Path(output_npy)
        print(f"Saving embeddings as numpy file: {output_npy}")
        np.save(output_npy, cell_embeddings)

    print(f"Successfully computed embeddings for {len(cell_embeddings)} cells!")
    print(f"Embedding dimension: {cell_embeddings.shape[1]}")

def main():
    """
    Main entry point for command line interface.
    """
    parser = argparse.ArgumentParser(
        description="Compute scGPT cell embeddings from scRNA-seq data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "h5ad_path",
        help="Path to input h5ad file (required, must be specified in command line)"
    )

    parser.add_argument(
        "--model-dir",
        default=".",
        help="Path to directory containing scGPT model files (args.json, best_model.pt, vocab.json)"
    )

    parser.add_argument(
        "--output-npy",
        default=None,
        help="Optional path to save embeddings as numpy file"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation"
    )

    parser.add_argument(
        "--no-save-h5ad",
        action="store_true",
        help="Disable saving embeddings back to the original h5ad file"
    )

    parser.add_argument(
        "--fast-transformer",
        action="store_true",
        help="Enable flash-attention fast transformer (requires flash-attn library)"
    )

    args = parser.parse_args()

    # Run pipeline
    run_embedding_pipeline(
        h5ad_path=args.h5ad_path,
        model_dir=args.model_dir,
        output_npy=args.output_npy,
        save_h5ad=not args.no_save_h5ad,
        batch_size=args.batch_size,
        use_fast_transformer=args.fast_transformer
    )

if __name__ == "__main__":
    main()
