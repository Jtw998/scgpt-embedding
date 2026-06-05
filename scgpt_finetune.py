"""
scGPT Contrastive Fine-tuning Script
Standalone script to fine-tune a pre-trained scGPT model with SimCLR contrastive loss
using gene dropout augmentation. Produces a fine-tuned model and an RNAHead projection
head for downstream embedding tasks.

Key features:
- SimCLR InfoNCE contrastive loss with gene dropout augmentation
- Creates two independent masked views per cell for contrastive learning
- RNAHead: mean-pool gene tokens then Linear projection (no L2 norm)
- Automatic device detection (CUDA > Apple Silicon MPS > CPU)
- Saves fine-tuned model and RNAHead for later use in embedding computation

Usage:
    python scgpt_finetune.py --h5ad data.h5ad --model-dir . --output-dir ./finetuned --epochs 30
"""
# Enable MPS fallback for unsupported operations
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

# Import reusable components from the local scgpt_embedding module
from scgpt_embedding import (
    GeneVocab,
    SparseGeneDataset,
    DataCollator,
    TransformerModel,
    load_scgpt_model,
    get_device,
    simclr_loss,
    RNAHead,
)


# ============================================================================
# Contrastive Fine-tuning
# ============================================================================

def scgpt_contrastive_train(
    model: nn.Module,
    adata: sc.AnnData,
    vocab: GeneVocab,
    model_configs: Dict,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-4,
    temperature: float = 0.1,
    proj_dim: int = 128,
    dropout_rate: float = 0.3,
    batch_size: int = 64,
    max_seq_len: int = 1200,
) -> Tuple[nn.Module, RNAHead]:
    """Fine-tune scGPT with SimCLR contrastive loss using gene dropout augmentation.

    During training, gene tokens in the sequence output are randomly masked to create
    two independent views per cell. The InfoNCE loss pushes views of the same cell
    together while pushing views of different cells apart.

    Args:
        model: Pre-trained scGPT TransformerModel.
        adata: AnnData object with expression data.
        vocab: GeneVocab instance.
        model_configs: Model configuration dictionary.
        device: torch device.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        temperature: SimCLR temperature parameter.
        proj_dim: Dimension of the projection head output.
        dropout_rate: Fraction of gene tokens to drop for each augmented view.
        batch_size: Batch size for training.
        max_seq_len: Maximum sequence length for gene token sampling.

    Returns:
        model: The fine-tuned TransformerModel (set to eval mode, gradients frozen).
        rna_head: RNAHead for producing downstream cell embeddings.
    """
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Projection head for SimCLR contrastive learning
    d_model = model_configs.get("embsize", 512)
    proj_head = nn.Sequential(
        nn.Linear(d_model, proj_dim),
        nn.ReLU(),
        nn.Linear(proj_dim, proj_dim),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()), lr=lr
    )

    # ---- Data preparation ----
    count_matrix = adata.X
    if not isinstance(count_matrix, np.ndarray):
        count_matrix = count_matrix.toarray()

    if "id_in_vocab" not in adata.var:
        gene_ids = np.array([vocab[g] if g in vocab else -1 for g in adata.var_names])
        adata.var["id_in_vocab"] = gene_ids
    else:
        gene_ids = np.array(adata.var["id_in_vocab"])

    model_configs = dict(model_configs)
    model_configs.setdefault("max_seq_len", max_seq_len)

    dataset = SparseGeneDataset(
        count_matrix=count_matrix,
        gene_ids=gene_ids,
        vocab=vocab,
        model_configs=model_configs,
        max_seq_len=max_seq_len,
    )

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab["<pad>"],
        pad_value=model_configs.get("pad_value", 0),
        max_length=max_seq_len,
        sampling=True,
        keep_first_n_tokens=1,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    model.to(device)

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Contrastive epoch {epoch+1}/{epochs}"):
            input_gene_ids = batch["gene"].to(device)
            input_values = batch["expr"].to(device)

            outputs = model(input_gene_ids, input_values)
            sequence_output = outputs["sequence_output"]            # (B, G+1, d_model)

            # Gene tokens only (skip CLS at position 0)
            gene_tokens = sequence_output[:, 1:, :]                  # (B, G, d_model)
            B, G, D = gene_tokens.shape

            # Valid-gene mask: only real genes, not pad tokens
            valid = (input_gene_ids[:, 1:] != vocab["<pad>"])       # (B, G)
            n_valid = valid.sum(dim=1).float()                       # (B,)
            n_keep_per = torch.clamp((n_valid * (1.0 - dropout_rate)).long(), min=1)  # (B,)
            max_keep = int(n_keep_per.max().item())

            def _make_contrastive_mask():
                scores = torch.rand(B, G, device=device)
                scores.masked_fill_(~valid, -1.0)
                _, idx = torch.topk(scores, max_keep, dim=1)          # (B, max_keep)
                col_range = torch.arange(max_keep, device=device).unsqueeze(0)  # (1, max_keep)
                row_mask = col_range < n_keep_per.unsqueeze(1)         # (B, max_keep)
                flat_row = torch.arange(B, device=device).repeat_interleave(n_keep_per)
                flat_col = idx[row_mask]
                m = torch.zeros(B, G, device=device)
                m[flat_row, flat_col] = 1.0
                return m

            mask1 = _make_contrastive_mask()
            mask2 = _make_contrastive_mask()

            masked1 = gene_tokens * mask1.unsqueeze(-1)             # (B, G, D)
            masked2 = gene_tokens * mask2.unsqueeze(-1)

            z1 = masked1.sum(dim=1) / n_keep_per.unsqueeze(1)       # (B, D)
            z2 = masked2.sum(dim=1) / n_keep_per.unsqueeze(1)

            h1 = proj_head(z1)                                      # (B, proj_dim)
            h2 = proj_head(z2)                                      # (B, proj_dim)

            loss = simclr_loss(h1, h2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    # ---- Post-training: build RNAHead and freeze model ----
    rna_head = RNAHead(input_dim=d_model, output_dim=d_model)
    rna_head.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, rna_head


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune scGPT with SimCLR contrastive loss using gene dropout augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--h5ad",
        required=True,
        help="Path to input h5ad file with scRNA-seq data",
    )
    parser.add_argument(
        "--model-dir",
        default=".",
        help="Path to directory containing scGPT model files (args.json, best_model.pt, vocab.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="./finetuned",
        help="Directory to save fine-tuned model (finetuned_model.pt) and RNA head (rna_head.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of contrastive fine-tuning epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="SimCLR temperature parameter",
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        default=128,
        help="Dimension of the projection head output during contrastive training",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Fraction of gene tokens to randomly mask for each augmented view",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1200,
        help="Maximum sequence length for gene token sampling",
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="Number of transformer layers to freeze (0 = unfreeze all)",
    )
    parser.add_argument(
        "--fast-transformer",
        action="store_true",
        help="Enable flash-attention fast transformer (requires flash-attn library)",
    )

    args = parser.parse_args()

    # ---- Validate inputs ----
    h5ad_path = Path(args.h5ad)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device ----
    device = get_device()
    print(f"Using device: {device}")

    # ---- Load model ----
    print("Loading pre-trained scGPT model...")
    model, vocab, model_configs = load_scgpt_model(
        model_dir=model_dir,
        device=device,
        use_fast_transformer=args.fast_transformer,
        trainable=True,
        freeze_layers_n=args.freeze_layers,
    )

    # ---- Load data ----
    print(f"Loading single-cell data from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # ---- Contrastive fine-tune ----
    print(f"\nStarting contrastive fine-tuning ({args.epochs} epochs)...")
    print(f"  Gene dropout rate: {args.dropout_rate}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Projection dim: {args.proj_dim}")
    print(f"  Batch size: {args.batch_size}")
    print()

    model, rna_head = scgpt_contrastive_train(
        model=model,
        adata=adata,
        vocab=vocab,
        model_configs=model_configs,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    # ---- Save outputs ----
    model_path = output_dir / "finetuned_model.pt"
    rna_head_path = output_dir / "rna_head.pt"

    torch.save(model.state_dict(), model_path)
    print(f"Fine-tuned model saved to: {model_path}")

    torch.save(rna_head.state_dict(), rna_head_path)
    print(f"RNA head saved to: {rna_head_path}")

    print("\nFine-tuning complete!")
    print(f"  Model: {model_path}")
    print(f"  RNA head: {rna_head_path}")


if __name__ == "__main__":
    main()
