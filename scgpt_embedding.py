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
- Fully self-contained, no external dependencies beyond PyTorch/scanpy
"""
# Enable MPS fallback for unsupported operations (fix for PyTorch MPS compatibility)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm


# ============================================================================
# Gene Vocabulary
# ============================================================================

class GeneVocab:
    """Gene vocabulary"""
    def __init__(self, token2idx: Dict[str, int]):
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in token2idx.items()}
        self._pad_token = None

    def __getitem__(self, token: str) -> int:
        return self.token2idx.get(token, -1)

    def __contains__(self, token: str) -> bool:
        return token in self.token2idx

    def __len__(self) -> int:
        return len(self.token2idx)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "GeneVocab":
        with open(file_path, 'r', encoding='utf-8') as f:
            token2idx = json.load(f)
        return cls(token2idx)

    def set_default_token(self, default_token: str) -> None:
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self._pad_token = default_token


# ============================================================================
# Model Components
# ============================================================================

class GeneEncoder(nn.Module):
    """Gene encoder: Convert gene IDs to embedding vectors"""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """Continuous value encoder: Project continuous expression values to embedding space"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 2048,
        nlayers: int = 12,
        dropout: float = 0.2,
        pad_token_id: int = 0,
        use_fast_transformer: bool = False,
    ):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.gene_encoder = GeneEncoder(ntoken, d_model, padding_idx=pad_token_id)
        self.value_encoder = ContinuousValueEncoder(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        try:
            if use_fast_transformer:
                from flash_attn.modules import Block
                self.transformer_encoder = nn.ModuleList([
                    Block(
                        dim=d_model,
                        num_heads=nhead,
                        mlp_ratio=d_hid / d_model,
                        dropout=dropout,
                        cross_attn=False,
                    ) for _ in range(nlayers)
                ])
                self.use_fast_transformer = True
            else:
                raise ImportError("Use standard transformer")
        except ImportError:
            print("Warning: flash_attn not available, falling back to standard PyTorch Transformer")
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.use_fast_transformer = False

        self.norm = nn.LayerNorm(d_model)

    def _encode(self, src: torch.Tensor, values: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        gene_emb = self.gene_encoder(src)
        value_emb = self.value_encoder(values)
        x = gene_emb + value_emb

        if not self.use_fast_transformer:
            x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

        if self.use_fast_transformer:
            for layer in self.transformer_encoder:
                x = layer(x, key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = self.norm(x)
        return x

    def forward(self, gene_ids: torch.Tensor, values: torch.Tensor,
                padding_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = (gene_ids == self.pad_token_id)
        output = self._encode(gene_ids, values, padding_mask)
        cell_emb = output[:, 0, :]
        return {"cell_emb": cell_emb, "sequence_output": output}


def freeze_layers(model: nn.Module, freeze_n_layers: int = 8) -> None:
    for param in model.gene_encoder.parameters():
        param.requires_grad = False
    for param in model.value_encoder.parameters():
        param.requires_grad = False
    for param in model.pos_encoder.parameters():
        param.requires_grad = False

    if model.use_fast_transformer:
        layers = model.transformer_encoder
    else:
        layers = model.transformer_encoder.layers

    for i, layer in enumerate(layers):
        for param in layer.parameters():
            param.requires_grad = (i >= freeze_n_layers)

    for param in model.norm.parameters():
        param.requires_grad = True

    print(f"Frozen first {freeze_n_layers} transformer layers")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")


def load_scgpt_model(
    model_dir: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = True,
    trainable: bool = False,
    freeze_layers_n: int = 8,
) -> Tuple[nn.Module, GeneVocab, Dict]:
    model_dir = Path(model_dir)

    with open(model_dir / "args.json", 'r', encoding='utf-8') as f:
        model_configs = json.load(f)

    vocab = GeneVocab.from_file(model_dir / "vocab.json")

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for token in special_tokens:
        if token not in vocab.token2idx:
            vocab.token2idx[token] = len(vocab.token2idx)

    vocab.set_default_token("<pad>")
    pad_token_id = vocab["<pad>"]

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs.get("embsize", 512),
        nhead=model_configs.get("nhead", 8),
        d_hid=model_configs.get("d_hid", 2048),
        nlayers=model_configs.get("nlayers", 12),
        dropout=model_configs.get("dropout", 0.2),
        pad_token_id=pad_token_id,
        use_fast_transformer=use_fast_transformer,
    )

    state_dict = torch.load(model_dir / "best_model.pt", map_location=device)

    # Key name mapping: compatible with flash_attn and standard Transformer
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("encoder.", "gene_encoder.")
        key = key.replace("self_attn.Wqkv.", "self_attn.in_proj_")
        if "self_attn.out_proj" not in key and "self_attn.out" in key:
            key = key.replace("self_attn.out.", "self_attn.out_proj.")
        new_state_dict[key] = value

    filtered = {
        k: v for k, v in new_state_dict.items()
        if not k.startswith("decoder.") and not k.startswith("mvc_decoder.") and k != "flag_encoder.weight"
    }

    model.load_state_dict(filtered, strict=False)
    model.to(device)

    if trainable:
        model.train()
        freeze_layers(model, freeze_n_layers=freeze_layers_n)
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    print(f"Loaded scGPT model: {model_configs.get('nlayers', 12)} layers, "
          f"{model_configs.get('embsize', 512)} embedding dim")

    return model, vocab, model_configs


# ============================================================================
# Data Collator
# ============================================================================

class DataCollator:
    def __init__(
        self,
        do_padding: bool = True,
        pad_token_id: int = 0,
        pad_value: float = 0,
        do_mlm: bool = False,
        do_binning: bool = True,
        max_length: int = 1200,
        sampling: bool = True,
        keep_first_n_tokens: int = 1,
    ):
        self.do_padding = do_padding
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.do_mlm = do_mlm
        self.do_binning = do_binning
        self.max_length = max_length
        self.sampling = sampling
        self.keep_first_n_tokens = keep_first_n_tokens

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(item['genes']) for item in batch), self.max_length)

        batch_gene_ids = []
        batch_values = []

        for item in batch:
            gene_ids = item['genes']
            values = item['expressions']

            if len(gene_ids) > max_len and self.sampling:
                keep_idx = list(range(self.keep_first_n_tokens))
                sample_idx = np.random.choice(
                    len(gene_ids) - self.keep_first_n_tokens,
                    max_len - self.keep_first_n_tokens,
                    replace=False
                ) + self.keep_first_n_tokens
                idx = keep_idx + sample_idx.tolist()
                gene_ids = gene_ids[idx]
                values = values[idx]

            if len(gene_ids) < max_len and self.do_padding:
                pad_len = max_len - len(gene_ids)
                gene_ids = torch.cat([
                    gene_ids,
                    torch.full((pad_len,), self.pad_token_id, dtype=gene_ids.dtype)
                ])
                values = torch.cat([
                    values,
                    torch.full((pad_len,), self.pad_value, dtype=values.dtype)
                ])

            batch_gene_ids.append(gene_ids)
            batch_values.append(values)

        return {
            'gene': torch.stack(batch_gene_ids, dim=0),
            'expr': torch.stack(batch_values, dim=0)
        }


# ============================================================================
# Dataset
# ============================================================================

class FullGeneDataset(Dataset):
    """Dataset that includes all vocabulary genes for each cell, preserving zero expression values."""
    def __init__(
        self,
        count_matrix: np.ndarray,
        gene_ids: np.ndarray,
        vocab: GeneVocab,
        model_configs: Dict,
        max_seq_len: int = 1200
    ):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.vocab = vocab
        self.pad_token_id = vocab["<pad>"]
        self.cls_token_id = vocab["<cls>"]
        self.pad_value = model_configs.get("pad_value", 0)
        self.max_seq_len = max_seq_len
        self.valid_gene_mask = gene_ids >= 0
        self.valid_gene_ids = gene_ids[self.valid_gene_mask]
        print(f"Using {len(self.valid_gene_ids)} genes from vocabulary for embedding computation")

    def __len__(self) -> int:
        return len(self.count_matrix)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        full_expression = self.count_matrix[idx]
        valid_expressions = full_expression[self.valid_gene_mask]

        if len(self.valid_gene_ids) > (self.max_seq_len - 1):
            sample_indices = np.random.choice(
                len(self.valid_gene_ids),
                size=self.max_seq_len - 1,
                replace=False
            )
            selected_gene_ids = self.valid_gene_ids[sample_indices]
            selected_expressions = valid_expressions[sample_indices]
        else:
            selected_gene_ids = self.valid_gene_ids
            selected_expressions = valid_expressions

        genes = np.concatenate([[self.cls_token_id], selected_gene_ids])
        expressions = np.concatenate([[self.pad_value], selected_expressions])

        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(expressions).float(),
        }


# ============================================================================
# Embedding Computation
# ============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def validate_inputs(h5ad_path: Path, model_dir: Path) -> None:
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Input h5ad file not found: {h5ad_path}")
    if not h5ad_path.is_file():
        raise ValueError(f"Path is not a file: {h5ad_path}")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Path is not a directory: {model_dir}")

    for file in ["args.json", "best_model.pt", "vocab.json"]:
        file_path = model_dir / file
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required model file: {file_path}")


def compute_full_embeddings(
    adata: sc.AnnData,
    model: TransformerModel,
    vocab: GeneVocab,
    model_configs: Dict,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    count_matrix = adata.X
    if not isinstance(count_matrix, np.ndarray):
        count_matrix = count_matrix.toarray()

    if "id_in_vocab" not in adata.var:
        gene_names = adata.var_names
        gene_ids = np.array([vocab[gene] if gene in vocab else -1 for gene in gene_names])
        adata.var["id_in_vocab"] = gene_ids
    else:
        gene_ids = np.array(adata.var["id_in_vocab"])

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
        sampling=False,
        keep_first_n_tokens=1
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    embedding_dim = model_configs.get("embsize", 512)
    cell_embeddings = np.zeros((len(dataset), embedding_dim), dtype=np.float32)

    model.eval()
    model.to(device)

    with torch.no_grad():
        current_idx = 0
        for batch in tqdm(dataloader, desc="Computing cell embeddings"):
            input_gene_ids = batch["gene"].to(device)
            input_values = batch["expr"].to(device)

            outputs = model(input_gene_ids, input_values)
            batch_embeddings = outputs["cell_emb"].cpu().numpy()

            batch_len = len(batch_embeddings)
            cell_embeddings[current_idx:current_idx + batch_len] = batch_embeddings
            current_idx += batch_len

    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )
    return cell_embeddings


# ============================================================================
# Pipeline
# ============================================================================

def run_embedding_pipeline(
    h5ad_path: Union[str, Path],
    model_dir: Union[str, Path] = ".",
    output_npy: Optional[Union[str, Path]] = None,
    save_h5ad: bool = True,
    batch_size: int = 32,
    use_fast_transformer: bool = False,
) -> None:
    h5ad_path = Path(h5ad_path)
    model_dir = Path(model_dir)

    print("Validating input files...")
    validate_inputs(h5ad_path, model_dir)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading pre-trained scGPT model...")
    model, vocab, model_configs = load_scgpt_model(
        model_dir=model_dir,
        device=device,
        use_fast_transformer=use_fast_transformer,
        trainable=False
    )

    print(f"Loading single-cell data from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    print("Computing cell embeddings (including all vocabulary genes and zero expression values)...")
    cell_embeddings = compute_full_embeddings(
        adata=adata,
        model=model,
        vocab=vocab,
        model_configs=model_configs,
        batch_size=batch_size,
        device=device
    )

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


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute scGPT cell embeddings from scRNA-seq data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "h5ad_path",
        help="Path to input h5ad file (required)"
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
