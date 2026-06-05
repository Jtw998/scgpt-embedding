"""
scGPT MLX — Cell embedding + contrastive fine-tuning on Apple Silicon using MLX.

Pure MLX — no PyTorch, no MPS fallback.
Uses `mx.fast.scaled_dot_product_attention` (Metal-accelerated flash attention).

Usage:
    # Embedding only
    python scgpt_mlx.py PBS_PBMC_CPM.h5ad --model-dir . --batch-size 16

    # Fine-tune
    python scgpt_mlx.py PBS_PBMC_CPM.h5ad --model-dir . --batch-size 16 --finetune --epochs 10

Weights are loaded from the PyTorch checkpoint once (via `torch.load`), then
all computation — including training — runs on MLX.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import scanpy as sc
from tqdm import tqdm


# ============================================================================
# Gene Vocabulary (pure Python, compatible with scgpt_embedding)
# ============================================================================

class GeneVocab:
    """Gene vocabulary — identical API to scgpt_embedding.GeneVocab."""
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

    def save_json(self, file_path: Union[Path, str]) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self._pad_token = default_token


# ============================================================================
# MLX Model Components
# ============================================================================

NEG_INF = -1e9  # safe negative infinity for attention masking


def positional_encoding(seq_len: int, d_model: int) -> mx.array:
    """Sinusoidal positional encoding: (seq_len, d_model)."""
    pos = mx.arange(seq_len).astype(mx.float32)[:, None]          # (S, 1)
    i = mx.arange(d_model // 2).astype(mx.float32)                 # (D/2,)
    div = mx.exp(i * (-math.log(10000.0) / d_model))               # (D/2,)
    sin = mx.sin(pos * div)                                        # (S, D/2)
    cos = mx.cos(pos * div)                                        # (S, D/2)
    pe = mx.stack([sin, cos], axis=-1).reshape(seq_len, d_model)  # interleave: sin,cos,sin,cos...
    return pe


class GeneEncoder(nn.Module):
    """Gene ID embedding + LayerNorm."""
    def __init__(self, ntoken: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(ntoken, d_model)
        self.enc_norm = nn.LayerNorm(d_model)
        self.padding_idx = padding_idx

    def __call__(self, x: mx.array) -> mx.array:
        return self.enc_norm(self.embedding(x))


class ValueEncoder(nn.Module):
    """Continuous expression value → embedding space, via 2-layer MLP."""
    def __init__(self, d_model: int, max_value: int = 512):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.clip(x[..., None], a_min=None, a_max=self.max_value)
        x = nn.relu(self.linear1(x))
        x = self.norm(self.linear2(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with fused QKV projection.

    Uses mlx.fast.scaled_dot_product_attention for Metal-accelerated attention.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.2):
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} must be divisible by nhead {nhead}"
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)  # fused Q, K, V
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_model)       # FFN
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None,
                 training: bool = False) -> mx.array:
        B, S, D = x.shape

        # ---- Pre-LN Self-Attention ----
        r = x
        nx = self.norm1(x)
        qkv = self.qkv_proj(nx).reshape(B, S, 3 * self.nhead * self.head_dim)
        qkv = qkv.reshape(B, S, 3, self.nhead, self.head_dim)
        # MLX sdpa expects (B, nhead, seq, D) — not (B, seq, nhead, D)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)  # (B, nhead, S, head_dim)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)
        a = a.transpose(0, 2, 1, 3).reshape(B, S, D)  # back to (B, S, D)
        x = r + (self.dropout(self.out_proj(a)) if training else self.out_proj(a))

        # ---- Pre-LN FFN ----
        r = x
        nx = self.norm2(x)
        h = nn.relu(self.linear1(nx))
        x = r + (self.dropout(self.linear2(h)) if training else self.linear2(h))

        return x


class SCGPTModel(nn.Module):
    """scGPT transformer model — gene+value embeddings → 12-layer encoder → CLS embedding."""

    def __init__(self, ntoken: int, d_model: int = 512, nhead: int = 8,
                 nlayers: int = 12, dropout: float = 0.2, pad_token_id: int = 0):
        super().__init__()
        self.gene_encoder = GeneEncoder(ntoken, d_model, padding_idx=pad_token_id)
        self.value_encoder = ValueEncoder(d_model)
        self.nlayers = nlayers
        for i in range(nlayers):
            setattr(self, f'layer{i}', TransformerBlock(d_model, nhead, dropout))
        self.d_model = d_model
        self.pad_token_id = pad_token_id

    def __call__(self, gene_ids: mx.array, values: mx.array,
                 padding_mask: Optional[mx.array] = None,
                 training: bool = False) -> Dict[str, mx.array]:
        """Forward pass.

        Args:
            gene_ids: (B, S) int32 — gene token IDs (CLS at position 0).
            values: (B, S) float32 — expression values.
            padding_mask: (B, S) bool — True = pad position.
            training: whether to apply dropout.

        Returns:
            {"cell_emb": (B, d_model), "sequence_output": (B, S, d_model)}
        """
        if padding_mask is None:
            padding_mask = gene_ids == self.pad_token_id

        B, S = gene_ids.shape

        # Gene embedding + value embedding
        x = self.gene_encoder(gene_ids) + self.value_encoder(values)

        # Positional encoding
        pe = positional_encoding(S, self.d_model)                  # (S, D)
        x = x + pe[None, :, :]                                      # broadcast: (B, S, D)

        # Attention mask: (B, S) → (B, 1, S) → additive mask
        # Boolean mask supported: True = attend, False = mask
        pad_key_mask = padding_mask[:, None, None, :]              # (B, 1, 1, S)
        attn_mask = mx.logical_not(pad_key_mask)                   # True where NOT pad

        # Transformer layers (registered as layer0..layerN−1)
        for i in range(self.nlayers):
            x = getattr(self, f'layer{i}')(x, attn_mask, training=training)

        # CLS token (position 0)
        cell_emb = x[:, 0, :]
        return {"cell_emb": cell_emb, "sequence_output": x}


# ============================================================================
# Weight Loading (PyTorch checkpoint → MLX)
# ============================================================================

def _load_pt_state_dict(pt_path: str) -> Dict[str, np.ndarray]:
    """Load a PyTorch checkpoint as numpy dict."""
    import torch
    sd = torch.load(pt_path, map_location="cpu")
    return {k: v.numpy() for k, v in sd.items()}


def _map_key(pt_key: str) -> Optional[List[str]]:
    """Map a PyTorch checkpoint key to an MLX nested parameter path.

    Returns None if the key should be skipped, or a list like ['layer0', 'norm1', 'weight'].
    Uses line-anchored replacements to avoid substring collisions
    (e.g., 'encoder.' inside 'value_encoder.' must not match).
    """
    # Skip decoder / unused weights
    if pt_key.startswith(("decoder.", "mvc_decoder.", "flag_encoder.")):
        return None

    mlx_key = pt_key

    # ^encoder. → gene_encoder.  (only at start of string)
    mlx_key = re.sub(r'^encoder\.', 'gene_encoder.', mlx_key)

    # ^transformer_encoder.layers.N → layerN
    mlx_key = re.sub(r'^transformer_encoder\.layers\.(\d+)', r'layer\1', mlx_key)

    # self_attn.Wqkv → qkv_proj; self_attn.out_proj → out_proj
    mlx_key = mlx_key.replace("self_attn.Wqkv", "qkv_proj")
    mlx_key = mlx_key.replace("self_attn.out_proj", "out_proj")

    return mlx_key.split(".")


def load_scgpt_model(
    model_dir: Union[str, Path],
    use_fast_transformer: bool = False,  # ignored; MLX always uses flash attention
) -> Tuple[SCGPTModel, GeneVocab, Dict]:
    """Load scGPT model and vocabulary from a model directory.

    Expects: args.json, vocab.json, best_model.pt
    Returns: (model, vocab, config)
    """
    model_dir = Path(model_dir)

    # ---- Config ----
    with open(model_dir / "args.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ---- Vocab ----
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for token in ["<pad>", "<cls>", "<eoc>"]:
        if token not in vocab.token2idx:
            vocab.token2idx[token] = len(vocab.token2idx)
    vocab.set_default_token("<pad>")

    # ---- Build model ----
    d_model = config.get("embsize", 512)
    nhead = config.get("nhead", 8)
    nlayers = config.get("nlayers", 12)
    dropout = config.get("dropout", 0.2)
    pad_id = vocab["<pad>"]

    model = SCGPTModel(
        ntoken=len(vocab),
        d_model=d_model,
        nhead=nhead,
        nlayers=nlayers,
        dropout=dropout,
        pad_token_id=pad_id,
    )

    # ---- Load weights ----
    pt_sd = _load_pt_state_dict(str(model_dir / "best_model.pt"))
    mlx_params = {}
    skipped = 0
    for pt_key, np_val in pt_sd.items():
        path = _map_key(pt_key)
        if path is None:
            skipped += 1
            continue
        # Build nested dict from path: ['layer0','norm1','weight'] → nested dicts
        node = mlx_params
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = mx.array(np_val)

    # Handle embedding size mismatch
    emb = mlx_params.setdefault("gene_encoder", {}).setdefault("embedding", {}).get("weight")
    if emb is not None:
        pt_vocab_size = emb.shape[0]
        our_vocab_size = len(vocab)
        if pt_vocab_size < our_vocab_size:
            extra = our_vocab_size - pt_vocab_size
            padding = mx.zeros((extra, d_model), dtype=emb.dtype)
            mlx_params["gene_encoder"]["embedding"]["weight"] = mx.concatenate([emb, padding], axis=0)
            print(f"Padded embedding: {pt_vocab_size} → {our_vocab_size} rows (special tokens)")

    model.update(mlx_params)

    print(f"Loaded scGPT model: {nlayers} layers, {d_model} dim, "
          f"{len(mlx_params)} / {len(pt_sd)} params ({skipped} skipped)")
    return model, vocab, config


# ============================================================================
# Embedding Computation
# ============================================================================

def compute_embeddings(
    adata: sc.AnnData,
    model: SCGPTModel,
    vocab: GeneVocab,
    config: Dict,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute CLS cell embeddings for all cells in adata.

    Data pipeline stays in numpy; only model forward runs on MLX.
    """
    # ---- Count matrix ----
    cm = adata.X
    if not isinstance(cm, np.ndarray):
        cm = cm.toarray()

    # ---- Gene IDs ----
    if "id_in_vocab" not in adata.var:
        gene_ids = np.array([vocab[g] if g in vocab else -1 for g in adata.var_names])
    else:
        gene_ids = np.array(adata.var["id_in_vocab"])

    valid = gene_ids >= 0
    n_dropped = len(gene_ids) - valid.sum()
    if n_dropped:
        print(f"Filtered {n_dropped} genes not in vocabulary, keeping {valid.sum()}")

    # ---- Constants ----
    max_len = config.get("max_seq_len", 1200)
    cls_id = vocab["<cls>"]
    pad_id = vocab["<pad>"]
    pad_val = config.get("pad_value", 0)
    d_model = config.get("embsize", 512)
    n_cells = len(cm)

    # ---- Pre-allocate embeddings ----
    all_embeddings = np.zeros((n_cells, d_model), dtype=np.float32)

    # ---- Batch loop ----
    for start in tqdm(range(0, n_cells, batch_size), desc="MLX embeddings"):
        end = min(start + batch_size, n_cells)
        B = end - start

        # Build per-cell gene lists (non-zero + in-vocab genes only)
        cell_genes, cell_exprs, cell_lens = [], [], []
        for i in range(B):
            row = cm[start + i]
            nz = np.nonzero(row)[0]
            nz_valid = nz[valid[nz]]
            g = gene_ids[nz_valid]
            v = row[nz_valid]

            # Insert CLS token at position 0
            g = np.insert(g, 0, cls_id)
            v = np.insert(v, 0, pad_val)

            # Truncate
            if len(g) > max_len:
                g = g[:max_len]
                v = v[:max_len]

            cell_genes.append(g)
            cell_exprs.append(v)
            cell_lens.append(len(g))

        # Pad to batch max length
        batch_max = min(max(cell_lens), max_len)
        gene_batch = np.full((B, batch_max), pad_id, dtype=np.int32)
        expr_batch = np.full((B, batch_max), pad_val, dtype=np.float32)
        for i in range(B):
            n = min(cell_lens[i], batch_max)
            gene_batch[i, :n] = cell_genes[i][:n]
            expr_batch[i, :n] = cell_exprs[i][:n]

        # Model forward (MLX)
        out = model(
            mx.array(gene_batch),
            mx.array(expr_batch),
            training=False,
        )
        emb = np.array(out["cell_emb"])

        all_embeddings[start:end] = emb

    # L2 normalize (same as PyTorch version)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    all_embeddings = all_embeddings / norms

    return all_embeddings


# ============================================================================
# SimCLR Contrastive Fine-tuning
# ============================================================================

class RNAHead(nn.Module):
    """RNA cell embedding head: mean-pool gene tokens → Linear projection (no L2 norm)."""
    def __init__(self, input_dim: int = 512, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def __call__(self, sequence_output: mx.array) -> mx.array:
        """Args: sequence_output: (B, G+1, D) — scGPT output with CLS at position 0."""
        gene_tokens = sequence_output[:, 1:, :]          # skip CLS
        return self.proj(gene_tokens.mean(axis=1))


def simclr_loss(z1: mx.array, z2: mx.array, temperature: float) -> mx.array:
    """SimCLR InfoNCE loss for two augmented views.

    Args:
        z1: (B, proj_dim) — projection of view 1
        z2: (B, proj_dim) — projection of view 2
        temperature: softmax temperature

    Returns: scalar cross-entropy loss.
    """
    z1 = z1 / (mx.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
    z2 = z2 / (mx.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
    B = z1.shape[0]
    z = mx.concatenate([z1, z2], axis=0)                # (2B, D)
    sim = z @ z.T / temperature                          # (2B, 2B)
    # Positive pairs: (i, i+B) and (i+B, i)
    sim_i_j = mx.diag(sim, k=B)
    sim_j_i = mx.diag(sim, k=-B)
    pos = mx.concatenate([sim_i_j, sim_j_i], axis=0).reshape(2 * B, 1)
    # Negatives: whole sim matrix with self-pairs set to -inf (MLX has no boolean indexing)
    neg = sim * (1.0 - mx.eye(2 * B)) + mx.eye(2 * B) * (-1e9)
    logits = mx.concatenate([pos, neg], axis=1)         # (2B, 1 + 2B)
    labels = mx.zeros(2 * B, dtype=mx.int32)
    return nn.losses.cross_entropy(logits, labels, reduction='mean')


def flatten_params(params):
    """Flatten MLX parameter dict to (leaf_values_list, key_paths_list)."""
    arrays = []
    paths = []
    def _walk(d, prefix=()):
        if isinstance(d, (list, tuple)):
            items = enumerate(d)
        elif isinstance(d, dict):
            items = sorted(d.items())
        else:
            arrays.append(d)
            paths.append(prefix)
            return
        for k, v in items:
            _walk(v, prefix + (k,))
    _walk(params)
    return arrays, paths


def unflatten_params(paths, arrays):
    """Reconstruct MLX parameter dict from paths + arrays."""
    root = {}
    for path, arr in zip(paths, arrays):
        node = root
        for part in path[:-1]:
            if isinstance(part, int) or (isinstance(node, dict) and part not in node):
                if isinstance(part, int):
                    # build list container
                    if isinstance(node, dict):
                        # Sequential 'layers' key → list value
                        pass
                if not isinstance(node, dict):
                    pass
            # Determine container type from the key
            next_key = path[path.index(part) + 1] if path.index(part) + 1 < len(path) else None
            if isinstance(next_key, int):
                # next level is integer-indexed → this level is a list
                if part not in node:
                    node[part] = []
                node = node[part]
                # extend list to fit
                while len(node) <= next_key:
                    node.append({})
            else:
                # next level is string-keyed → this level is a dict
                if part not in node:
                    node[part] = {}
                node = node[part]
        node[path[-1]] = arr
    return root


class FineTuneWrapper(nn.Module):
    """Wrapper combining scGPT model + projection head for gradient computation."""

    def __init__(self, model: SCGPTModel, proj_head: nn.Module):
        super().__init__()
        self.scgpt = model
        self.proj = proj_head


def scgpt_contrastive_train(
    model: SCGPTModel,
    vocab: GeneVocab,
    config: Dict,
    adata,
    epochs: int = 30,
    lr: float = 1e-4,
    temperature: float = 0.1,
    proj_dim: int = 128,
    dropout_rate: float = 0.3,
    batch_size: int = 32,
    max_seq_len: int = 1200,
):
    """Fine-tune scGPT with SimCLR contrastive loss using gene dropout augmentation.

    Returns: (fine_tuned_model, rna_head)
    """
    d_model = config.get("embsize", 512)
    pad_id = vocab["<pad>"]
    cls_id = vocab["<cls>"]

    # ---- Projection head ----
    proj_head = nn.Sequential(
        nn.Linear(d_model, proj_dim),
        nn.ReLU(),
        nn.Linear(proj_dim, proj_dim),
    )

    # ---- Data: pre-build padded batches ----
    cm = adata.X
    if not isinstance(cm, np.ndarray):
        cm = cm.toarray()

    if "id_in_vocab" in adata.var:
        gene_ids_arr = np.array(adata.var["id_in_vocab"])
    else:
        gene_ids_arr = np.array([vocab[g] if g in vocab else -1 for g in adata.var_names])

    valid_mask = gene_ids_arr >= 0
    n_cells = len(cm)

    print("Tokenizing cells...")
    all_genes, all_values = [], []
    for i in range(n_cells):
        row = cm[i]
        nz = np.nonzero(row)[0]
        nz_valid = nz[valid_mask[nz]]
        g = np.insert(gene_ids_arr[nz_valid], 0, cls_id)
        v = np.insert(row[nz_valid], 0, config.get("pad_value", 0))
        if len(g) > max_seq_len:
            g, v = g[:max_seq_len], v[:max_seq_len]
        all_genes.append(g)
        all_values.append(v)

    # Build padded batches
    n_batches = (n_cells + batch_size - 1) // batch_size
    batches = []
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_cells)
        indices = list(range(start, end))
        batch_max = max(len(all_genes[i]) for i in indices)
        B = len(indices)
        gene_batch = np.full((B, batch_max), pad_id, dtype=np.int32)
        expr_batch = np.zeros((B, batch_max), dtype=np.float32)
        for j, idx in enumerate(indices):
            n = len(all_genes[idx])
            gene_batch[j, :n] = all_genes[idx]
            expr_batch[j, :n] = all_values[idx]
        batches.append((mx.array(gene_batch), mx.array(expr_batch)))

    print(f"  {n_cells} cells → {n_batches} batches")

    # ---- Prepare optimizer ----
    model_optimizer = optim.Adam(learning_rate=lr)
    head_optimizer = optim.SGD(learning_rate=lr)

    def train_step(model_params, proj_params, gene_batch, expr_batch):
        model.update(model_params)
        proj_head.update(proj_params)

        out = model(gene_batch, expr_batch, training=True)
        seq = out["sequence_output"]
        gene_tokens = seq[:, 1:, :]                       # (B, G, D)
        B, G, D = gene_tokens.shape

        valid = gene_batch[:, 1:] != pad_id               # (B, G)
        n_valid = valid.sum(axis=1)                       # (B,)
        n_keep = mx.clip((n_valid * (1.0 - dropout_rate)).astype(mx.int32), 1, None)
        max_keep = int(n_keep.max().item())

        def make_mask():
            scores = mx.random.uniform(shape=(B, G))
            scores = mx.where(valid, scores, mx.full(scores.shape, float('-inf')))
            idx_all = mx.argpartition(scores, G - max_keep, axis=1)
            idx = idx_all[:, G - max_keep:]               # (B, max_keep)
            m = mx.zeros((B, G))
            for i in range(B):
                m[i, idx[i, :]] = 1.0
            return m

        mask1 = make_mask()
        mask2 = make_mask()

        z1 = (gene_tokens * mask1[:, :, None]).sum(axis=1) / max_keep
        z2 = (gene_tokens * mask2[:, :, None]).sum(axis=1) / max_keep

        h1 = proj_head(z1)
        h2 = proj_head(z2)
        return simclr_loss(h1, h2, temperature)

    # ---- Training loop ----
    grad_fn = mx.value_and_grad(train_step, argnums=[0, 1])

    for epoch in range(epochs):
        epoch_loss = 0.0

        for gene_b, expr_b in tqdm(batches, desc=f"MLX epoch {epoch+1}/{epochs}"):
            (loss_val, grads) = grad_fn(
                model.parameters(), proj_head.parameters(), gene_b, expr_b
            )

            model_optimizer.update(model, grads[0])
            head_optimizer.update(proj_head, grads[1])
            mx.eval(model.parameters(), model_optimizer.state,
                    proj_head.parameters(), head_optimizer.state)
            epoch_loss += loss_val.item()

        avg = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs} — avg loss: {avg:.4f}")

    # ---- Build RNAHead ----
    rna_head = RNAHead(input_dim=d_model, output_dim=d_model)
    print("Fine-tuning complete.")
    return model, rna_head


def save_mlx_weights(model: SCGPTModel, rna_head: RNAHead, output_dir: Path):
    """Save fine-tuned model + RNA head weights as .npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, module in [("finetuned_model", model), ("rna_head", rna_head)]:
        arrays, paths = flatten_params(module.parameters())
        flat = {}
        for p, a in zip(paths, arrays):
            flat[".".join(p)] = np.array(a)
        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path, **flat)
        print(f"Saved {name} → {out_path}")


# ============================================================================
# Pipeline
# ============================================================================

def run_embedding_pipeline(
    h5ad_path: Union[str, Path],
    model_dir: Union[str, Path] = ".",
    output_npy: Optional[Union[str, Path]] = None,
    save_h5ad: bool = True,
    batch_size: int = 32,
) -> None:
    """End-to-end: load model → compute embeddings → save."""
    h5ad_path = Path(h5ad_path)
    model_dir = Path(model_dir)

    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad not found: {h5ad_path}")
    for f_name in ["args.json", "best_model.pt", "vocab.json"]:
        if not (model_dir / f_name).exists():
            raise FileNotFoundError(f"Missing: {model_dir / f_name}")

    print(f"MLX default device: {mx.default_device()}")

    # Load
    print("Loading scGPT model...")
    model, vocab, config = load_scgpt_model(model_dir)
    mx.eval(model.parameters())  # materialize

    # Data
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # Embed
    embeddings = compute_embeddings(adata, model, vocab, config, batch_size=batch_size)

    # Save
    if save_h5ad:
        adata.obsm["X_scGPT"] = embeddings
        out_path = h5ad_path.with_name(f"{h5ad_path.stem}_scGPT.h5ad")
        adata.write_h5ad(out_path)
        print(f"Saved: {out_path}")

    if output_npy:
        np.save(output_npy, embeddings)
        print(f"Saved: {output_npy}")

    print(f"Done — {len(embeddings)} embeddings, shape {embeddings.shape}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="scGPT MLX — embedding + contrastive fine-tuning on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("h5ad", help="Path to .h5ad file")
    parser.add_argument("--model-dir", default=".", help="Directory with args.json, best_model.pt, vocab.json")
    parser.add_argument("--output-npy", default=None, help="Save embeddings as .npy")
    parser.add_argument("--no-save-h5ad", action="store_true", help="Don't write embeddings to h5ad")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    # Fine-tuning flags
    parser.add_argument("--finetune", action="store_true", help="Enable contrastive fine-tuning")
    parser.add_argument("--output-dir", default="./finetuned_mlx", help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="SimCLR temperature")
    parser.add_argument("--proj-dim", type=int, default=128, help="Projection head output dim")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Gene dropout fraction")
    parser.add_argument("--max-seq-len", type=int, default=1200, help="Max sequence length")

    args = parser.parse_args()

    # ---- Load model ----
    print("Loading scGPT model...")
    model, vocab, config = load_scgpt_model(args.model_dir)
    mx.eval(model.parameters())

    # ---- Load data ----
    print(f"Loading {args.h5ad}...")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # ---- Embeddings (before fine-tuning) ----
    if not args.finetune:
        embeddings = compute_embeddings(adata, model, vocab, config, batch_size=args.batch_size)
        if not args.no_save_h5ad:
            adata.obsm["X_scGPT"] = embeddings
            out_path = Path(args.h5ad).with_name(f"{Path(args.h5ad).stem}_scGPT.h5ad")
            adata.write_h5ad(out_path)
            print(f"Saved: {out_path}")
        if args.output_npy:
            np.save(args.output_npy, embeddings)
            print(f"Saved: {args.output_npy}")
        print(f"Done — {len(embeddings)} embeddings, shape {embeddings.shape}")

    # ---- Fine-tuning ----
    if args.finetune:
        print(f"\n{'='*60}")
        print(f"MLX Contrastive Fine-tuning")
        print(f"{'='*60}")
        print(f"  epochs: {args.epochs}")
        print(f"  lr: {args.lr}")
        print(f"  temperature: {args.temperature}")
        print(f"  dropout_rate: {args.dropout_rate}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  max_seq_len: {args.max_seq_len}")

        model, rna_head = scgpt_contrastive_train(
            model=model,
            vocab=vocab,
            config=config,
            adata=adata,
            epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            proj_dim=args.proj_dim,
            dropout_rate=args.dropout_rate,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )

        save_mlx_weights(model, rna_head, Path(args.output_dir))

        # Post-finetune embeddings
        print("\nComputing post-finetune embeddings...")
        embeddings = compute_embeddings(adata, model, vocab, config, batch_size=args.batch_size)
        if not args.no_save_h5ad:
            adata.obsm["X_scGPT"] = embeddings
            out_path = Path(args.h5ad).with_name(f"{Path(args.h5ad).stem}_scGPT_ft.h5ad")
            adata.write_h5ad(out_path)
            print(f"Saved: {out_path}")
        print(f"Done — {len(embeddings)} embeddings, shape {embeddings.shape}")


if __name__ == "__main__":
    main()
