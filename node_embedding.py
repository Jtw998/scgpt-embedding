"""
Node embedding computation module: core model components and utilities for scGPT-based cell embeddings.
"""
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scanpy as sc
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from sklearn.metrics import roc_auc_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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

    def save_json(self, file_path: Union[Path, str]) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self._pad_token = default_token


# ============================================================================
# Model Components
# ============================================================================

class GeneEncoder(nn.Module):
    """Gene encoder: Convert gene IDs to embedding vectors."""
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
    """Continuous value encoder: Project continuous expression values to embedding space."""
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
    """Freeze the first N layers of the transformer encoder."""
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
    """Load pre-trained scGPT model."""
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

    # Only keep encoder weights (exclude decoder)
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
    """Data collator for batching samples."""
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
# Cell Embedding Computation
# ============================================================================

def compute_cell_embeddings(
    adata: AnnData,
    model_dir: Union[str, Path] = None,
    model: nn.Module = None,
    vocab: GeneVocab = None,
    model_configs: Dict = None,
    cell_embedding_mode: str = "cls",
    batch_size: int = 64,
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = False,
) -> np.ndarray:
    """Compute scGPT embeddings for cells."""
    if model is None:
        if model_dir is None:
            raise ValueError("Either model_dir or model must be provided")
        model, vocab, model_configs = load_scgpt_model(
            model_dir, device,
            use_fast_transformer=use_fast_transformer,
            trainable=False
        )

    assert vocab is not None
    assert model_configs is not None

    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    count_matrix = adata.X
    if not isinstance(count_matrix, np.ndarray):
        count_matrix = count_matrix.toarray()

    if "id_in_vocab" not in adata.var:
        gene_names = adata.var_names
        gene_ids = np.array([vocab[gene] if gene in vocab else -1 for gene in gene_names])
        adata.var["id_in_vocab"] = gene_ids
    else:
        gene_ids = np.array(adata.var["id_in_vocab"])

    valid_mask = gene_ids >= 0
    if np.sum(valid_mask) < len(gene_ids):
        print(f"Filtered out {len(gene_ids) - np.sum(valid_mask)} genes not in vocabulary")
        adata = adata[:, valid_mask]
        gene_ids = gene_ids[valid_mask]
        count_matrix = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    class CellEmbeddingDataset(Dataset):
        def __init__(self, count_matrix, gene_ids):
            self.count_matrix = count_matrix
            self.gene_ids = gene_ids

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = self.gene_ids[nonzero_idx]
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs.get("pad_value", 0))
            return {
                "id": idx,
                "genes": torch.from_numpy(genes).long(),
                "expressions": torch.from_numpy(values).float(),
            }

    dataset = CellEmbeddingDataset(count_matrix, gene_ids)
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab["<pad>"],
        pad_value=model_configs.get("pad_value", 0),
        do_mlm=False,
        do_binning=True,
        max_length=model_configs.get("max_length", 1200),
        sampling=True,
        keep_first_n_tokens=1,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    embedding_dim = model_configs.get("embsize", 512)
    cell_embeddings = np.zeros((len(dataset), embedding_dim), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        count = 0
        for batch in tqdm(data_loader, desc="Computing cell embeddings"):
            input_gene_ids = batch["gene"].to(device)
            input_values = batch["expr"].to(device)
            outputs = model(input_gene_ids, input_values)

            if cell_embedding_mode == "cls":
                embeddings = outputs["cell_emb"]
            else:
                raise ValueError(f"Unsupported embedding mode: {cell_embedding_mode}")

            embeddings = embeddings.cpu().numpy()
            batch_len = len(embeddings)
            cell_embeddings[count:count + batch_len] = embeddings
            count += batch_len

    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )
    return cell_embeddings


# ============================================================================
# KNN Graph Construction
# ============================================================================

def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build KNN graph from cell embeddings."""
    n_cells = embeddings.shape[0]

    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False
        print("Warning: faiss not installed, using scikit-learn (slower)")

    if use_faiss:
        embeddings = embeddings.astype(np.float32)
        if metric == "euclidean":
            index = faiss.IndexFlatL2(embeddings.shape[1])
        else:
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])

        index.add(embeddings)
        distances, indices = index.search(embeddings, k + 1)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    else:
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k, metric=metric)
        knn.fit(embeddings)
        distances, indices = knn.kneighbors()

    edge_list = []
    for i in range(n_cells):
        for j in range(k):
            edge_list.append([i, indices[i, j]])

    edge_index = np.array(edge_list).T
    distances_flat = distances.flatten()
    edge_weights = 1.0 / (distances_flat + 1e-8)
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)

    return edge_index, edge_weights


# ============================================================================
# Patient Graph Construction
# ============================================================================

def create_patient_graphs(
    adata: AnnData,
    patient_id_col: str = "patient_id",
    knn_k: int = 15,
    similarity_metric: str = "euclidean",
    graph_mode: str = "embedding",
) -> Dict[str, Dict]:
    """
    Create graph structures for each patient.

    Args:
        adata: AnnData object containing cell embeddings and patient IDs.
        patient_id_col: Patient ID column name.
        knn_k: Number of KNN neighbors.
        similarity_metric: Distance metric ("euclidean" or "cosine").
        graph_mode: Graph construction mode:
            - "embedding": use scGPT embeddings for KNN
            - "expression": use gene expression for KNN
            - "hybrid": concatenate embeddings and expression features

    Returns:
        patient_graphs: Dict mapping patient ID -> graph data dict.
    """
    if "X_scGPT" not in adata.obsm:
        raise ValueError("Cell embeddings not found, run compute_cell_embeddings first")

    if patient_id_col not in adata.obs:
        raise ValueError(f"Patient ID column '{patient_id_col}' not found in adata.obs")

    patient_ids = adata.obs[patient_id_col].values

    # Select highly variable genes for expression features
    if 'highly_variable' not in adata.var.columns:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', layer=None)
    highly_variable_mask = adata.var['highly_variable'].values
    print(f"Selected {np.sum(highly_variable_mask)} highly variable genes for feature fusion")

    patient_graphs = {}
    unique_patients = np.unique(patient_ids)

    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_indices = np.where(patient_mask)[0]
        print(f"Processing patient {patient}: {len(patient_indices)} cells")

        if len(patient_indices) < 2:
            print(f"Warning: Patient {patient} has only {len(patient_indices)} cells, skipping")
            continue

        patient_embeddings = adata.obsm["X_scGPT"][patient_mask]
        patient_expression = adata.X[patient_mask][:, highly_variable_mask]
        if sparse.issparse(patient_expression):
            patient_expression = patient_expression.toarray()

        if graph_mode == "embedding":
            knn_features = patient_embeddings
        elif graph_mode == "expression":
            knn_features = patient_expression
        else:  # hybrid
            knn_features = np.concatenate([patient_embeddings, patient_expression], axis=1)

        edge_index, edge_weights = build_knn_graph(
            knn_features, k=knn_k, metric=similarity_metric
        )

        node_features = patient_embeddings

        cell_type_labels = None
        if "cell_type" in adata.obs:
            cell_type_labels = adata.obs["cell_type"].values[patient_mask]

        prognosis_labels = None
        if "prognosis" in adata.obs:
            prognosis_labels = adata.obs["prognosis"].values[patient_mask]

        microenvironment_labels = None
        if "tme_subtype" in adata.obs:
            microenvironment_labels = adata.obs["tme_subtype"].values[patient_mask]

        patient_graphs[patient] = {
            "node_features": node_features,
            "gene_expression": patient_expression,
            "edge_index": edge_index,
            "edge_weights": edge_weights,
            "cell_indices": patient_indices,
            "cell_type_labels": cell_type_labels,
            "prognosis_labels": prognosis_labels,
            "microenvironment_labels": microenvironment_labels,
            "num_cells": len(patient_indices),
        }

    print(f"Created graphs for {len(patient_graphs)} patients")
    return patient_graphs


# ============================================================================
# End-to-End Pipeline
# ============================================================================

def process_single_cell_data(
    h5ad_path: Union[str, Path],
    model_dir: Union[str, Path],
    patient_id_col: str = "patient_id",
    knn_k: int = 15,
    batch_size: int = 64,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[AnnData, Dict]:
    """End-to-end single-cell data processing pipeline."""
    print(f"Loading single-cell data: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    print("Computing cell embeddings...")
    cell_embeddings = compute_cell_embeddings(
        adata, model_dir, batch_size=batch_size, device=device
    )
    adata.obsm["X_scGPT"] = cell_embeddings

    print("Building patient graphs...")
    patient_graphs = create_patient_graphs(
        adata, patient_id_col=patient_id_col, knn_k=knn_k
    )

    return adata, patient_graphs


# ============================================================================
# scGPT Fine Tuner
# ============================================================================

class SCGPTFineTuner:
    """
    scGPT fine-tuner for weakly-supervised prognosis prediction tasks.
    """
    def __init__(
        self,
        model: nn.Module,
        vocab: GeneVocab,
        config: Dict,
        device: torch.device,
        patient_prognosis_map: Dict[str, int],
        class_weights: torch.Tensor = None,
    ):
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = device
        self.patient_prognosis_map = patient_prognosis_map

        # Config
        self.epochs = config["scgpt"].get("epochs", 15)
        self.lr = config["scgpt"].get("learning_rate", 2e-5)
        self.weight_decay = config["scgpt"].get("weight_decay", 1e-4)
        self.contrastive_weight = config["scgpt"].get("contrastive_loss_weight", 0.2)
        self.patience = config["scgpt"].get("patience", 5)
        self.batch_size = config["scgpt"].get("batch_size", 32)
        self.mask_prob = 0.15

        # Optimizer and scheduler
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # Loss
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.projection_head = nn.Sequential(
            nn.Linear(model.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        ).to(device)

        # Augmentation
        self.noise_std = 0.05

    def _augment_batch(self, gene_ids: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Data augmentation: randomly mask genes and add expression noise."""
        batch_size, seq_len = gene_ids.shape

        mask = torch.rand(batch_size, seq_len, device=self.device) < self.mask_prob
        mask[:, 0] = False  # Keep CLS token
        gene_ids = gene_ids.masked_fill(mask, self.vocab["<pad>"])

        values = values + torch.randn_like(values) * self.noise_std
        return gene_ids, values

    def _contrastive_loss(self, cell_emb: torch.Tensor, patient_ids: List[str]) -> torch.Tensor:
        """
        Contrastive loss: cells from the same patient should be more similar,
        cells from different prognosis should be less similar.
        """
        batch_size = cell_emb.shape[0]
        if batch_size < 2:
            return 0.0

        emb_norm = F.normalize(cell_emb, p=2, dim=1)
        sim_matrix = torch.matmul(emb_norm, emb_norm.T)
        sim_matrix = sim_matrix - torch.eye(batch_size, device=self.device) * 1e12  # Exclude self

        # Same patient = positive pair
        patient_indices = {pid: i for i, pid in enumerate(set(patient_ids))}
        patient_labels = torch.tensor([patient_indices[pid] for pid in patient_ids], device=self.device)
        pos_mask = patient_labels.unsqueeze(0) == patient_labels.unsqueeze(1)
        pos_mask = pos_mask & ~torch.eye(batch_size, dtype=torch.bool, device=self.device)

        temperature = 0.1
        logits = sim_matrix / temperature
        labels = pos_mask.float()

        loss = 0.0
        n_positives = pos_mask.sum(dim=1)
        for i in range(batch_size):
            if n_positives[i] == 0:
                continue
            loss_i = F.cross_entropy(logits[i:i+1], labels[i:i+1].argmax(dim=1))
            loss += loss_i

        return loss / max(1, (n_positives > 0).sum())

    def train_epoch(self, adata: AnnData, collator: DataCollator) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        patient_ids = list(self.patient_prognosis_map.keys())
        random.shuffle(patient_ids)

        for patient_id in tqdm(patient_ids, desc="Fine-tuning scGPT"):
            patient_mask = adata.obs["patient_id"] == patient_id
            patient_cells = adata[patient_mask]
            prognosis_label = self.patient_prognosis_map[patient_id]

            features = []
            gene_names = patient_cells.var_names.tolist()

            for i in range(patient_cells.n_obs):
                expr = patient_cells.X[i].toarray().flatten()
                non_zero_idx = expr > 0
                genes = gene_names[non_zero_idx]
                expr_vals = expr[non_zero_idx]

                genes = ["<cls>"] + list(genes)
                expr_vals = [0.0] + list(expr_vals)
                gene_ids = [self.vocab.get(g, self.vocab["<pad>"]) for g in genes]

                features.append({
                    "gene": torch.tensor(gene_ids, dtype=torch.long),
                    "expr": torch.tensor(expr_vals, dtype=torch.float)
                })

            batch = collator(features)
            gene_ids = batch["gene"].to(self.device)
            expr_vals = batch["expr"].to(self.device)
            padding_mask = (gene_ids == self.vocab["<pad>"])

            gene_ids_aug, expr_vals_aug = self._augment_batch(gene_ids, expr_vals)

            outputs = self.model(gene_ids_aug, expr_vals_aug, padding_mask=padding_mask)
            cell_emb = outputs["cell_emb"]

            # Multi-instance learning: average cell embeddings -> patient embedding
            patient_emb = cell_emb.mean(dim=0, keepdim=True)
            logits = self.projection_head(patient_emb)
            label = torch.tensor([prognosis_label], dtype=torch.long, device=self.device)

            ce_loss = self.ce_loss_fn(logits, label)
            contrastive_loss = self._contrastive_loss(cell_emb, [patient_id] * len(cell_emb))

            loss = ce_loss + self.contrastive_weight * contrastive_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / n_batches if n_batches > 0 else 0.0

    def validate(self, adata: AnnData, collator: DataCollator, val_patients: List[str]) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for patient_id in val_patients:
                patient_mask = adata.obs["patient_id"] == patient_id
                patient_cells = adata[patient_mask]
                prognosis_label = self.patient_prognosis_map[patient_id]

                features = []
                gene_names = patient_cells.var_names.tolist()
                for i in range(patient_cells.n_obs):
                    expr = patient_cells.X[i].toarray().flatten()
                    non_zero_idx = expr > 0
                    genes = gene_names[non_zero_idx]
                    expr_vals = expr[non_zero_idx]

                    genes = ["<cls>"] + list(genes)
                    expr_vals = [0.0] + list(expr_vals)
                    gene_ids = [self.vocab.get(g, self.vocab["<pad>"]) for g in genes]

                    features.append({
                        "gene": torch.tensor(gene_ids, dtype=torch.long),
                        "expr": torch.tensor(expr_vals, dtype=torch.float)
                    })

                batch = collator(features)
                gene_ids = batch["gene"].to(self.device)
                expr_vals = batch["expr"].to(self.device)
                padding_mask = (gene_ids == self.vocab["<pad>"])

                outputs = self.model(gene_ids, expr_vals, padding_mask=padding_mask)
                cell_emb = outputs["cell_emb"]
                patient_emb = cell_emb.mean(dim=0, keepdim=True)
                logits = self.projection_head(patient_emb)
                pred = logits.softmax(dim=1)[:, 1].item()

                all_preds.append(pred)
                all_labels.append(prognosis_label)

        if len(set(all_labels)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(all_labels, all_preds)

        acc = sum((p > 0.5) == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        return {
            "auc": auc,
            "accuracy": acc,
            "predictions": all_preds,
            "labels": all_labels
        }

    def save_finetuned_model(self, save_path: Union[str, Path]) -> None:
        """Save the fine-tuned model."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "projection_head_state_dict": self.projection_head.state_dict(),
            "config": self.config
        }, save_path)
        print(f"Saved fine-tuned model to {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python node_embedding.py <h5ad_file_path>")
        sys.exit(1)

    h5ad_path = sys.argv[1]
    model_dir = "./blood"

    adata, patient_graphs = process_single_cell_data(
        h5ad_path=h5ad_path,
        model_dir=model_dir,
        patient_id_col="patient_id",
        knn_k=15,
        batch_size=64,
        device="cuda",
    )

    print(f"Completed! {adata.n_obs} cells, {len(patient_graphs)} patient graphs")
