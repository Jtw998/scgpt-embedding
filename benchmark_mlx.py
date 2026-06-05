"""
Benchmark MLX fine-tuning: compare clustering metrics before vs. after SimCLR.
Pure MLX — no PyTorch, no MPS fallback.

Usage:
    python benchmark_mlx.py --subset 5000 --epochs 10 --batch-size 16
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import scanpy as sc
from tqdm import tqdm

from scgpt_mlx import (
    GeneVocab, SCGPTModel, load_scgpt_model, compute_embeddings,
    simclr_loss, flatten_params, RNAHead,
)
from scgpt_embedding import clustering_metrics, embedding_stats


def run_benchmark(
    h5ad_path: str = "PBS_PBMC_CPM.h5ad",
    model_dir: str = ".",
    output_dir: str = "./benchmark_mlx_output",
    subset: int = 5000,
    epochs: int = 10,
    batch_size: int = 16,
    embed_batch_size: int = 16,
    lr: float = 1e-4,
    temperature: float = 0.1,
    dropout_rate: float = 0.3,
    proj_dim: int = 128,
    max_seq_len: int = 1200,
    max_cluster_samples: int = 20000,
    seed: int = 42,
):
    np.random.seed(seed)
    mx.random.seed(seed)

    h5ad_path = Path(h5ad_path)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print("Loading scGPT model...")
    model, vocab, config = load_scgpt_model(model_dir)
    mx.eval(model.parameters())

    d_model = config.get("embsize", 512)
    pad_id = vocab["<pad>"]
    cls_id = vocab["<cls>"]

    # ---- Load data ----
    print(f"Loading {h5ad_path}...")
    adata_full = sc.read_h5ad(h5ad_path)
    print(f"Full data: {adata_full.n_obs} cells, {adata_full.n_vars} genes")

    if subset and subset < adata_full.n_obs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata_full.n_obs, subset, replace=False)
        adata = adata_full[idx].copy()
        print(f"Subsampled to {adata.n_obs} cells")
    else:
        adata = adata_full

    cell_type_labels = adata.obs["cell_type"].values.astype(str)

    # ---- Pre-finetune embeddings ----
    print("\n" + "=" * 60)
    print("BEFORE FINETUNING (MLX)")
    print("=" * 60)
    t0 = time.time()
    emb_pre = compute_embeddings(adata, model, vocab, config, batch_size=embed_batch_size)
    t_pre = time.time() - t0

    pre_cluster = clustering_metrics(emb_pre, cell_type_labels, max_samples=max_cluster_samples)
    pre_spread = embedding_stats(emb_pre, cell_type_labels)
    pre_metrics = {"embed_time_s": round(t_pre, 1), **pre_cluster, **pre_spread}
    print(f"  Embedding time: {t_pre:.1f}s")
    for k, v in pre_metrics.items():
        print(f"  {k}: {v:.4f}")

    log = [{"epoch": "pre", **pre_metrics}]

    # ---- Setup fine-tuning ----
    print("\n" + "=" * 60)
    print("FINE-TUNING (MLX, streaming per epoch)")
    print("=" * 60)

    proj_head = nn.Sequential(
        nn.Linear(d_model, proj_dim),
        nn.ReLU(),
        nn.Linear(proj_dim, proj_dim),
    )

    model_optimizer = optim.Adam(learning_rate=lr)
    head_optimizer = optim.SGD(learning_rate=lr)

    # ---- Pre-tokenize ----
    cm = adata.X
    if not isinstance(cm, np.ndarray):
        cm = cm.toarray()

    if "id_in_vocab" in adata.var:
        gene_ids_arr = np.array(adata.var["id_in_vocab"])
    else:
        gene_ids_arr = np.array([vocab[g] if g in vocab else -1 for g in adata.var_names])

    valid_mask = gene_ids_arr >= 0
    n_cells = len(cm)

    print("Tokenizing...")
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
        bm = max(len(all_genes[i]) for i in indices)
        B = len(indices)
        gb = np.full((B, bm), pad_id, dtype=np.int32)
        eb = np.zeros((B, bm), dtype=np.float32)
        for j, idx in enumerate(indices):
            n = len(all_genes[idx])
            gb[j, :n] = all_genes[idx]
            eb[j, :n] = all_values[idx]
        batches.append((mx.array(gb), mx.array(eb)))

    print(f"  {n_cells} cells → {n_batches} batches")

    # ---- Train step (shared closure) ----
    def train_step(model_params, proj_params, gene_batch, expr_batch):
        model.update(model_params)
        proj_head.update(proj_params)
        out = model(gene_batch, expr_batch, training=True)
        tokens = out["sequence_output"][:, 1:, :]
        B, G, D = tokens.shape
        valid = gene_batch[:, 1:] != pad_id
        n_valid = valid.sum(axis=1)
        n_keep = mx.clip((n_valid * (1.0 - dropout_rate)).astype(mx.int32), 1, None)
        max_keep = int(n_keep.max().item())

        def make_mask():
            scores = mx.random.uniform(shape=(B, G))
            scores = mx.where(valid, scores, mx.full(scores.shape, float('-inf')))
            idx_all = mx.argpartition(scores, G - max_keep, axis=1)
            idx = idx_all[:, G - max_keep:]
            m = mx.zeros((B, G))
            for i in range(B):
                m[i, idx[i, :]] = 1.0
            return m

        mask1, mask2 = make_mask(), make_mask()
        z1 = (tokens * mask1[:, :, None]).sum(axis=1) / max_keep
        z2 = (tokens * mask2[:, :, None]).sum(axis=1) / max_keep
        return simclr_loss(proj_head(z1), proj_head(z2), temperature)

    grad_fn = mx.value_and_grad(train_step, argnums=[0, 1])

    # ---- Training loop with per-epoch eval ----
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        pbar = tqdm(batches, desc=f"MLX epoch {epoch}/{epochs}")
        for bi, (gene_b, expr_b) in enumerate(pbar):
            (loss_val, grads) = grad_fn(
                model.parameters(), proj_head.parameters(), gene_b, expr_b
            )
            model_optimizer.update(model, grads[0])
            head_optimizer.update(proj_head, grads[1])
            mx.eval(model.parameters(), model_optimizer.state,
                    proj_head.parameters(), head_optimizer.state)
            epoch_loss += loss_val.item()
            if bi % 5 == 0:
                pbar.set_postfix({"loss": f"{loss_val.item():.4f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")

        # ---- Post-epoch eval ----
        t0 = time.time()
        emb_curr = compute_embeddings(adata, model, vocab, config, batch_size=embed_batch_size)
        t_eval = time.time() - t0

        c = clustering_metrics(emb_curr, cell_type_labels, max_samples=max_cluster_samples)
        s = embedding_stats(emb_curr, cell_type_labels)
        epoch_metrics = {"epoch": epoch, "loss": round(avg_loss, 4), "eval_time_s": round(t_eval, 1), **c, **s}
        log.append(epoch_metrics)

        print(f"  silhouette={c['silhouette']:.4f}  db={c['davies_bouldin']:.4f}  ch={c['calinski_harabasz']:.1f}")
        print(f"  intra={s['intra_cluster_dist']:.4f}  inter={s['inter_cluster_dist']:.4f}  ratio={s['inter_intra_ratio']:.4f}")

    # ---- Save log ----
    log_path = output_dir / "benchmark_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nBenchmark log saved to {log_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY (MLX)")
    print("=" * 60)
    pre = log[0]
    best = max(log[1:], key=lambda x: x.get("silhouette", float("-inf")))
    last = log[-1]

    print(f"{'Metric':<30} {'Pre':>12} {'Best':>12} {'Last':>12} {'Δ (best-pre)':>14}")
    print("-" * 80)
    for key in ["silhouette", "davies_bouldin", "calinski_harabasz",
                "inter_intra_ratio", "intra_cluster_dist", "inter_cluster_dist"]:
        if key in pre and key in best:
            pre_v = pre[key]
            best_v = best[key]
            last_v = last.get(key, float("nan"))
            delta = best_v - pre_v
            if key == "davies_bouldin":
                direction = "↓ better"
            elif key == "intra_cluster_dist":
                direction = "↓ better"
            else:
                direction = "↑ better"
            print(f"{key:<30} {pre_v:12.4f} {best_v:12.4f} {last_v:12.4f} {delta:+14.4f}  {direction}")


def main():
    parser = argparse.ArgumentParser(
        description="scGPT MLX Benchmark — before/after contrastive fine-tuning on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5ad", dest="h5ad_path", default="PBS_PBMC_CPM.h5ad")
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--output-dir", default="./benchmark_mlx_output")
    parser.add_argument("--subset", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--max-seq-len", type=int, default=1200)
    args = parser.parse_args()
    run_benchmark(**vars(args))


if __name__ == "__main__":
    main()
