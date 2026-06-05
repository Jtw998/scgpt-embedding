"""
Benchmark scGPT fine-tuning: compare clustering metrics before vs. after SimCLR fine-tuning.
Uses a random subset for speed, streaming metrics per epoch.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm

from scgpt_embedding import (
    GeneVocab,
    SparseGeneDataset,
    DataCollator,
    TransformerModel,
    load_scgpt_model,
    get_device,
    compute_full_embeddings,
    simclr_loss,
    clustering_metrics,
    embedding_stats,
)

# ======================================================================
# Per-epoch fine-tuning with metric streaming
# ======================================================================

def run_benchmark(
    h5ad_path: str,
    model_dir: str,
    output_dir: str,
    subset: int = 50000,
    epochs: int = 5,
    batch_size: int = 64,
    embed_batch_size: int = 32,
    lr: float = 1e-4,
    temperature: float = 0.1,
    dropout_rate: float = 0.3,
    proj_dim: int = 128,
    max_seq_len: int = 1200,
    max_cluster_samples: int = 20000,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    h5ad_path = Path(h5ad_path)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

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

    # ---- Load model ----
    print("Loading pre-trained scGPT...")
    model, vocab, model_configs = load_scgpt_model(
        model_dir, device, use_fast_transformer=False, trainable=False
    )

    # ---- Pre-finetune embeddings ----
    print("\n" + "=" * 60)
    print("BEFORE FINETUNING")
    print("=" * 60)
    t0 = time.time()
    emb_pre = compute_full_embeddings(
        adata, model, vocab, model_configs,
        batch_size=embed_batch_size, device=device,
    )
    t_pre = time.time() - t0

    metrics_pre_cluster = clustering_metrics(emb_pre, cell_type_labels, max_samples=max_cluster_samples)
    metrics_pre_spread = embedding_stats(emb_pre, cell_type_labels)
    pre_metrics = {**metrics_pre_cluster, **metrics_pre_spread}
    print(f"  Embedding time: {t_pre:.1f}s")
    for k, v in pre_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- Metrics log for streaming ----
    log = []
    log.append({"epoch": "pre", **pre_metrics})

    # ---- Setup fine-tuning ----
    print("\n" + "=" * 60)
    print("FINE-TUNING (streaming per epoch)")
    print("=" * 60)

    # Make model trainable
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    d_model = model_configs.get("embsize", 512)
    proj_head = torch.nn.Sequential(
        torch.nn.Linear(d_model, proj_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(proj_dim, proj_dim),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()), lr=lr
    )

    # Data
    count_matrix = adata.X
    if not isinstance(count_matrix, np.ndarray):
        count_matrix = count_matrix.toarray()

    if "id_in_vocab" not in adata.var:
        gene_ids_arr = np.array([vocab[g] if g in vocab else -1 for g in adata.var_names])
        adata.var["id_in_vocab"] = gene_ids_arr
    else:
        gene_ids_arr = np.array(adata.var["id_in_vocab"])

    model_configs = dict(model_configs)
    model_configs.setdefault("max_seq_len", max_seq_len)

    dataset = SparseGeneDataset(count_matrix, gene_ids_arr, vocab, model_configs, max_seq_len)
    collator = DataCollator(
        do_padding=True, pad_token_id=vocab["<pad>"],
        pad_value=model_configs.get("pad_value", 0),
        max_length=max_seq_len, sampling=True, keep_first_n_tokens=1,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset),
        collate_fn=collator, drop_last=False, num_workers=0, pin_memory=False,
    )

    # ---- Training + selective eval (only on eval_epochs) ----
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            gene_ids = batch["gene"].to(device)
            values = batch["expr"].to(device)

            outputs = model(gene_ids, values)
            gene_tokens = outputs["sequence_output"][:, 1:, :]
            B, G, D = gene_tokens.shape

            # Valid-gene mask: only real genes, not pad tokens
            valid = (gene_ids[:, 1:] != vocab["<pad>"])               # (B, G)
            n_valid = valid.sum(dim=1).float()                         # (B,)
            n_keep_per = torch.clamp((n_valid * (1.0 - dropout_rate)).long(), min=1)
            max_keep = int(n_keep_per.max().item())

            def make_mask():
                scores = torch.rand(B, G, device=device)
                scores.masked_fill_(~valid, -1.0)
                _, idx = torch.topk(scores, max_keep, dim=1)
                col_range = torch.arange(max_keep, device=device).unsqueeze(0)
                row_mask = col_range < n_keep_per.unsqueeze(1)
                flat_row = torch.arange(B, device=device).repeat_interleave(n_keep_per)
                flat_col = idx[row_mask]
                m = torch.zeros(B, G, device=device)
                m[flat_row, flat_col] = 1.0
                return m

            mask1 = make_mask()
            mask2 = make_mask()

            z1 = (gene_tokens * mask1.unsqueeze(-1)).sum(dim=1) / n_keep_per.unsqueeze(1)
            z2 = (gene_tokens * mask2.unsqueeze(-1)).sum(dim=1) / n_keep_per.unsqueeze(1)

            h1 = proj_head(z1)
            h2 = proj_head(z2)

            loss = simclr_loss(h1, h2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batches += 1
            if n_batches % 10 == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")

        # Only run full eval on checkpoint epochs and final epoch
        eval_epochs = set(range(1, epochs + 1))
        do_eval = epoch in eval_epochs
        epoch_metrics = {"epoch": epoch, "loss": avg_loss, "eval": do_eval}

        if do_eval:
            model.eval()
            t0 = time.time()
            emb_curr = compute_full_embeddings(
                adata, model, vocab, model_configs,
                batch_size=embed_batch_size, device=device,
            )
            t_eval = time.time() - t0

            c = clustering_metrics(emb_curr, cell_type_labels, max_samples=max_cluster_samples)
            s = embedding_stats(emb_curr, cell_type_labels)
            epoch_metrics.update({"eval_time_s": t_eval, **c, **s})

            print(f"  silhouette={c['silhouette']:.4f}  db={c['davies_bouldin']:.4f}  ch={c['calinski_harabasz']:.1f}")
            print(f"  intra={s['intra_cluster_dist']:.4f}  inter={s['inter_cluster_dist']:.4f}  ratio={s['inter_intra_ratio']:.4f}")

            model.train()

        log.append(epoch_metrics)

    # ---- Save log ----
    log_path = output_dir / "benchmark_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nBenchmark log saved to {log_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    pre = log[0]
    best = max(log[1:], key=lambda x: x["silhouette"])
    last = log[-1]

    print(f"{'Metric':<30} {'Pre':>12} {'Best':>12} {'Last':>12} {'Δ (best-pre)':>14}")
    print("-" * 80)
    for key in ["silhouette", "davies_bouldin", "calinski_harabasz", "inter_intra_ratio", "intra_cluster_dist", "inter_cluster_dist"]:
        if key in pre and key in best:
            pre_v = pre[key]
            best_v = best[key]
            last_v = last.get(key, float("nan"))
            delta = best_v - pre_v
            direction = "↑" if (key in ("davies_bouldin", "intra_cluster_dist")) else "↑"
            if key == "davies_bouldin":
                direction = "↓ better"
            elif key == "intra_cluster_dist":
                direction = "↓ better"
            else:
                direction = "↑ better"
            print(f"{key:<30} {pre_v:12.4f} {best_v:12.4f} {last_v:12.4f} {delta:+14.4f}  {direction}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", dest="h5ad_path", default="PBS_PBMC_CPM.h5ad")
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--output-dir", default="./benchmark_output")
    parser.add_argument("--subset", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--max-seq-len", type=int, default=1200)
    args = parser.parse_args()
    run_benchmark(**vars(args))


if __name__ == "__main__":
    main()
