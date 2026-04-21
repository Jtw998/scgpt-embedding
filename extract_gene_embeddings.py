"""
Extract fixed scGPT gene embeddings from pre-trained weights
Directly extracts gene embeddings from scGPT's token embedding layer without running inference on expression data
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

def extract_gene_embeddings(
    model_dir: str,
    gene_order_file: str,
    output_npy: str,
    vocab_file: str = "vocab.json",
    model_file: str = "best_model.pt",
    embedding_key: str = "encoder.token_embedding.weight"
) -> None:
    """
    Extract scGPT gene embeddings in the order of your gene list

    Args:
        model_dir: Directory containing scGPT pre-trained model files
        gene_order_file: Path to TSV/CSV file containing ordered gene list (must have 'gene_name' column)
        output_npy: Path to save output gene embeddings as npy file
        vocab_file: Name of vocabulary file in model directory
        model_file: Name of model weight file in model directory
        embedding_key: Key for embedding weights in the model state dict
    """
    # Validate input paths
    model_dir = Path(model_dir)
    gene_order_path = Path(gene_order_file)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    vocab_path = model_dir / vocab_file
    model_path = model_dir / model_file

    for path in [vocab_path, model_path, gene_order_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # 1. Load scGPT vocabulary
    print(f"Loading gene vocabulary from: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        gene_vocab = json.load(f)
    print(f"Total genes in vocabulary: {len(gene_vocab)}")

    # 2. Load pre-trained model and extract gene embedding weights
    print(f"Loading pre-trained model from: {model_path}")
    model_state = torch.load(model_path, map_location="cpu")

    if embedding_key not in model_state:
        # Try alternative key naming conventions
        alternative_keys = [k for k in model_state.keys() if "embedding.weight" in k and "gene" in k.lower()]
        if alternative_keys:
            embedding_key = alternative_keys[0]
            print(f"Using alternative embedding key: {embedding_key}")
        else:
            raise KeyError(f"Embedding key '{embedding_key}' not found in model state dict. "
                          f"Available keys with 'embedding': {[k for k in model_state.keys() if 'embedding' in k]}")

    gene_emb_weight = model_state[embedding_key].numpy()
    print(f"Loaded embedding matrix shape: {gene_emb_weight.shape} [vocab_size, embedding_dim]")

    # 3. Read ordered gene list
    print(f"Loading ordered gene list from: {gene_order_path}")
    if gene_order_path.suffix.lower() in [".tsv", ".txt"]:
        gene_df = pd.read_csv(gene_order_path, sep="\t")
    elif gene_order_path.suffix.lower() == ".csv":
        gene_df = pd.read_csv(gene_order_path)
    else:
        raise ValueError(f"Unsupported gene file format: {gene_order_path.suffix}. Use TSV or CSV.")

    if "gene_name" not in gene_df.columns:
        raise ValueError("Gene order file must contain 'gene_name' column")

    hvg_ordered = gene_df["gene_name"].tolist()
    print(f"Total genes in ordered list: {len(hvg_ordered)}")

    # 4. Extract embeddings in gene order
    print("Extracting embeddings in specified gene order...")
    emb_list = []
    matched_count = 0

    for gene in hvg_ordered:
        if gene in gene_vocab:
            emb = gene_emb_weight[gene_vocab[gene]]
            matched_count += 1
        else:
            # Initialize unmatched genes with small variance random values
            emb = np.random.randn(gene_emb_weight.shape[1]) * 0.02
        emb_list.append(emb)

    print(f"Matched {matched_count}/{len(hvg_ordered)} genes to vocabulary")
    if matched_count < len(hvg_ordered):
        print(f"{len(hvg_ordered) - matched_count} genes not found in vocabulary, initialized with random values")

    # 5. Save embeddings
    output_emb = np.stack(emb_list)
    np.save(output_npy, output_emb)
    print(f"✅ Successfully saved gene embeddings to: {output_npy}")
    print(f"Output shape: {output_emb.shape} [num_genes, embedding_dim], fully aligned with your gene order")

def main():
    parser = argparse.ArgumentParser(
        description="Extract fixed scGPT gene embeddings from pre-trained weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "model_dir",
        help="Directory containing scGPT pre-trained model files (vocab.json and best_model.pt)"
    )

    parser.add_argument(
        "gene_order_file",
        help="Path to TSV/CSV file containing ordered gene list (must have 'gene_name' column)"
    )

    parser.add_argument(
        "output_npy",
        help="Path to save output gene embeddings as npy file"
    )

    parser.add_argument(
        "--vocab-file",
        default="vocab.json",
        help="Name of vocabulary file in model directory"
    )

    parser.add_argument(
        "--model-file",
        default="best_model.pt",
        help="Name of model weight file in model directory"
    )

    args = parser.parse_args()

    extract_gene_embeddings(
        model_dir=args.model_dir,
        gene_order_file=args.gene_order_file,
        output_npy=args.output_npy,
        vocab_file=args.vocab_file,
        model_file=args.model_file
    )

if __name__ == "__main__":
    main()
