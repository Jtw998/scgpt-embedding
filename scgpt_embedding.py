#!/usr/bin/env python3
"""
Production-grade scGPT Embedding Computation Tool
Native implementation fully aligned with official scGPT preprocessing and inference logic
Loads pre-trained weights and vocabulary from the same directory by default
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List

# scGPT imports
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch

# -------------------------- Configuration --------------------------
# Default file names (expected in the same directory as this script)
DEFAULT_VOCAB_FILENAME = "vocab.json"
DEFAULT_MODEL_FILENAME = "best_model.pt"
DEFAULT_CONFIG_FILENAME = "args.json"

# Official pre-trained model default parameters (for whole-human model)
DEFAULT_MODEL_CONFIG = {
    "embsize": 512,
    "nheads": 8,
    "d_hid": 512,
    "nlayers": 12,
    "n_bins": 51,
    "input_emb_style": "category",
    "pad_token": "<pad>",
    "pad_value": -2,
    "max_seq_len": 1201,  # 1200 HVGs + 1 <CLS> token
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------- Core Implementation --------------------------
class scGPTEmbeddingEngine:
    """
    Production-grade scGPT embedding computation engine
    Fully compatible with official pre-trained weights
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        freeze_weights: bool = True,
        use_fast_attention: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        """
        Initialize scGPT embedding engine

        Args:
            model_dir: Directory containing pre-trained model files, defaults to current script directory
            vocab_path: Path to vocabulary JSON file, defaults to model_dir/vocab.json
            model_path: Path to pre-trained model checkpoint, defaults to model_dir/best_model.pt
            config_path: Path to model configuration JSON, defaults to model_dir/args.json
            device: Computation device (auto-detect if not specified: cuda > mps > cpu)
            freeze_weights: Freeze model weights for inference (disable gradient computation)
            use_fast_attention: Use FlashAttention for faster inference (requires flash-attn package)
        """
        # Resolve paths
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir).resolve()

        self.vocab_path = Path(vocab_path).resolve() if vocab_path else model_dir / DEFAULT_VOCAB_FILENAME
        self.model_path = Path(model_path).resolve() if model_path else model_dir / DEFAULT_MODEL_FILENAME
        self.config_path = Path(config_path).resolve() if config_path else model_dir / DEFAULT_CONFIG_FILENAME

        # Validate required files exist
        self._validate_file_dependencies()

        # Auto-detect device
        self.device = self._resolve_device(device)
        logger.info(f"Using computation device: {self.device}")

        # Load configuration
        self.config = self._load_model_config()
        self.config["use_fast_attention"] = use_fast_attention

        # Custom sequence length override
        if max_seq_len is not None:
            if max_seq_len < 2:
                raise ValueError("max_seq_len must be at least 2 (1 gene + 1 CLS token)")
            self.config["max_seq_len"] = max_seq_len
            logger.warning(
                f"Using custom maximum sequence length: {max_seq_len}\n"
                "⚠️  This differs from the pre-trained configuration (1201) and may degrade embedding performance.\n"
                "Only use custom lengths if you are fine-tuning the model on your own data."
            )

        # Load vocabulary
        self.vocab = self._load_gene_vocabulary()

        # Load pre-trained model
        self.model = self._load_pretrained_model(freeze_weights=freeze_weights)

        logger.info("scGPT embedding engine initialized successfully")

    def _validate_file_dependencies(self) -> None:
        """Validate that all required model files exist"""
        missing_files: List[str] = []
        if not self.vocab_path.exists():
            missing_files.append(str(self.vocab_path))
        if not self.model_path.exists():
            missing_files.append(str(self.model_path))

        if missing_files:
            raise FileNotFoundError(
                f"Missing required dependencies: {', '.join(missing_files)}\n"
                "Download pre-trained model files from: https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo"
            )

    def _resolve_device(self, device_override: Optional[str]) -> str:
        """Resolve computation device with priority: cuda > mps > cpu"""
        if device_override:
            return device_override
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"

    def _load_model_config(self) -> Dict:
        """Load model configuration, fall back to defaults if config file missing"""
        if self.config_path.exists():
            logger.info(f"Loading model configuration from {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Merge with default config for missing keys
            for key, default_value in DEFAULT_MODEL_CONFIG.items():
                if key not in config:
                    config[key] = default_value
                    logger.warning(f"Missing config key '{key}', using default value: {default_value}")
            return config
        else:
            logger.warning(f"Config file {self.config_path} not found, using default pre-trained model configuration")
            return DEFAULT_MODEL_CONFIG.copy()

    def _load_gene_vocabulary(self) -> GeneVocab:
        """Load gene vocabulary and ensure special tokens are present"""
        logger.info(f"Loading gene vocabulary from {self.vocab_path}")
        vocab = GeneVocab.from_file(self.vocab_path)

        # Ensure required special tokens exist
        special_tokens = [self.config["pad_token"], "<cls>", "<eoc>"]
        for token in special_tokens:
            if token not in vocab:
                vocab.append_token(token)
                logger.info(f"Added missing special token: {token}")

        # Set default index to pad token
        vocab.set_default_index(vocab[self.config["pad_token"]])
        logger.info(f"Loaded vocabulary with {len(vocab)} entries")
        return vocab

    def _load_pretrained_model(self, freeze_weights: bool = True) -> TransformerModel:
        """Load pre-trained scGPT model weights"""
        logger.info(f"Loading pre-trained model from {self.model_path}")
        model = TransformerModel(
            ntokens=len(self.vocab),
            d_model=self.config["embsize"],
            nhead=self.config["nheads"],
            d_hid=self.config["d_hid"],
            nlayers=self.config["nlayers"],
            vocab=self.vocab,
            pad_token=self.config["pad_token"],
            pad_value=self.config["pad_value"],
            n_input_bins=self.config["n_bins"],
            input_emb_style=self.config["input_emb_style"],
            use_fast_transformer=self.config["use_fast_attention"],
        )

        # Load checkpoint with compatibility handling
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        state_dict = {}
        for key, value in checkpoint.items():
            # Remove distributed training prefix if present
            cleaned_key = key[7:] if key.startswith("module.") else key
            state_dict[cleaned_key] = value

        # Load weights with strict=False to ignore missing heads (e.g. classification heads not used for embedding)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            logger.debug(f"Ignored missing keys (not required for embedding): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.debug(f"Ignored unexpected keys: {load_result.unexpected_keys}")

        # Freeze weights if in inference mode
        if freeze_weights:
            for param in model.parameters():
                param.requires_grad = False

        # Move to target device and set to eval mode
        model.to(self.device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded model with {total_params / 1e6:.2f}M parameters")

        # Auto-extend positional encoding to support custom sequence lengths
        if hasattr(model, "pos_encoder"):
            current_max_pos = model.pos_encoder.pe.size(0)
            required_max_pos = self.config["max_seq_len"]
            if required_max_pos > current_max_pos:
                logger.info(f"Extending positional encoding from {current_max_pos} to {required_max_pos} positions")
                # Linear interpolation to extend positional encoding while preserving pre-trained distribution
                old_pe = model.pos_encoder.pe.data
                # Reshape for interpolation: [seq_len, 1, dim] -> [1, dim, seq_len]
                new_pe = torch.nn.functional.interpolate(
                    old_pe.permute(1, 2, 0),
                    size=required_max_pos,
                    mode="linear",
                    align_corners=False
                ).permute(2, 0, 1)  # Restore original shape: [new_seq_len, 1, dim]
                # Update model positional encoding parameter
                model.pos_encoder.pe = torch.nn.Parameter(new_pe, requires_grad=False)
                model.pos_encoder.max_len = required_max_pos
                logger.info("Positional encoding extended successfully")

        return model

    def preprocess_input_data(
        self,
        adata: sc.AnnData,
        batch_key: Optional[str] = None,
        n_highly_variable_genes: Optional[int] = None,
        min_gene_counts: int = 3,
    ) -> sc.AnnData:
        """
        Preprocess raw count data fully aligned with scGPT pre-training pipeline
        This step is critical for correct embedding results

        Args:
            adata: Input AnnData object with raw UMI counts in .X
            batch_key: Column name in .obs containing batch information (for batch-aware HVG selection)
            n_highly_variable_genes: Number of highly variable genes to use (HVGs, default: auto-match max_seq_len - 1)
            min_gene_counts: Minimum number of cells a gene must be expressed in to be kept

        Returns:
            Preprocessed AnnData with binned expression values in .layers["X_binned"]
        """
        # Validate input is raw count data
        if adata.X.min() < 0:
            raise ValueError("Input .X contains negative values, expected raw UMI count matrix")

        # Auto-set HVG count to match sequence length (reserve 1 position for CLS token)
        if n_highly_variable_genes is None:
            n_highly_variable_genes = self.config["max_seq_len"] - 1
            logger.info(f"Auto-selected {n_highly_variable_genes} highly variable genes for sequence length {self.config['max_seq_len']}")
        else:
            expected_gene_count = self.config["max_seq_len"] - 1
            if n_highly_variable_genes != expected_gene_count:
                logger.warning(
                    f"Selected {n_highly_variable_genes} HVGs but sequence length requires {expected_gene_count} genes + 1 CLS token.\n"
                    f"Sequences will be padded/truncated to {self.config['max_seq_len']} automatically."
                )

        # Create copy to avoid modifying original data
        processed_adata = adata.copy()

        # Filter genes not present in vocabulary
        initial_gene_count = processed_adata.n_vars
        processed_adata.var["in_scgpt_vocab"] = [gene in self.vocab for gene in processed_adata.var_names]
        processed_adata = processed_adata[:, processed_adata.var["in_scgpt_vocab"]].copy()
        logger.info(f"Filtered out {initial_gene_count - processed_adata.n_vars} genes not present in vocabulary")
        logger.info(f"Remaining genes: {processed_adata.n_vars}")

        if processed_adata.n_vars < 100:
            raise RuntimeError(
                f"Only {processed_adata.n_vars} genes matched to vocabulary. "
                "Verify gene names are standard HGNC symbols matching the pre-trained vocabulary."
            )

        # Run official scGPT preprocessing pipeline
        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=min_gene_counts,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            subset_hvg=n_highly_variable_genes,
            hvg_flavor="seurat_v3",
            binning=self.config["n_bins"],
            result_binned_key="X_binned",
        )

        preprocessor(processed_adata, batch_key=batch_key)
        logger.info(f"Preprocessing completed, final gene count: {processed_adata.n_vars}")
        return processed_adata

    def compute_embeddings(
        self,
        adata: sc.AnnData,
        auto_preprocess: bool = True,
        batch_key: Optional[str] = None,
        n_highly_variable_genes: Optional[int] = None,
        inference_batch_size: int = 64,
        return_gene_embeddings: bool = False,
        return_sequence_embeddings: bool = False,
        cell_embedding_obsm_key: str = "X_scGPT",
        gene_embedding_varm_key: str = "scGPT_gene_emb",
    ) -> Union[sc.AnnData, Tuple[sc.AnnData, np.ndarray, np.ndarray]]:
        """
        Compute scGPT embeddings for input single-cell data

        Args:
            adata: Input AnnData object (raw counts if auto_preprocess=True, preprocessed otherwise)
            auto_preprocess: Run preprocessing pipeline automatically (set to False if data is already preprocessed)
            batch_key: Batch column name for preprocessing (only used if auto_preprocess=True)
            n_highly_variable_genes: Number of HVGs to select (only used if auto_preprocess=True)
            inference_batch_size: Batch size for inference (adjust based on available VRAM)
            return_gene_embeddings: Return pre-trained gene embedding matrix for all vocabulary genes
            return_sequence_embeddings: Return per-token sequence embeddings from transformer output
            cell_embedding_obsm_key: Key to store cell embeddings in adata.obsm
            gene_embedding_varm_key: Key to store gene embeddings in adata.varm

        Returns:
            AnnData with cell embeddings stored in .obsm[cell_embedding_obsm_key]
            Optional additional returns: (full_gene_embedding_matrix, sequence_embedding_tensor)
        """
        # Run preprocessing if enabled
        if auto_preprocess:
            adata = self.preprocess_input_data(
                adata,
                batch_key=batch_key,
                n_highly_variable_genes=n_highly_variable_genes,
            )

        # Validate required preprocessing output exists
        if "X_binned" not in adata.layers:
            raise ValueError(
                "Missing binned expression matrix in .layers['X_binned']. "
                "Either enable auto_preprocess or run preprocess_input_data() first."
            )

        # Prepare input data
        binned_expression = adata.layers["X_binned"]
        if sc.sparse.issparse(binned_expression):
            binned_expression = binned_expression.toarray()

        gene_names = adata.var_names.tolist()
        gene_vocab_ids = np.array(self.vocab(gene_names))

        # Tokenize and pad sequences
        logger.info("Tokenizing and padding input sequences")
        tokenized_data = tokenize_and_pad_batch(
            binned_expression,
            gene_vocab_ids,
            max_len=self.config["max_seq_len"],
            vocab=self.vocab,
            pad_token=self.config["pad_token"],
            pad_value=self.config["pad_value"],
            append_cls=True,
            include_zero_gene=True,
        )

        # Move tensors to device
        input_gene_ids = tokenized_data["genes"].to(self.device)
        input_expression_values = tokenized_data["values"].to(self.device)
        padding_mask = input_gene_ids.eq(self.vocab[self.config["pad_token"]])

        # Run inference
        logger.info(f"Computing embeddings with batch size: {inference_batch_size}")
        with torch.no_grad():
            # 1. Compute cell-level embeddings (CLS token output)
            cell_embeddings = self.model.encode_batch(
                input_gene_ids,
                input_expression_values.float(),
                src_key_padding_mask=padding_mask,
                batch_size=inference_batch_size,
                time_step=0,  # Return embedding from CLS token position
                return_np=True,
            )

            # Store cell embeddings in AnnData
            adata.obsm[cell_embedding_obsm_key] = cell_embeddings
            logger.info(f"Cell embeddings stored in adata.obsm['{cell_embedding_obsm_key}'], shape: {cell_embeddings.shape}")

            # 2. Extract gene-level embeddings if requested
            full_gene_embeddings = None
            if return_gene_embeddings:
                full_gene_embeddings = self.model.encoder.embedding.weight.detach().cpu().numpy()
                # Store embeddings for genes present in current dataset
                adata.varm[gene_embedding_varm_key] = full_gene_embeddings[gene_vocab_ids]
                logger.info(f"Gene embeddings stored in adata.varm['{gene_embedding_varm_key}'], shape: {adata.varm[gene_embedding_varm_key].shape}")

            # 3. Extract per-token sequence embeddings if requested
            sequence_embeddings = None
            if return_sequence_embeddings:
                sequence_embeddings_list = []
                num_samples = input_gene_ids.shape[0]
                for i in range(0, num_samples, inference_batch_size):
                    batch_gene_ids = input_gene_ids[i:i+inference_batch_size]
                    batch_values = input_expression_values[i:i+inference_batch_size]
                    batch_mask = padding_mask[i:i+inference_batch_size]

                    transformer_output = self.model._encode(
                        batch_gene_ids,
                        batch_values,
                        src_key_padding_mask=batch_mask,
                    )
                    sequence_embeddings_list.append(transformer_output.detach().cpu().numpy())

                sequence_embeddings = np.concatenate(sequence_embeddings_list, axis=0)
                logger.info(f"Sequence embeddings computed, shape: {sequence_embeddings.shape}")

        # Return appropriate results based on request
        if return_gene_embeddings and return_sequence_embeddings:
            return adata, full_gene_embeddings, sequence_embeddings
        elif return_gene_embeddings:
            return adata, full_gene_embeddings
        elif return_sequence_embeddings:
            return adata, sequence_embeddings
        else:
            return adata

    def get_single_gene_embedding(self, gene_symbol: str) -> Optional[np.ndarray]:
        """
        Get pre-trained embedding for a single gene

        Args:
            gene_symbol: Standard HGNC gene symbol

        Returns:
            Gene embedding vector (d_model,) or None if gene not in vocabulary
        """
        if gene_symbol not in self.vocab:
            logger.warning(f"Gene '{gene_symbol}' not found in vocabulary")
            return None
        gene_id = self.vocab[gene_symbol]
        return self.model.encoder.embedding.weight[gene_id].detach().cpu().numpy()

# -------------------------- Command Line Interface --------------------------
def main():
    parser = argparse.ArgumentParser(description="scGPT Embedding Computation Tool")
    parser.add_argument("--input", "-i", required=True, help="Input AnnData file path (.h5ad)")
    parser.add_argument("--output", "-o", required=True, help="Output AnnData file path (.h5ad)")
    parser.add_argument("--model-dir", help="Directory containing pre-trained model files (default: current directory)")
    parser.add_argument("--batch-key", help="Batch column name in .obs for batch-aware preprocessing")
    parser.add_argument("--n-hvg", type=int, default=1200, help="Number of highly variable genes to use (default: 1200)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--device", help="Computation device override (cuda/mps/cpu)")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing (input data is already preprocessed)")
    parser.add_argument("--return-gene-emb", action="store_true", help="Return and store gene embeddings in .varm")
    parser.add_argument("--max-seq-len", type=int, help="Custom maximum sequence length (default: 1201, pre-trained standard)")

    args = parser.parse_args()

    # Initialize engine
    engine = scGPTEmbeddingEngine(
        model_dir=args.model_dir,
        device=args.device,
        max_seq_len=args.max_seq_len
    )

    # Load input data
    logger.info(f"Loading input data from {args.input}")
    adata = sc.read(args.input)

    # Compute embeddings
    adata = engine.compute_embeddings(
        adata,
        auto_preprocess=not args.no_preprocess,
        batch_key=args.batch_key,
        n_highly_variable_genes=args.n_hvg,
        inference_batch_size=args.batch_size,
        return_gene_embeddings=args.return_gene_emb,
    )

    # Save output
    logger.info(f"Saving results to {args.output}")
    adata.write(args.output, compression="gzip")
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
