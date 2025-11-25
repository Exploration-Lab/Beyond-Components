import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from transformer_lens import HookedTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
import os
import math
import wandb
import transformer_lens.utils as utils
import json
import hashlib
from pathlib import Path
import datetime

# Import visualization utilities
from ..utils.visualization import (
    visualize_masks as viz_masks,
    visualize_masked_singular_values as viz_masked_svs,
    plot_training_history as plot_history
)


def mask_fn(x):
    """
    Apply sigmoid to the input tensor and clamp values between 0 and 1.
    
    Args:
        x: Input tensor
    
    Returns:
        Masked tensor with values between 0 and 1
    """
    return torch.sigmoid(x)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaskedTransformerCircuit:
    def __init__(
        self,
        model: HookedTransformer,
        device: Optional[torch.device] = None,
        cache_svd: bool = True,
        mask_init_value: float = 1.0,
        svd_cache_dir: str = "svd_cache",
        force_recompute_svd: bool = False,
        mask_mlp: bool = True,
        l1_reg: bool = True,
        train_masks: Optional[List[str]] = None,
    ):
        """
        Initialize the masked transformer circuit.

        Args:
            model: HookedTransformer model
            device: Device to run on (defaults to model's device)
            cache_svd: Whether to cache SVD decompositions
            mask_init_value: Initial value for masks (1.0 = no masking)
            svd_cache_dir: Directory to save/load SVD decompositions
            force_recompute_svd: Force recomputation of SVD even if cached
            mask_mlp: Whether to mask MLP layers in addition to attention
            train_masks: List of mask types to train (options: 'QK', 'OV', 'MLP_in', 'MLP_out').
                        If None, trains all masks (default behavior).
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.cache_svd = cache_svd
        self.svd_cache_dir = Path(svd_cache_dir)
        self.force_recompute_svd = force_recompute_svd
        self.mask_mlp = mask_mlp

        # Parse and normalize train_masks argument
        if train_masks is None:
            # Default: train all masks
            self.trainable_mask_types = {'qk', 'ov', 'mlp_in', 'mlp_out'}
        else:
            # Normalize to lowercase and create set
            self.trainable_mask_types = {mask.strip().lower() for mask in train_masks}
            # Validate mask types
            valid_masks = {'qk', 'ov', 'mlp_in', 'mlp_out'}
            invalid_masks = self.trainable_mask_types - valid_masks
            if invalid_masks:
                raise ValueError(f"Invalid mask types: {invalid_masks}. Valid options: {valid_masks}")

        logger.info(f"Training masks: {self.trainable_mask_types}")

        # Create cache directory if it doesn't exist
        if self.cache_svd:
            self.svd_cache_dir.mkdir(parents=True, exist_ok=True)

        # Get model configuration
        self.cfg = model.cfg
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_model = model.cfg.d_model
        self.d_head = model.cfg.d_head
        self.d_mlp = model.cfg.d_mlp

        # Generate a unique model identifier for caching
        self.model_id = self._generate_model_id()

        self.l1_reg = l1_reg
        self.mask_init_value = mask_init_value
        # Save state dict for easier access
        self.state_dict = {name: param for name, param in model.state_dict().items()}

        # Initialize masks for each head
        self.qk_masks = nn.ParameterDict()
        self.ov_masks = nn.ParameterDict()

        # Initialize masks for MLP layers
        if self.mask_mlp:
            self.mlp_in_masks = nn.ParameterDict()
            self.mlp_out_masks = nn.ParameterDict()

        # Cache for SVD components
        self.svd_cache = {}

        # NEW: Storage for corrupted activations during forward pass
        self.corrupted_activations = None
        self.clean_last_idx = None

        # Initialize masks and optionally precompute SVD
        self._initialize_masks(init_value=mask_init_value)
        
        if cache_svd:
            self._load_or_compute_svd()
            
        # Move to device
        self.to(self.device)
        
    def _generate_model_id(self):
        """Generate a unique identifier for the model based on its configuration."""
        config_str = f"{self.cfg.model_name}_{self.n_layers}_{self.n_heads}_{self.d_model}_{self.d_head}"
        model_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{config_str}_{model_hash}"
    
    def _get_svd_filepath(self, layer: int, head: int, component: str):
        """Get the filepath for saving/loading SVD components."""
        filename = f"{self.model_id}_layer{layer}_head{head}_{component}.pt"
        return self.svd_cache_dir / filename
    
    def _get_metadata_filepath(self):
        """Get the filepath for the metadata file."""
        return self.svd_cache_dir / f"{self.model_id}_metadata.json"
    
    def _save_model_metadata(self):
        """Save model metadata for verification when loading SVD cache."""
        metadata = {
            'model_name': self.cfg.model_name,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_head': self.d_head,
            'd_mlp': self.d_mlp,
            'model_id': self.model_id,
            'creation_time': str(datetime.datetime.now())
        }
        
        with open(self._get_metadata_filepath(), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _verify_cached_metadata(self):
        """Verify that cached SVD files match current model configuration."""
        metadata_path = self._get_metadata_filepath()
        
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if key parameters match
            return (metadata['model_name'] == self.cfg.model_name and
                    metadata['n_layers'] == self.n_layers and
                    metadata['n_heads'] == self.n_heads and
                    metadata['d_model'] == self.d_model and
                    metadata['d_head'] == self.d_head and
                    metadata.get('d_mlp', 0) == self.d_mlp)
        except:
            return False
    
    def _save_svd_components(self, layer: int, head: int, component: str, U, S, Vh, original_matrix=None):
        """Save SVD components to disk."""
        filepath = self._get_svd_filepath(layer, head, component)
        
        svd_data = {
            'U': U.cpu(),
            'S': S.cpu(),
            'Vh': Vh.cpu(),
            'layer': layer,
            'head': head,
            'component': component,
        }
        
        # Optionally save the original matrix for verification
        if original_matrix is not None:
            svd_data['original_matrix'] = original_matrix.cpu()
        
        torch.save(svd_data, filepath)
        logger.info(f"Saved SVD for layer {layer}, head {head}, {component} to {filepath}")
    
    def _load_svd_components(self, layer: int, head: int, component: str):
        """Load SVD components from disk."""
        filepath = self._get_svd_filepath(layer, head, component)
        
        if not filepath.exists():
            return None
        
        try:
            svd_data = torch.load(filepath, map_location='cpu')
            
            # Move to appropriate device
            U = svd_data['U'].to(self.device)
            S = svd_data['S'].to(self.device)
            Vh = svd_data['Vh'].to(self.device)
            
            # Load original matrix if available
            original_matrix = None
            if 'original_matrix' in svd_data:
                original_matrix = svd_data['original_matrix'].to(self.device)
            
            # logger.info(f"Loaded SVD for layer {layer}, head {head}, {component} from {filepath}")
            
            return U, S, Vh, original_matrix
            
        except Exception as e:
            logger.warning(f"Failed to load SVD from {filepath}: {e}")
            return None
    
    def _initialize_masks(self, init_value: float = None, eps: float = 1e-3):
        """Initialize learnable masks for each attention head and MLP layer."""

        # Initialize attention masks
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_key = f'differential_head_{layer}_{head}'

                qk_dim = self.d_head
                ov_dim = self.d_head + 1

                # QK masks
                if 'qk' in self.trainable_mask_types:
                    if init_value is None:
                        # RANDOM initialization:
                        qk_p = torch.rand(qk_dim).clamp(eps, 1 - eps)
                        qk_logit = torch.log(qk_p / (1 - qk_p))
                    else:
                        # CONSTANT initialization
                        if 0 < init_value < 1:
                            logit_init = torch.log(torch.tensor(init_value / (1 - init_value)))
                        else:
                            logit_init = torch.tensor(-10.0 if init_value == 0 else 10.0)
                        qk_logit = torch.ones(qk_dim) * logit_init
                    self.qk_masks[head_key] = nn.Parameter(qk_logit, requires_grad=True)
                else:
                    # Not training: fix to 1.0 (logit = 10.0 gives sigmoid ≈ 1.0)
                    qk_logit = torch.ones(qk_dim) * 10.0
                    self.qk_masks[head_key] = nn.Parameter(qk_logit, requires_grad=False)

                # OV masks
                if 'ov' in self.trainable_mask_types:
                    if init_value is None:
                        # RANDOM initialization:
                        ov_p = torch.rand(ov_dim).clamp(eps, 1 - eps)
                        ov_logit = torch.log(ov_p / (1 - ov_p))
                    else:
                        # CONSTANT initialization
                        if 0 < init_value < 1:
                            logit_init = torch.log(torch.tensor(init_value / (1 - init_value)))
                        else:
                            logit_init = torch.tensor(-10.0 if init_value == 0 else 10.0)
                        ov_logit = torch.ones(ov_dim) * logit_init
                    self.ov_masks[head_key] = nn.Parameter(ov_logit, requires_grad=True)
                else:
                    # Not training: fix to 1.0
                    ov_logit = torch.ones(ov_dim) * 10.0
                    self.ov_masks[head_key] = nn.Parameter(ov_logit, requires_grad=False)
        
        # Initialize MLP masks
        if self.mask_mlp:
            for layer in range(self.n_layers):
                mlp_key = f'mlp_{layer}'

                # Dimensions for augmented matrices
                mlp_in_dim = min(self.d_model + 1, self.d_mlp)
                mlp_out_dim = min(self.d_mlp + 1, self.d_model)

                # MLP_in masks
                if 'mlp_in' in self.trainable_mask_types:
                    if init_value is None:
                        # RANDOM initialization:
                        mlp_in_p = torch.rand(mlp_in_dim).clamp(eps, 1 - eps)
                        mlp_in_logit = torch.log(mlp_in_p / (1 - mlp_in_p))
                    else:
                        # CONSTANT initialization
                        if 0 < init_value < 1:
                            logit_init = torch.log(torch.tensor(init_value / (1 - init_value)))
                        else:
                            logit_init = torch.tensor(-10.0 if init_value == 0 else 10.0)
                        mlp_in_logit = torch.ones(mlp_in_dim) * logit_init
                    self.mlp_in_masks[mlp_key] = nn.Parameter(mlp_in_logit, requires_grad=True)
                else:
                    # Not training: fix to 1.0
                    mlp_in_logit = torch.ones(mlp_in_dim) * 10.0
                    self.mlp_in_masks[mlp_key] = nn.Parameter(mlp_in_logit, requires_grad=False)

                # MLP_out masks
                if 'mlp_out' in self.trainable_mask_types:
                    if init_value is None:
                        # RANDOM initialization:
                        mlp_out_p = torch.rand(mlp_out_dim).clamp(eps, 1 - eps)
                        mlp_out_logit = torch.log(mlp_out_p / (1 - mlp_out_p))
                    else:
                        # CONSTANT initialization
                        if 0 < init_value < 1:
                            logit_init = torch.log(torch.tensor(init_value / (1 - init_value)))
                        else:
                            logit_init = torch.tensor(-10.0 if init_value == 0 else 10.0)
                        mlp_out_logit = torch.ones(mlp_out_dim) * logit_init
                    self.mlp_out_masks[mlp_key] = nn.Parameter(mlp_out_logit, requires_grad=True)
                else:
                    # Not training: fix to 1.0
                    mlp_out_logit = torch.ones(mlp_out_dim) * 10.0
                    self.mlp_out_masks[mlp_key] = nn.Parameter(mlp_out_logit, requires_grad=False)
    
    def _compute_qk_svd(self, layer: int, head: int):
        """Compute SVD for Query-Key matrix."""
        # Extract weights and biases
        W_Q = self.state_dict[f'blocks.{layer}.attn.W_Q'][head].to(self.device)
        W_K = self.state_dict[f'blocks.{layer}.attn.W_K'][head].to(self.device)
        b_Q = self.state_dict[f'blocks.{layer}.attn.b_Q'][head].to(self.device)
        b_K = self.state_dict[f'blocks.{layer}.attn.b_K'][head].to(self.device)
        
        # Construct augmented W_QK matrix
        W_QK = torch.zeros(self.d_model+1, self.d_model+1, device=self.device, dtype=W_Q.dtype)
        W_QK[0, 0] = torch.dot(b_Q, b_K)
        W_QK[0, 1:] = b_Q @ W_K.T
        W_QK[1:, 0] = W_Q @ b_K
        W_QK[1:, 1:] = W_Q @ W_K.T
        
        # Compute SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_QK, full_matrices=False)
        
        return U, S, Vh, W_QK
    
    def _compute_ov_svd(self, layer: int, head: int):
        """Compute SVD for Output-Value matrix."""
        # Extract weights and biases
        W_V = self.state_dict[f'blocks.{layer}.attn.W_V'][head].to(self.device)
        W_O = self.state_dict[f'blocks.{layer}.attn.W_O'][head].to(self.device)
        b_V = self.state_dict[f'blocks.{layer}.attn.b_V'][head].to(self.device)
        b_O = self.state_dict[f'blocks.{layer}.attn.b_O'].to(self.device)
        
        # Construct W_OV matrix
        W_OV = torch.zeros(self.d_model+1, self.d_model, device=self.device, dtype=W_V.dtype)
        b_eff = b_V @ W_O + b_O/self.n_heads
        W_OV[0] = b_eff
        W_OV[1:] = W_V @ W_O
        
        # Compute SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_OV, full_matrices=False)
        
        return U, S, Vh, W_OV
    
    def _compute_mlp_in_svd(self, layer: int):
        """Compute SVD for MLP input matrix (W_in) with bias."""
        # Extract weights and bias
        W_in = self.state_dict[f'blocks.{layer}.mlp.W_in'].to(self.device)
        b_in = self.state_dict[f'blocks.{layer}.mlp.b_in'].to(self.device)
        
        # Construct augmented W_in matrix
        W_in_aug = torch.zeros(self.d_model+1, self.d_mlp, device=self.device, dtype=W_in.dtype)
        W_in_aug[0] = b_in
        W_in_aug[1:] = W_in
        
        # Compute SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_in_aug, full_matrices=False)
        
        return U, S, Vh, W_in_aug
    
    def _compute_mlp_out_svd(self, layer: int):
        """Compute SVD for MLP output matrix (W_out) with bias."""
        # Extract weights and bias
        W_out = self.state_dict[f'blocks.{layer}.mlp.W_out'].to(self.device)
        b_out = self.state_dict[f'blocks.{layer}.mlp.b_out'].to(self.device)
        
        # Construct augmented W_out matrix
        W_out_aug = torch.zeros(self.d_mlp+1, self.d_model, device=self.device, dtype=W_out.dtype)
        W_out_aug[0] = b_out
        W_out_aug[1:] = W_out
        
        # Compute SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_out_aug, full_matrices=False)
        
        return U, S, Vh, W_out_aug
    
    def _load_or_compute_svd(self):
        """Load SVD from disk if available, otherwise compute and save."""
        logger.info("Loading or computing SVD decompositions...")
        
        # Check if we should use cached SVD
        use_cache = (not self.force_recompute_svd and 
                    self._verify_cached_metadata())
        
        if not use_cache:
            logger.info("Computing SVD from scratch...")
            self._save_model_metadata()
        
        # Process attention heads
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_key = f'differential_head_{layer}_{head}'
                
                # QK components
                qk_cache_key = f"{head_key}_qk"
                qk_loaded = False
                
                if use_cache:
                    loaded_data = self._load_svd_components(layer, head, 'qk')
                    if loaded_data is not None:
                        self.svd_cache[qk_cache_key] = loaded_data
                        qk_loaded = True
                
                if not qk_loaded:
                    logger.info(f"Computing SVD for {head_key}_qk")
                    U_qk, S_qk, Vh_qk, W_QK = self._compute_qk_svd(layer, head)
                    self.svd_cache[qk_cache_key] = (U_qk, S_qk, Vh_qk, W_QK)
                    self._save_svd_components(layer, head, 'qk', U_qk, S_qk, Vh_qk, W_QK)
                
                # OV components
                ov_cache_key = f"{head_key}_ov"
                ov_loaded = False
                
                if use_cache:
                    loaded_data = self._load_svd_components(layer, head, 'ov')
                    if loaded_data is not None:
                        self.svd_cache[ov_cache_key] = loaded_data
                        ov_loaded = True
                
                if not ov_loaded:
                    logger.info(f"Computing SVD for {head_key}_ov")
                    U_ov, S_ov, Vh_ov, W_OV = self._compute_ov_svd(layer, head)
                    self.svd_cache[ov_cache_key] = (U_ov, S_ov, Vh_ov, W_OV)
                    self._save_svd_components(layer, head, 'ov', U_ov, S_ov, Vh_ov, W_OV)
        
        # Process MLP layers if enabled
        if self.mask_mlp:
            for layer in range(self.n_layers):
                # MLP input components
                mlp_in_cache_key = f"mlp_{layer}_in"
                mlp_in_loaded = False
                
                if use_cache:
                    loaded_data = self._load_svd_components(layer, -1, 'mlp_in')
                    if loaded_data is not None:
                        self.svd_cache[mlp_in_cache_key] = loaded_data
                        mlp_in_loaded = True
                
                if not mlp_in_loaded:
                    logger.info(f"Computing SVD for MLP layer {layer} input")
                    U_in, S_in, Vh_in, W_in = self._compute_mlp_in_svd(layer)
                    self.svd_cache[mlp_in_cache_key] = (U_in, S_in, Vh_in, W_in)
                    self._save_svd_components(layer, -1, 'mlp_in', U_in, S_in, Vh_in, W_in)
                
                # MLP output components
                mlp_out_cache_key = f"mlp_{layer}_out"
                mlp_out_loaded = False
                
                if use_cache:
                    loaded_data = self._load_svd_components(layer, -1, 'mlp_out')
                    if loaded_data is not None:
                        self.svd_cache[mlp_out_cache_key] = loaded_data
                        mlp_out_loaded = True
                
                if not mlp_out_loaded:
                    logger.info(f"Computing SVD for MLP layer {layer} output")
                    U_out, S_out, Vh_out, W_out = self._compute_mlp_out_svd(layer)
                    self.svd_cache[mlp_out_cache_key] = (U_out, S_out, Vh_out, W_out)
                    self._save_svd_components(layer, -1, 'mlp_out', U_out, S_out, Vh_out, W_out)
        
        logger.info("SVD loading/computation complete.")
    
    def clear_svd_cache(self):
        """Clear the SVD cache directory."""
        if self.svd_cache_dir.exists():
            import shutil
            shutil.rmtree(self.svd_cache_dir)
            logger.info(f"Cleared SVD cache directory: {self.svd_cache_dir}")
            
    def get_masked_weights(self, layer: int, head: int):
        """
        Get masked weight matrices for a specific head.

        Args:
            layer: Layer index
            head: Head index

        Returns:
            Dictionary with W_QK and W_OV matrices for augmented attention computation
        """
        head_key = f'differential_head_{layer}_{head}'

        # Get SVD components from cache
        qk_cache_key = f"{head_key}_qk"
        ov_cache_key = f"{head_key}_ov"

        # Check if SVD is in cache, if not compute/load it
        if qk_cache_key not in self.svd_cache or ov_cache_key not in self.svd_cache:
            self._load_or_compute_svd()

        # Retrieve from cache
        U_qk, S_qk, Vh_qk, W_QK_orig = self.svd_cache[qk_cache_key]
        U_ov, S_ov, Vh_ov, W_OV_orig = self.svd_cache[ov_cache_key]

        # If neither QK nor OV is being trained, return original matrices
        if 'qk' not in self.trainable_mask_types and 'ov' not in self.trainable_mask_types:
            return {'W_QK': W_QK_orig, 'W_OV': W_OV_orig}

        # Apply masking to singular values
        if self.l1_reg:
            qk_mask = mask_fn(self.qk_masks[head_key])
            ov_mask = mask_fn(self.ov_masks[head_key])
        else:
            qk_mask, ov_mask = self.sample_hard_concrete_masks()
            qk_mask = qk_mask[head_key]
            ov_mask = ov_mask[head_key]
        
        S_qk_masked = S_qk[:self.d_head] * qk_mask
        S_ov_masked = S_ov[:self.d_head+1] * ov_mask

        # Reconstruct with masks (use original if not training that mask type)
        if 'qk' in self.trainable_mask_types:
            W_QK = U_qk[:, :self.d_head] @ torch.diag(S_qk_masked) @ Vh_qk[:self.d_head, :]
        else:
            W_QK = W_QK_orig

        if 'ov' in self.trainable_mask_types:
            W_OV = U_ov[:, :self.d_head+1] @ torch.diag(S_ov_masked) @ Vh_ov[:self.d_head+1, :]
        else:
            W_OV = W_OV_orig
        
        return {'W_QK': W_QK, 'W_OV': W_OV, 
                'U_ov': U_ov, 'S_ov': S_ov, 'Vh_ov': Vh_ov, 'ov_mask': ov_mask}
    
    def get_masked_mlp_weights(self, layer: int):
        """
        Get masked weight matrices for MLP layer.

        Args:
            layer: Layer index

        Returns:
            Dictionary with W_in and W_out matrices for augmented MLP computation
        """
        if not self.mask_mlp:
            return None

        mlp_key = f'mlp_{layer}'

        # Get SVD components from cache
        mlp_in_cache_key = f"mlp_{layer}_in"
        mlp_out_cache_key = f"mlp_{layer}_out"

        # Check if SVD is in cache, if not compute/load it
        if mlp_in_cache_key not in self.svd_cache or mlp_out_cache_key not in self.svd_cache:
            self._load_or_compute_svd()

        # Retrieve from cache
        U_in, S_in, Vh_in, W_in_orig = self.svd_cache[mlp_in_cache_key]
        U_out, S_out, Vh_out, W_out_orig = self.svd_cache[mlp_out_cache_key]

        # If neither MLP mask is being trained, return original matrices
        if 'mlp_in' not in self.trainable_mask_types and 'mlp_out' not in self.trainable_mask_types:
            return {'W_in': W_in_orig, 'W_out': W_out_orig}
        
        # Sample masks
        if self.l1_reg:
            mlp_in_mask = mask_fn(self.mlp_in_masks[mlp_key])
            mlp_out_mask = mask_fn(self.mlp_out_masks[mlp_key])
        else:
            mlp_in_mask, mlp_out_mask = self.sample_mlp_hard_concrete_masks()
            mlp_in_mask = mlp_in_mask[mlp_key]
            mlp_out_mask = mlp_out_mask[mlp_key]
        
        # Apply masks and reconstruct
        k_in = len(mlp_in_mask)
        k_out = len(mlp_out_mask)
        
        S_in_masked = S_in[:k_in] * mlp_in_mask
        S_out_masked = S_out[:k_out] * mlp_out_mask

        # Reconstruct with masks (use original if not training that mask type)
        if 'mlp_in' in self.trainable_mask_types:
            W_in = U_in[:, :k_in] @ torch.diag(S_in_masked) @ Vh_in[:k_in, :]
        else:
            W_in = W_in_orig

        if 'mlp_out' in self.trainable_mask_types:
            W_out = U_out[:, :k_out] @ torch.diag(S_out_masked) @ Vh_out[:k_out, :]
        else:
            W_out = W_out_orig
        
        return {'W_in': W_in, 'W_out': W_out,
                'U_in': U_in, 'S_in': S_in, 'Vh_in': Vh_in, 'mlp_in_mask': mlp_in_mask,
                'U_out': U_out, 'S_out': S_out, 'Vh_out': Vh_out, 'mlp_out_mask': mlp_out_mask}
    
    def forward_pass_through_model(self, input_ids: torch.Tensor, 
                                   attention_mask: Optional[torch.Tensor] = None,
                                   corrupted_activations: Optional[Dict] = None,
                                   clean_last_idx: Optional[torch.Tensor] = None):
        """
        Forward pass through the model with masked weights and optional activation patching.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            corrupted_activations: Dictionary with corrupted activations for patching
            clean_last_idx: Indices of last valid tokens for patching [batch_size]
            
        Returns:
            Logits from the masked model
        """
        # Store corrupted activations for use in forward pass
        self.corrupted_activations = corrupted_activations
        self.clean_last_idx = clean_last_idx
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        with torch.no_grad():
            _, cache = self.model.run_with_cache(input_ids)
            token_embeddings = cache['hook_embed']
            position_embeddings = cache['hook_pos_embed']
            del cache, _
            hidden_states = token_embeddings + position_embeddings
        
        # Create causal mask
        causal_mask = torch.ones((seq_len, seq_len), device=device).triu(diagonal=1).bool()
        
        # Process each layer
        for layer in range(self.n_layers):
            # Layer normalization before attention
            try:
                ln1_weight = self.state_dict[f'blocks.{layer}.ln1.w'].to(device)
                ln1_bias = self.state_dict[f'blocks.{layer}.ln1.b'].to(device)
            except KeyError:
                ln1_weight = torch.ones(self.d_model, device=device)
                ln1_bias = torch.zeros(self.d_model, device=device)

            # Apply layer norm
            ln1_out = self._apply_layer_norm(hidden_states, ln1_weight, ln1_bias)

            # DEBUG: Check layer norm output
            if torch.isnan(ln1_out).any() or torch.isinf(ln1_out).any():
                logger.error(f"❌ NaN/Inf after layer {layer} ln1!")
                logger.error(f"  Num NaN: {torch.isnan(ln1_out).sum().item()}")
                logger.error(f"  Hidden states had NaN: {torch.isnan(hidden_states).any().item()}")

            # Compute attention with masked weights and optional patching
            attn_output = self._compute_multi_head_attention(ln1_out, layer, causal_mask, attention_mask)

            # DEBUG: Check attention output
            if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
                logger.error(f"❌ NaN/Inf after layer {layer} attention!")
                logger.error(f"  Num NaN: {torch.isnan(attn_output).sum().item()}")
                logger.error(f"  Num Inf: {torch.isinf(attn_output).sum().item()}")

            # Residual connection
            hidden_states = hidden_states + attn_output

            # DEBUG: Check hidden states after attention residual
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                logger.error(f"❌ NaN/Inf in hidden_states after layer {layer} attention residual!")
                logger.error(f"  Num NaN: {torch.isnan(hidden_states).sum().item()}")
            
            try:
                # Layer normalization before MLP
                ln2_weight = self.state_dict[f'blocks.{layer}.ln2.w'].to(device)
                ln2_bias = self.state_dict[f'blocks.{layer}.ln2.b'].to(device)
            except KeyError:
                ln2_weight = torch.ones(self.d_model, device=device)
                ln2_bias = torch.zeros(self.d_model, device=device)
            
            # Apply layer norm
            ln2_out = self._apply_layer_norm(hidden_states, ln2_weight, ln2_bias)

            # DEBUG: Check MLP input
            if torch.isnan(ln2_out).any() or torch.isinf(ln2_out).any():
                logger.error(f"❌ NaN/Inf after layer {layer} ln2!")
                logger.error(f"  Num NaN: {torch.isnan(ln2_out).sum().item()}")

            # MLP with optional patching
            mlp_output = self._compute_mlp(ln2_out, layer)

            # DEBUG: Check MLP output
            if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
                logger.error(f"❌ NaN/Inf after layer {layer} MLP!")
                logger.error(f"  Num NaN: {torch.isnan(mlp_output).sum().item()}")
                logger.error(f"  Num Inf: {torch.isinf(mlp_output).sum().item()}")

            # Residual connection
            hidden_states = hidden_states + mlp_output

            # DEBUG: Check final hidden states for this layer
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                logger.error(f"❌ NaN/Inf in hidden_states after layer {layer} complete!")
                logger.error(f"  Num NaN: {torch.isnan(hidden_states).sum().item()}")
                logger.error(f"  This is the FIRST layer where NaN appears")
                # Return early to prevent cascading NaN
                return hidden_states
        
        # Final layer norm if present
        if hasattr(self.model, 'ln_final'):
            try:
                ln_final_weight = self.state_dict['ln_final.w'].to(device)
                ln_final_bias = self.state_dict['ln_final.b'].to(device)
            except KeyError:
                ln_final_weight = 1
                ln_final_bias = 0
            hidden_states = self._apply_layer_norm(hidden_states, ln_final_weight, ln_final_bias)
        
        # Get logits
        with torch.no_grad():
            W_U = self.state_dict['unembed.W_U'].to(device)
            b_U = self.state_dict['unembed.b_U'].to(device)
        
        logits = torch.matmul(hidden_states, W_U) + b_U
        
        # Clear cached corrupted activations
        self.corrupted_activations = None
        self.clean_last_idx = None
        
        return logits
    
    def _apply_layer_norm(self, x, weight, bias):
        """Apply layer normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.model.cfg.eps)
        return normed * weight + bias
    
    def _compute_multi_head_attention(self, hidden_states, layer, causal_mask, attention_mask=None):
        """
        Compute multi-head attention using effective weights W_QK and W_OV with optional activation patching.
        """
        B, L, d_model = hidden_states.shape
        device = hidden_states.device
        head_dim = d_model // self.n_heads
        
        # Augment hidden states with ones: [1, hidden_states]
        ones = torch.ones(B, L, 1, device=device)
        hidden_aug = torch.cat([ones, hidden_states], dim=2)  # (B, L, d_model+1)
        
        # Initialize output tensor
        output = torch.zeros(B, L, d_model, device=device)
        
        # Process each attention head
        for head in range(self.n_heads):
            # Get effective weights for this head
            w = self.get_masked_weights(layer, head)
            W_QK = w['W_QK']  # (d_model+1, d_model+1)
            W_OV = w['W_OV']  # (d_model+1, d_model)

            # DEBUG: Check masked weights
            if torch.isnan(W_QK).any() or torch.isinf(W_QK).any():
                logger.error(f"❌ NaN/Inf in W_QK for layer {layer} head {head}!")
                logger.error(f"  Num NaN: {torch.isnan(W_QK).sum().item()}")
                logger.error(f"  Num Inf: {torch.isinf(W_QK).sum().item()}")
                if not torch.isnan(W_QK).all():
                    valid = ~(torch.isnan(W_QK) | torch.isinf(W_QK))
                    logger.error(f"  Valid W_QK range: [{W_QK[valid].min().item():.4f}, {W_QK[valid].max().item():.4f}]")
                # Check mask values that created this
                head_key = f'differential_head_{layer}_{head}'
                qk_mask_logit = self.qk_masks[head_key]
                qk_mask_val = torch.sigmoid(qk_mask_logit)
                logger.error(f"  QK mask logits had NaN: {torch.isnan(qk_mask_logit).any().item()}")
                logger.error(f"  QK mask values had NaN: {torch.isnan(qk_mask_val).any().item()}")
                if not torch.isnan(qk_mask_logit).any():
                    logger.error(f"  QK mask logit range: [{qk_mask_logit.min().item():.4f}, {qk_mask_logit.max().item():.4f}]")

            if torch.isnan(W_OV).any() or torch.isinf(W_OV).any():
                logger.error(f"❌ NaN/Inf in W_OV for layer {layer} head {head}!")
                logger.error(f"  Num NaN: {torch.isnan(W_OV).sum().item()}")
                logger.error(f"  Num Inf: {torch.isinf(W_OV).sum().item()}")

            # For patching, we need SVD components
            U_ov = w['U_ov']
            S_ov = w['S_ov']
            Vh_ov = w['Vh_ov']
            ov_mask = w['ov_mask']
            
            # Compute attention scores
            q_proj = torch.matmul(hidden_aug, W_QK)  # (B, L, d_model+1)
            attn_scores = torch.bmm(q_proj, hidden_aug.transpose(1, 2))  # (B, L, L)

            # DEBUG: Check attention scores before scaling
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                logger.error(f"❌ NaN/Inf in layer {layer} head {head} attn_scores BEFORE scaling!")
                logger.error(f"  q_proj had NaN: {torch.isnan(q_proj).any().item()}")
                logger.error(f"  hidden_aug had NaN: {torch.isnan(hidden_aug).any().item()}")
                logger.error(f"  W_QK had NaN: {torch.isnan(W_QK).any().item()}")
                if not torch.isnan(attn_scores).all():
                    valid = ~(torch.isnan(attn_scores) | torch.isinf(attn_scores))
                    logger.error(f"  Valid score range: [{attn_scores[valid].min().item():.4f}, {attn_scores[valid].max().item():.4f}]")

            # Scale attention scores
            attn_scores = attn_scores / math.sqrt(head_dim)

            # DEBUG: Check after scaling
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                logger.error(f"❌ NaN/Inf in layer {layer} head {head} attn_scores AFTER scaling!")
                logger.error(f"  Scale factor: {math.sqrt(head_dim)}")

            # Apply masks
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0), -float('inf'))

            if attention_mask is not None:
                expanded_mask = attention_mask.unsqueeze(1).expand(-1, L, -1)
                attn_scores = attn_scores.masked_fill(~expanded_mask, -1e9)

            # Compute attention probabilities
            attn_probs = F.softmax(attn_scores, dim=-1)  # (B, L, L)

            # DEBUG: Check softmax output
            if torch.isnan(attn_probs).any() or torch.isinf(attn_probs).any():
                logger.error(f"❌ NaN/Inf in layer {layer} head {head} attn_probs after softmax!")
                logger.error(f"  Num NaN: {torch.isnan(attn_probs).sum().item()}")
                logger.error(f"  Num Inf: {torch.isinf(attn_probs).sum().item()}")
                # Check if input had extreme values
                finite_mask = torch.isfinite(attn_scores)
                if finite_mask.any():
                    logger.error(f"  Finite attn_scores range: [{attn_scores[finite_mask].min().item():.4f}, {attn_scores[finite_mask].max().item():.4f}]")
                logger.error(f"  All attn_scores are -inf: {(attn_scores == -float('inf')).all().item()}")
            
            # Compute context vectors (weighted sum of values)
            context_aug = torch.bmm(attn_probs, hidden_aug)  # (B, L, d_model+1)
            
            # NEW: Apply activation patching for OV if enabled and training OV masks
            if ('ov' in self.trainable_mask_types and 
                self.corrupted_activations is not None and 
                self.clean_last_idx is not None):
                
                # Get corrupted attention-weighted context for this head at last token
                # This is [B, d_model+1] - the full augmented context from corrupted run
                context_corrupt_last = self.corrupted_activations['ov'][layer][head]  # [B, d_model+1]

                # Project clean context using U (input space to SVD space)
                # W_OV = U @ diag(S) @ Vh, so context @ W_OV = context @ U @ diag(S) @ Vh
                y_clean = context_aug @ U_ov  # [B, L, d_model]

                # Project corrupted context using U
                y_corrupt = context_corrupt_last @ U_ov  # [B, d_model]
                
                # Mix at last token positions: M ⊙ y_clean + (1-M) ⊙ y_corrupt
                # Only use first (d_head+1) SVD components
                batch_indices = torch.arange(B, device=device)
                y_clean_last = y_clean[batch_indices, self.clean_last_idx, :self.d_head+1]  # [B, d_head+1]
                y_corrupt_trunc = y_corrupt[:, :self.d_head+1]  # [B, d_head+1]

                y_mixed = (ov_mask[None, :] * y_clean_last +
                          (1 - ov_mask[None, :]) * y_corrupt_trunc)  # [B, d_head+1]

                # Replace last token position with mixed
                y_clean[batch_indices, self.clean_last_idx, :self.d_head+1] = y_mixed

                # Apply singular values and back-project using Vh
                # y = context @ U @ diag(S) @ Vh
                y_with_S = y_clean[:, :, :self.d_head+1] * S_ov[None, None, :self.d_head+1]
                head_output = y_with_S @ Vh_ov[:self.d_head+1, :]  # [B, L, d_model]
                
            else:
                # Standard forward without patching
                head_output = torch.matmul(context_aug, W_OV)  # (B, L, d_model)
            
            # Add to output
            output += head_output

        return output
    
    def _compute_mlp(self, hidden_states, layer):
        """Compute MLP layer with optional masking and activation patching."""
        device = hidden_states.device
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Get masked weights if MLP masking is enabled
        masked_weights = self.get_masked_mlp_weights(layer)
        
        if masked_weights is None:
            # Original implementation without masking
            with torch.no_grad():
                W_in = self.state_dict[f'blocks.{layer}.mlp.W_in'].to(device)
                b_in = self.state_dict[f'blocks.{layer}.mlp.b_in'].to(device)
                W_out = self.state_dict[f'blocks.{layer}.mlp.W_out'].to(device)
                b_out = self.state_dict[f'blocks.{layer}.mlp.b_out'].to(device)
            
            # First linear layer
            intermediate = torch.matmul(hidden_states, W_in) + b_in
            
            # Apply activation function
            act_fn = getattr(utils, self.cfg.act_fn)
            intermediate = act_fn(intermediate)
            
            # Second linear layer
            output = torch.matmul(intermediate, W_out) + b_out
        else:
            # Masked implementation with augmented matrices and optional patching
            W_in_aug = masked_weights['W_in']  # [d_model+1, d_mlp]
            W_out_aug = masked_weights['W_out']  # [d_mlp+1, d_model]
            
            # Get SVD components for patching
            U_in = masked_weights['U_in']
            S_in = masked_weights['S_in']
            Vh_in = masked_weights['Vh_in']
            mlp_in_mask = masked_weights['mlp_in_mask']
            
            U_out = masked_weights['U_out']
            S_out = masked_weights['S_out']
            Vh_out = masked_weights['Vh_out']
            mlp_out_mask = masked_weights['mlp_out_mask']
            
            # Augment hidden states with ones
            ones = torch.ones(batch_size, seq_len, 1, device=device)
            hidden_aug = torch.cat([ones, hidden_states], dim=2)  # [B, L, d_model+1]
            
            # NEW: Apply activation patching for MLP_in if enabled
            if ('mlp_in' in self.trainable_mask_types and 
                self.corrupted_activations is not None and 
                self.clean_last_idx is not None):
                
                # Get corrupted MLP input at last token
                x_corrupt_last = self.corrupted_activations['mlp_in'][layer]  # [B, d_model]

                # Augment with ones
                ones_corrupt = torch.ones(batch_size, 1, device=device)
                x_corrupt_aug = torch.cat([ones_corrupt, x_corrupt_last], dim=1)  # [B, d_model+1]

                # Project using U (input space to SVD space)
                # W_in_aug = U @ diag(S) @ Vh
                y_clean = hidden_aug @ U_in  # [B, L, rank]
                y_corrupt = x_corrupt_aug @ U_in  # [B, rank]

                # Mix at last token positions
                batch_indices = torch.arange(batch_size, device=device)
                k_in = len(mlp_in_mask)
                y_mixed = (mlp_in_mask[None, :] * y_clean[batch_indices, self.clean_last_idx, :k_in] +
                          (1 - mlp_in_mask[None, :]) * y_corrupt[:, :k_in])  # [B, k_in]

                # Replace last token position
                y_clean[batch_indices, self.clean_last_idx, :k_in] = y_mixed

                # Apply singular values and back-project using Vh
                y_masked = y_clean[:, :, :k_in] * S_in[None, None, :k_in]
                intermediate = y_masked @ Vh_in[:k_in, :]  # [B, L, d_mlp]
                
            else:
                # Standard MLP input without patching
                intermediate = torch.matmul(hidden_aug, W_in_aug)  # [B, L, d_mlp]
            
            # Apply activation function
            act_fn = getattr(utils, self.cfg.act_fn)
            intermediate = act_fn(intermediate)
            
            # Augment intermediate with ones for output computation
            ones_int = torch.ones(batch_size, seq_len, 1, device=device)
            intermediate_aug = torch.cat([ones_int, intermediate], dim=2)  # [B, L, d_mlp+1]
            
            # NEW: Apply activation patching for MLP_out if enabled
            if ('mlp_out' in self.trainable_mask_types and 
                self.corrupted_activations is not None and 
                self.clean_last_idx is not None):
                
                # Get corrupted MLP hidden activation at last token
                h_corrupt_last = self.corrupted_activations['mlp_out'][layer]  # [B, d_mlp]

                # Augment with ones
                ones_corrupt_h = torch.ones(batch_size, 1, device=device)
                h_corrupt_aug = torch.cat([ones_corrupt_h, h_corrupt_last], dim=1)  # [B, d_mlp+1]

                # Project using U (input space to SVD space)
                # W_out_aug = U @ diag(S) @ Vh
                y_clean_out = intermediate_aug @ U_out  # [B, L, d_model]
                y_corrupt_out = h_corrupt_aug @ U_out  # [B, d_model]

                # Mix at last token positions
                batch_indices = torch.arange(batch_size, device=device)
                k_out = len(mlp_out_mask)
                y_mixed_out = (mlp_out_mask[None, :] * y_clean_out[batch_indices, self.clean_last_idx, :k_out] + 
                              (1 - mlp_out_mask[None, :]) * y_corrupt_out[:, :k_out])  # [B, k_out]
                
                # Replace last token position
                y_clean_out[batch_indices, self.clean_last_idx, :k_out] = y_mixed_out

                # Apply singular values and back-project using Vh
                y_masked_out = y_clean_out[:, :, :k_out] * S_out[None, None, :k_out]
                output = y_masked_out @ Vh_out[:k_out, :]  # [B, L, d_model]
                
            else:
                # Standard MLP output without patching
                output = torch.matmul(intermediate_aug, W_out_aug)  # [B, L, d_model]
        
        return output
    
    def find_logit_diff_loss(self, input_ids, attention_mask=None, temperature=1.0, sequence_lengths=None, 
                             indirect_object_index=None, subject_index=None,
                             corrupted_activations=None, clean_last_idx=None):
        '''
        Calculate logit difference loss indirect object and subject logits for masked model outputs and full model outputs.
        '''
        device = input_ids.device
        # Forward pass through full model
        with torch.no_grad():
            full_model_logits = self.model(input_ids, attention_mask=attention_mask)
        
        # Forward pass through masked model with optional patching
        masked_model_logits = self.forward_pass_through_model(
            input_ids, attention_mask, 
            corrupted_activations=corrupted_activations,
            clean_last_idx=clean_last_idx
        )
        
        if indirect_object_index is None or subject_index is None:
            raise ValueError("Both indirect_object_index and subject_index must be provided.")
        
        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=device)
        
        full_indirect_logits = full_model_logits[batch_indices, sequence_lengths - 1, indirect_object_index[:,0]]
        full_subject_logits = full_model_logits[batch_indices, sequence_lengths - 1, subject_index[:,0]]
        
        masked_indirect_logits = masked_model_logits[batch_indices, sequence_lengths - 1, indirect_object_index[:,0]]
        masked_subject_logits = masked_model_logits[batch_indices, sequence_lengths - 1, subject_index[:,0]]
        
        logit_diff_masked = masked_indirect_logits - masked_subject_logits
        logit_diff_full = full_indirect_logits - full_subject_logits
        
        return logit_diff_masked.mean(), logit_diff_full.mean()
    
    def find_KL_divergence(self, input_ids, attention_mask=None, temperature=1.0, sequence_lengths=None,
                           sequence_lengths_corrupt=None, indirect_object_index=None, subject_index=None,
                           corrupted_activations=None, clean_last_idx=None):
        """
        Calculate KL divergence between full model and masked model outputs with optional activation patching.
        """
        device = input_ids.device
        
        # Forward pass through full model
        with torch.no_grad():
            full_model_logits = self.model(input_ids, attention_mask=attention_mask)
        
        # Forward pass through masked model with optional patching
        masked_model_logits = self.forward_pass_through_model(
            input_ids, attention_mask,
            corrupted_activations=corrupted_activations,
            clean_last_idx=clean_last_idx
        )

        # DEBUG: Check masked logits for NaN
        if torch.isnan(masked_model_logits).any() or torch.isinf(masked_model_logits).any():
            logger.error(f"❌ NaN/Inf in masked_model_logits!")
            logger.error(f"  Shape: {masked_model_logits.shape}")
            logger.error(f"  Num NaN: {torch.isnan(masked_model_logits).sum().item()}")
            logger.error(f"  Num Inf: {torch.isinf(masked_model_logits).sum().item()}")
            if not torch.isnan(masked_model_logits).all():
                valid_mask = ~(torch.isnan(masked_model_logits) | torch.isinf(masked_model_logits))
                logger.error(f"  Valid logits range: [{masked_model_logits[valid_mask].min().item():.4f}, {masked_model_logits[valid_mask].max().item():.4f}]")

        # Get the position of the last token in each sequence
        last_token_indices = sequence_lengths - 1

        # Gather the logits for those positions
        batch_indices = torch.arange(input_ids.shape[0], device=device)

        full_logits = full_model_logits[batch_indices, last_token_indices]
        masked_logits = masked_model_logits[batch_indices, last_token_indices]

        # DEBUG: Check extracted logits
        if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
            logger.error(f"❌ NaN/Inf in extracted masked_logits (last token)!")
            logger.error(f"  Shape: {masked_logits.shape}")
            logger.error(f"  Num NaN: {torch.isnan(masked_logits).sum().item()}")

        # Compute KL divergence
        log_probs_full = F.log_softmax(full_logits / temperature, dim=-1)
        probs_full = torch.exp(log_probs_full)
        log_probs_masked = F.log_softmax(masked_logits / temperature, dim=-1)

        # DEBUG: Check softmax outputs
        if torch.isnan(log_probs_masked).any() or torch.isinf(log_probs_masked).any():
            logger.error(f"❌ NaN/Inf in log_probs_masked after softmax!")
            logger.error(f"  Input (masked_logits) had NaN: {torch.isnan(masked_logits).any().item()}")
            logger.error(f"  Input (masked_logits) range: [{masked_logits.min().item():.4f}, {masked_logits.max().item():.4f}]")

        kl_div = F.kl_div(log_probs_masked, probs_full, reduction='batchmean')
        
        # Compute accuracy metrics
        masked_accuracy = (masked_logits.argmax(dim=-1) == indirect_object_index[:,0]).float().mean()
        full_model_accuracy = (full_logits.argmax(dim=-1) == indirect_object_index[:,0]).float().mean()
        exact_match = (masked_logits.argmax(dim=-1) == full_logits.argmax(dim=-1)).float().mean()
        
        return kl_div, masked_accuracy, full_model_accuracy, exact_match
    
    def sample_hard_concrete_masks(self):
        """Sample attention masks using hard concrete distribution"""
        beta = 2/3
        l = -0.1
        r = 1.1
        epsilon = 1e-6
        
        qk_sampled_masks = {}
        ov_sampled_masks = {}
        
        for key in self.qk_masks:
            log_alpha = self.qk_masks[key]
            u = torch.rand_like(log_alpha).clamp(epsilon, 1-epsilon)
            s = mask_fn((torch.log(u) - torch.log(1-u) + log_alpha)/beta)
            s_bar = s * (r - l) + l
            z = torch.clamp(s_bar, 0, 1)
            qk_sampled_masks[key] = z
        
        for key in self.ov_masks:
            log_alpha = self.ov_masks[key]
            u = torch.rand_like(log_alpha).clamp(epsilon, 1-epsilon)
            s = mask_fn((torch.log(u) - torch.log(1-u) + log_alpha)/beta)
            s_bar = s * (r - l) + l
            z = torch.clamp(s_bar, 0, 1)
            ov_sampled_masks[key] = z
        
        return qk_sampled_masks, ov_sampled_masks
    
    def sample_mlp_hard_concrete_masks(self):
        """Sample MLP masks using hard concrete distribution"""
        if not self.mask_mlp:
            return None, None
        
        beta = 2/3
        l = -0.1
        r = 1.1
        epsilon = 1e-6
        
        mlp_in_sampled_masks = {}
        mlp_out_sampled_masks = {}
        
        for key in self.mlp_in_masks:
            log_alpha = self.mlp_in_masks[key]
            u = torch.rand_like(log_alpha).clamp(epsilon, 1-epsilon)
            s = mask_fn((torch.log(u) - torch.log(1-u) + log_alpha)/beta)
            s_bar = s * (r - l) + l
            z = torch.clamp(s_bar, 0, 1)
            mlp_in_sampled_masks[key] = z
        
        for key in self.mlp_out_masks:
            log_alpha = self.mlp_out_masks[key]
            u = torch.rand_like(log_alpha).clamp(epsilon, 1-epsilon)
            s = mask_fn((torch.log(u) - torch.log(1-u) + log_alpha)/beta)
            s_bar = s * (r - l) + l
            z = torch.clamp(s_bar, 0, 1)
            mlp_out_sampled_masks[key] = z
        
        return mlp_in_sampled_masks, mlp_out_sampled_masks
    
    def get_thresholded_masks(self, threshold=1e-2):
        """Get binary masks by applying a threshold to the sigmoid of mask parameters.

        Args:
            threshold: Components with sigmoid(mask) < threshold are considered pruned.
                      Default 1e-2 (1% strength) for practical binary decisions."""
        thresholded_masks = {
            'qk_masks': {},
            'ov_masks': {},
            'mlp_in_masks': {},
            'mlp_out_masks': {}
        }
        
        for key in self.qk_masks:
            mask = mask_fn(self.qk_masks[key])
            thresholded_masks['qk_masks'][key] = mask
        
        for key in self.ov_masks:
            mask = mask_fn(self.ov_masks[key])
            thresholded_masks['ov_masks'][key] = mask
        
        if self.mask_mlp:
            for key in self.mlp_in_masks:
                mask = mask_fn(self.mlp_in_masks[key])
                thresholded_masks['mlp_in_masks'][key] = mask
            
            for key in self.mlp_out_masks:
                mask = mask_fn(self.mlp_out_masks[key])
                thresholded_masks['mlp_out_masks'][key] = mask
        
        return thresholded_masks
    
    def validate_with_thresholded_masks(self, input_ids, attention_mask=None, sequence_lengths=None,
                                       threshold=1e-2, corrupted_activations=None, clean_last_idx=None):
        """Validate model with thresholded binary masks.

        Args:
            threshold: Components with sigmoid(mask) < threshold are considered pruned.
                      Default 1e-2 (1% strength) for practical binary decisions."""
        with torch.no_grad():
            kl_div, _, _, _ = self.find_KL_divergence(
                input_ids, attention_mask, sequence_lengths=sequence_lengths,
                corrupted_activations=corrupted_activations,
                clean_last_idx=clean_last_idx
            )
            l1_penalty = self.get_l1_penalty()
            sparsity_stats = self.get_sparsity_stats()
        
        return {
            'kl_div': kl_div.item(),
            'l1_penalty': l1_penalty.item(),
            'sparsity_stats': sparsity_stats
        }
    
    def get_l1_penalty(self):
        """Calculate L1 norm of masks being trained (after sigmoid)."""
        l1_norm = 0.0

        if 'qk' in self.trainable_mask_types:
            for key in self.qk_masks:
                l1_norm += mask_fn(self.qk_masks[key]).sum()

        if 'ov' in self.trainable_mask_types:
            for key in self.ov_masks:
                l1_norm += mask_fn(self.ov_masks[key]).sum()

        if 'mlp_in' in self.trainable_mask_types:
            for key in self.mlp_in_masks:
                l1_norm += mask_fn(self.mlp_in_masks[key]).sum()

        if 'mlp_out' in self.trainable_mask_types:
            for key in self.mlp_out_masks:
                l1_norm += mask_fn(self.mlp_out_masks[key]).sum()

        return l1_norm
    
    def get_l0_penalty(self):
        """Calculate the L0 complexity loss for masks being trained."""
        beta = torch.tensor(2/3, device=self.device)
        gamma = torch.tensor(-0.1, device=self.device)
        zeta = torch.tensor(1.1, device=self.device)

        l0_norm = 0.0

        if 'qk' in self.trainable_mask_types:
            for key in self.qk_masks:
                log_alpha = self.qk_masks[key]
                prob_non_zero = mask_fn(log_alpha - beta * torch.log(-gamma/zeta))
                l0_norm += prob_non_zero.sum()

        if 'ov' in self.trainable_mask_types:
            for key in self.ov_masks:
                log_alpha = self.ov_masks[key]
                prob_non_zero = mask_fn(log_alpha - beta * torch.log(-gamma/zeta))
                l0_norm += prob_non_zero.sum()

        if self.mask_mlp:
            if 'mlp_in' in self.trainable_mask_types:
                for key in self.mlp_in_masks:
                    log_alpha = self.mlp_in_masks[key]
                    prob_non_zero = mask_fn(log_alpha - beta * torch.log(-gamma/zeta))
                    l0_norm += prob_non_zero.sum()

            if 'mlp_out' in self.trainable_mask_types:
                for key in self.mlp_out_masks:
                    log_alpha = self.mlp_out_masks[key]
                    prob_non_zero = mask_fn(log_alpha - beta * torch.log(-gamma/zeta))
                    l0_norm += prob_non_zero.sum()

        return l0_norm
    
    def train_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        sequence_lengths_corrupt: Optional[torch.Tensor] = None,
        num_iterations: int = 1000,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        l1_weight: float = 0.1,
        # target_kl: float = 0.01,
        temperature: float = 1.0,
        eval_interval: int = 50,
        patience: int = 5,
        min_lr: float = 1e-6,
        lr_factor: float = 0.5,
        annealing_lambda_epochs: int = 0,
        indirect_objects: Optional[torch.Tensor] = None,
        subjects: Optional[torch.Tensor] = None,
        corrupted_activations: Optional[Dict] = None,
        clean_last_idx: Optional[torch.Tensor] = None,
    ):
        """
        Train masks to minimize KL divergence with L1 regularization and activation patching.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            sequence_lengths: Sequence lengths for clean inputs
            sequence_lengths_corrupt: Sequence lengths for corrupted inputs (unused, kept for compatibility)
            num_iterations: Maximum number of training iterations
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            l1_weight: Weight for L1 penalty
            # target_kl: Target KL divergence
            temperature: Temperature for softmax
            eval_interval: Evaluate every n iterations
            patience: Patience for early stopping/LR reduction
            min_lr: Minimum learning rate
            lr_factor: Factor to reduce learning rate
            annealing_lambda_epochs: Epochs for annealing (unused, kept for compatibility)
            indirect_objects: Indirect object token IDs for accuracy
            subjects: Subject token IDs for accuracy
            corrupted_activations: Dictionary with corrupted activations for patching
            clean_last_idx: Indices of last valid tokens [batch_size]
            
        Returns:
            Dictionary with training history
        """
        device = input_ids.device
        
        # Move inputs to correct device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Collect only trainable mask parameters
        parameters = []

        if 'qk' in self.trainable_mask_types:
            parameters += list(self.qk_masks.values())

        if 'ov' in self.trainable_mask_types:
            parameters += list(self.ov_masks.values())

        if self.mask_mlp:
            if 'mlp_in' in self.trainable_mask_types:
                parameters += list(self.mlp_in_masks.values())
            if 'mlp_out' in self.trainable_mask_types:
                parameters += list(self.mlp_out_masks.values())

        # Filter to only include parameters that require grad
        parameters = [p for p in parameters if p.requires_grad]

        if len(parameters) == 0:
            raise ValueError("No trainable mask parameters found. Check trainable_mask_types configuration.")

        # Create optimizer with explicit hyperparameters
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),  # Standard momentum and RMSprop decay
            eps=1e-7,            # Numerical stability (increased from 1e-8 to prevent NaN in momentum)
            weight_decay=weight_decay
        )

        # Warmup configuration
        warmup_steps = getattr(self.cfg, 'warmup_steps', 0)
        if warmup_steps > 0:
            logger.info(f"Using linear warmup for {warmup_steps} steps")

        # Initialize training history
        history = {
            'iteration': [],
            'loss': [],
            'kl_div': [],
            'l1_penalty': [],
            'lr': [],
            'logit_diff_masked': [],
            'logit_diff_clean': [],
            'accuracy': [],
            'exact_match': [],
            'masked_accuracy': [],
        }
        
        # Training loop
        best_loss = float('inf')
        wait = 0
        current_lr = learning_rate
        
        pbar = tqdm(range(num_iterations), desc="Training masks")
        for iteration in pbar:

            # Apply warmup schedule (only if warmup_steps > 0)
            if warmup_steps > 0 and iteration < warmup_steps:
                # Linear warmup: 0 -> learning_rate
                warmup_factor = (iteration + 1) / warmup_steps
                current_lr = learning_rate * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            elif iteration == warmup_steps:
                # End of warmup, reset to full LR
                current_lr = learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                logger.info(f"Warmup complete, LR set to {current_lr}")

            # Forward pass and calculate loss with activation patching
            try:
                kl_div, masked_accuracy, full_model_accuracy, exact_match = self.find_KL_divergence(
                    input_ids, attention_mask,
                    temperature, sequence_lengths, sequence_lengths_corrupt,
                    indirect_object_index=indirect_objects,
                    subject_index=subjects,
                    corrupted_activations=corrupted_activations,
                    clean_last_idx=clean_last_idx
                )
            except Exception as e:
                logger.error(f"❌ Exception in find_KL_divergence at iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break

            # DEBUG: Check KL divergence for NaN
            if torch.isnan(kl_div) or torch.isinf(kl_div):
                logger.error(f"❌ NaN/Inf detected in KL divergence at iteration {iteration}")
                logger.error(f"  KL value: {kl_div.item()}")
                logger.error("  Problem is in forward pass or KL computation")
                break

            # Calculate penalty
            if self.l1_reg:
                l1_penalty = self.get_l1_penalty()
            else:
                l1_penalty = self.get_l0_penalty()

            # DEBUG: Check L1 penalty for NaN
            if torch.isnan(l1_penalty) or torch.isinf(l1_penalty):
                logger.error(f"❌ NaN/Inf detected in L1 penalty at iteration {iteration}")
                logger.error(f"  L1 value: {l1_penalty.item()}")
                logger.error("  Problem is in mask values")
                # Log mask statistics
                for idx, key in enumerate(list(self.qk_masks.keys())[:3]):
                    mask_val = torch.sigmoid(self.qk_masks[key])
                    logger.error(f"  QK mask {idx}: min={mask_val.min().item():.6f}, max={mask_val.max().item():.6f}, has_nan={torch.isnan(mask_val).any().item()}")
                break

            loss = kl_div + l1_weight * l1_penalty
            
            # Check for NaN/Inf before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"❌ NaN/Inf detected in loss at iteration {iteration}. Stopping inner loop early.")
                logger.error(f"  Loss: {loss.item() if not torch.isnan(loss) else 'nan'}")
                logger.error(f"  KL: {kl_div.item() if not torch.isnan(kl_div) else 'nan'}")
                logger.error(f"  L1: {l1_penalty.item() if not torch.isnan(l1_penalty) else 'nan'}")
                logger.error(f"  L1 weight: {l1_weight}")

                # Log some mask values to help debug
                first_key = list(self.qk_masks.keys())[0]
                logger.error(f"  Sample QK mask values: {torch.sigmoid(self.qk_masks[first_key][:5])}")
                logger.error(f"  Sample QK logits: {self.qk_masks[first_key][:5]}")
                break

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Check for NaN in gradients BEFORE clipping
            has_nan_grad = False
            for param_idx, param in enumerate(parameters):
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    logger.error(f"❌ NaN/Inf in gradient for parameter {param_idx} at iteration {iteration}")
                    logger.error(f"  Gradient shape: {param.grad.shape}")
                    logger.error(f"  Num NaN: {torch.isnan(param.grad).sum().item()}")
                    logger.error(f"  Num Inf: {torch.isinf(param.grad).sum().item()}")
                    if not torch.isnan(param.grad).all():
                        logger.error(f"  Grad min: {param.grad[~torch.isnan(param.grad)].min().item()}")
                        logger.error(f"  Grad max: {param.grad[~torch.isnan(param.grad)].max().item()}")
                    break

            if has_nan_grad:
                logger.error("  Stopping training due to NaN in gradients")
                break

            # CRITICAL FIX: Gradient clipping to prevent explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

            # Log gradient norm for debugging (only occasionally to avoid spam)
            if iteration % 10 == 0:
                logger.debug(f"Gradient norm at iteration {iteration}: {grad_norm:.4f}")

            optimizer.step()

            # Check for NaN in parameters after optimizer step
            has_nan_params = False
            for param_idx, param in enumerate(parameters):
                if torch.isnan(param).any() or torch.isinf(param).any():
                    has_nan_params = True
                    logger.error(f"❌ NaN/Inf detected in parameter {param_idx} at iteration {iteration}. Stopping inner loop early.")
                    logger.error(f"  Parameter shape: {param.shape}")
                    logger.error(f"  Num NaNs: {torch.isnan(param).sum().item()}")
                    logger.error(f"  Num Infs: {torch.isinf(param).sum().item()}")
                    break

            if has_nan_params:
                break

            # Log progress
            pbar.set_description(f"Loss: {loss.item():.4f}, KL: {kl_div.item():.4f}, L1: {l1_penalty.item():.4f}")
            
            # Evaluate periodically
            if iteration % eval_interval == 0 or iteration == num_iterations - 1:
                # Record history
                history['iteration'].append(iteration)
                history['loss'].append(loss.item())
                history['kl_div'].append(kl_div.item())
                history['l1_penalty'].append(l1_penalty.item())
                history['lr'].append(current_lr)
                history['accuracy'].append(full_model_accuracy.item())
                history['exact_match'].append(exact_match.item())
                history['masked_accuracy'].append(masked_accuracy.item())
                
                # Check early stopping/LR reduction (skip during warmup)
                if iteration >= warmup_steps:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        wait = 0
                    else:
                        wait += 1

                    # Reduce learning rate if needed (ReduceLROnPlateau)
                    if wait >= patience:
                        if current_lr <= min_lr:
                            logger.info(f"Early stopping at iteration {iteration}")
                            break

                        current_lr = max(current_lr * lr_factor, min_lr)
                        logger.info(f"Reducing learning rate to {current_lr}")

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr

                        wait = 0
        
        # Final evaluation
        with torch.no_grad():
            final_kl, masked_accuracy, full_model_accuracy, exact_match = self.find_KL_divergence(
                input_ids, attention_mask, temperature, sequence_lengths,
                indirect_object_index=indirect_objects, 
                subject_index=subjects,
                corrupted_activations=corrupted_activations,
                clean_last_idx=clean_last_idx
            )
            if self.l1_reg:
                final_l1 = self.get_l1_penalty().item()
            else:
                final_l1 = self.get_l0_penalty().item()
        
        logger.info(f"Training complete. Final KL: {final_kl:.6f}, Final L1: {final_l1:.6f}")
        
        return history
    
    def visualize_masks(self, output_dir=None, data_type='gp'):
        """
        Visualize learned masks for all attention heads and MLP layers.

        This method now delegates to the visualization module for improved aesthetics.

        Args:
            output_dir: Directory to save visualizations
            data_type: Type of data (e.g., 'ioi', 'gp') for IOI color coding
        """
        viz_masks(self, output_dir=output_dir, mask_fn=mask_fn, data_type=data_type)

    def visualize_masked_singular_values(self, output_dir=None):
        """
        Visualize mask_value * singular_value for each component.

        This shows the effective strength of each singular value component after masking.
        This method now delegates to the visualization module for improved aesthetics.
        """
        viz_masked_svs(self, output_dir=output_dir, mask_fn=mask_fn)

    def plot_training_history(self, history, output_dir=None):
        """
        Plot training history.

        This method now delegates to the visualization module for improved aesthetics.
        """
        plot_history(history, output_dir=output_dir)
    
    def get_sparsity_stats(self):
        """Get statistics about mask sparsity."""
        stats = {
            'layer': [],
            'head': [],
            'qk_sparsity': [],
            'ov_sparsity': [],
            'overall_sparsity': []
        }

        if self.mask_mlp:
            stats['mlp_in_sparsity'] = []
            stats['mlp_out_sparsity'] = []

        total_masked = 0
        total_values = 0
        threshold = 1e-3  # Components < 0.1% strength are considered functionally pruned
        
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_key = f'differential_head_{layer}_{head}'
                
                # Get mask values
                qk_mask = mask_fn(self.qk_masks[head_key]).detach().cpu()
                ov_mask = mask_fn(self.ov_masks[head_key]).detach().cpu()
                
                # Calculate sparsity
                qk_sparsity = (qk_mask < threshold).float().mean().item()
                ov_sparsity = (ov_mask < threshold).float().mean().item()
                
                # Update statistics
                stats['layer'].append(layer)
                stats['head'].append(head)
                stats['qk_sparsity'].append(qk_sparsity)
                stats['ov_sparsity'].append(ov_sparsity)
                stats['overall_sparsity'].append((qk_sparsity + ov_sparsity) / 2)
                
                # Update totals
                total_masked += (qk_mask < threshold).sum().item() + (ov_mask < threshold).sum().item()
                total_values += len(qk_mask) + len(ov_mask)
            
            if self.mask_mlp:
                mlp_key = f'mlp_{layer}'
                
                # Get MLP mask values
                mlp_in_mask = mask_fn(self.mlp_in_masks[mlp_key]).detach().cpu()
                mlp_out_mask = mask_fn(self.mlp_out_masks[mlp_key]).detach().cpu()
                
                # Calculate MLP sparsity
                mlp_in_sparsity = (mlp_in_mask < threshold).float().mean().item()
                mlp_out_sparsity = (mlp_out_mask < threshold).float().mean().item()
                
                stats['mlp_in_sparsity'].append(mlp_in_sparsity)
                stats['mlp_out_sparsity'].append(mlp_out_sparsity)
                
                # Update totals
                total_masked += (mlp_in_mask < threshold).sum().item() + (mlp_out_mask < threshold).sum().item()
                total_values += len(mlp_in_mask) + len(mlp_out_mask)
        
        # Add overall statistics
        stats['total_sparsity'] = total_masked / total_values if total_values > 0 else 0

        return stats

    def get_relative_and_full_sparsity(self, threshold=1e-3):
        """
        Calculate both relative and full sparsity metrics.

        Relative sparsity: Sparsity relative to the reduced rank we train (e.g., d_head=64 for QK)
        Full sparsity: Sparsity relative to the full theoretical dimension (e.g., d_model+1=769 for QK)

        Args:
            threshold: Mask values below this are considered pruned (default 1e-3)

        Returns:
            dict with:
                - num_active_components: Count of components with mask >= threshold
                - relative_sparsity: 1 - (active / reduced_rank_total)
                - full_sparsity: 1 - (active / full_dimension_total)
                - relative_denominator: Total reduced rank components
                - full_denominator: Total full dimension components
        """
        # Dimensions
        d_model = self.d_model
        d_head = self.d_head
        d_mlp = self.d_mlp
        n_heads = self.n_layers * self.n_heads
        n_layers = self.n_layers

        # Rank (what we actually train/mask)
        rank_qk = d_head                    # 64
        rank_ov = d_head + 1                # 65
        rank_mlp_in = min(d_model + 1, d_mlp)   # 769
        rank_mlp_out = min(d_mlp + 1, d_model)  # 768

        # Full dimensions (theoretical maximum from SVD)
        dim_qk = d_model + 1                    # 769 (augmented W_QK)
        dim_ov = min(d_model + 1, d_model)      # 768 (augmented W_OV)
        dim_mlp_in = min(d_model + 1, d_mlp)    # 769
        dim_mlp_out = min(d_mlp + 1, d_model)   # 768

        # Calculate denominators
        relative_denominator = n_heads * (rank_qk + rank_ov)
        full_denominator = n_heads * (dim_qk + dim_ov)

        if self.mask_mlp:
            relative_denominator += n_layers * (rank_mlp_in + rank_mlp_out)
            full_denominator += n_layers * (dim_mlp_in + dim_mlp_out)

        # Count active components (mask >= threshold)
        num_active = 0

        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_key = f'differential_head_{layer}_{head}'

                qk_mask = mask_fn(self.qk_masks[head_key]).detach()
                ov_mask = mask_fn(self.ov_masks[head_key]).detach()

                # Count components >= threshold (active)
                num_active += (qk_mask >= threshold).sum().item()
                num_active += (ov_mask >= threshold).sum().item()

            if self.mask_mlp:
                mlp_key = f'mlp_{layer}'
                mlp_in_mask = mask_fn(self.mlp_in_masks[mlp_key]).detach()
                mlp_out_mask = mask_fn(self.mlp_out_masks[mlp_key]).detach()

                num_active += (mlp_in_mask >= threshold).sum().item()
                num_active += (mlp_out_mask >= threshold).sum().item()

        # Calculate sparsity metrics
        relative_sparsity = 1.0 - (num_active / relative_denominator) if relative_denominator > 0 else 0.0
        full_sparsity = 1.0 - (num_active / full_denominator) if full_denominator > 0 else 0.0

        return {
            'num_active_components': num_active,
            'relative_sparsity': relative_sparsity,
            'full_sparsity': full_sparsity,
            'relative_denominator': relative_denominator,
            'full_denominator': full_denominator,
            'relative_sparsity_pct': relative_sparsity * 100,
            'full_sparsity_pct': full_sparsity * 100,
        }

    def to(self, device):
        """Move model to device."""
        for key in self.qk_masks:
            self.qk_masks[key] = self.qk_masks[key].to(device)
        for key in self.ov_masks:
            self.ov_masks[key] = self.ov_masks[key].to(device)
        
        if self.mask_mlp:
            for key in self.mlp_in_masks:
                self.mlp_in_masks[key] = self.mlp_in_masks[key].to(device)
            for key in self.mlp_out_masks:
                self.mlp_out_masks[key] = self.mlp_out_masks[key].to(device)
        
        return self