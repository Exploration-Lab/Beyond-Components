#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for SVD-based Transformer Circuit Discovery with Activation Patching
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import logging
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import data_loader as local_data_loader
from src.utils.utils import get_data_column_names, get_indirect_objects_and_subjects
from collections import deque
import pandas as pd
import seaborn as sns

# Set style for aesthetic plots
sns.set_style("whitegrid")

# Import our circuit discovery implementation
from src.models.masked_transformer_circuit import MaskedTransformerCircuit

# IOI head categories with colors - using vibrant, publication-quality color palette
IOI_HEAD_CATEGORIES = {
    'Previous Token Heads': {
        'color': '#E63946',  # Vibrant Red
        'heads': [(2, 2), (4, 11)]
    },
    'Duplicate Token Heads': {
        'color': '#1D3557',  # Navy Blue
        'heads': [(0, 1), (3, 0), (0, 10)]
    },
    'Induction Heads': {
        'color': '#2A9D8F',  # Teal/Green
        'heads': [(5, 5), (6, 9), (5, 8), (5, 9)]
    },
    'S-Inhibition Heads': {
        'color': '#8338EC',  # Vivid Purple
        'heads': [(7, 3), (7, 9), (8, 6), (8, 10)]
    },
    'Negative Name Mover Heads': {
        'color': '#FF6B35',  # Vibrant Orange
        'heads': [(10, 7), (11, 10)]
    },
    'Name Mover Heads': {
        'color': '#F4A261',  # Sandy Orange/Gold
        'heads': [(9, 9), (9, 6), (10, 0)]
    },
    'Backup Name Mover Heads': {
        'color': '#8B4513',  # Saddle Brown
        'heads': [(9, 0), (9, 7), (10, 1), (10, 2),
                 (10, 6), (10, 10), (11, 2), (11, 9)]
    },
    'Other Heads': {
        'color': '#CCCCCC',  # Light Gray
        'heads': []  # Will be filled automatically with non-circuit heads
    }
}

def get_head_color_ioi(layer, head):
    """Get color for a head based on IOI circuit categories"""
    for category, info in IOI_HEAD_CATEGORIES.items():
        if (layer, head) in info['heads']:
            return info['color'], category
    return IOI_HEAD_CATEGORIES['Other Heads']['color'], 'Other Heads'

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train SVD-based circuit discovery")
    parser.add_argument("--config", type=str, default="circuit_config.yaml", help="Path to config file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--train_masks", type=str, default=None,
                       help="Comma-separated list of masks to train (QK,OV,MLP_in,MLP_out). Default: all")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_dir, experiment_name):
    """Setup logging to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(experiment_name)
    return logger

def extract_corrupted_activations(model, corrupted_cache, corrupt_last_idx, device):
    """
    Extract last token activations from corrupted forward pass cache
    
    Args:
        model: HookedTransformer model
        corrupted_cache: Cache from corrupted forward pass
        corrupt_last_idx: Indices of last valid tokens [batch_size]
        device: torch device
        
    Returns:
        Dictionary with corrupted activations for OV, MLP_in, MLP_out
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    batch_size = corrupt_last_idx.shape[0]
    batch_indices = torch.arange(batch_size, device=device)
    
    corrupted_activations = {
        'ov': {},      # [layer][head] -> [batch, d_model+1] (augmented context)
        'mlp_in': {},  # [layer] -> [batch, d_model]
        'mlp_out': {}  # [layer] -> [batch, d_mlp]
    }
    
    for layer in range(n_layers):
        # For OV patching, we need context vectors in augmented formulation: attn_pattern @ [1, input]
        # Note: The corrupted forward pass uses the standard pretrained model (not augmented),
        # so we need to construct the augmented context from cached activations

        # Get the input to attention (after layer norm) at last token positions
        attn_in_corrupt = corrupted_cache[f'blocks.{layer}.ln1.hook_normalized']  # [B, seq, d_model]
        attn_in_last = attn_in_corrupt[batch_indices, corrupt_last_idx, :]  # [B, d_model]

        # Get attention patterns for all heads at last token positions
        attn_pattern_corrupt = corrupted_cache[f'blocks.{layer}.attn.hook_pattern']  # [B, n_heads, seq_q, seq_k]
        attn_weights_last = attn_pattern_corrupt[batch_indices, :, corrupt_last_idx, :]  # [B, n_heads, seq]

        corrupted_activations['ov'][layer] = {}

        for head in range(n_heads):
            # Compute attention-weighted sum: context = attn_weights @ inputs
            attn_w = attn_weights_last[:, head, :]  # [B, seq]

            # Standard context (without augmentation)
            context_standard = torch.matmul(attn_w.unsqueeze(1), attn_in_corrupt).squeeze(1)  # [B, d_model]

            # Augment with bias term: [1, context_standard]
            # The '1' coefficient sums the attention weights (which sum to 1.0 after softmax)
            ones = torch.ones(batch_size, 1, device=device)
            context_corrupt = torch.cat([ones, context_standard], dim=1)  # [B, d_model+1]

            corrupted_activations['ov'][layer][head] = context_corrupt
        
        # MLP_in: residual stream input to MLP at last token (after layer norm)
        mlp_in_corrupt = corrupted_cache[f'blocks.{layer}.ln2.hook_normalized']
        corrupted_activations['mlp_in'][layer] = mlp_in_corrupt[batch_indices, corrupt_last_idx, :]
        
        # MLP_out: hidden activation after activation function at last token
        mlp_out_corrupt = corrupted_cache[f'blocks.{layer}.mlp.hook_post']
        corrupted_activations['mlp_out'][layer] = mlp_out_corrupt[batch_indices, corrupt_last_idx, :]
    
    return corrupted_activations

def create_sparsity_progression_plots(sparsity_progression, output_dir, logger):
    """
    Create plots showing sparsity progression over training steps.

    Args:
        sparsity_progression: List of dicts with step-wise sparsity and metrics
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if len(sparsity_progression) == 0:
        return

    # Convert to DataFrame
    df = pd.DataFrame(sparsity_progression)

    # Create plots for metrics vs training step
    y_metrics = [
        ('train_kl', 'KL Divergence', 'blue'),
        ('train_loss', 'Training Loss', 'red'),
        ('train_accuracy', 'Accuracy', 'green'),
        ('relative_sparsity', 'Relative Sparsity (%)', 'purple'),
        ('full_sparsity', 'Full Sparsity (%)', 'orange')
    ]

    for metric_key, metric_label, color in y_metrics:
        if metric_key not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['step'], df[metric_key],
               marker='o', linestyle='-', linewidth=2, markersize=4,
               alpha=0.7, color=color, label=metric_label)

        ax.set_xlabel('Training Step', fontsize=18, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
        # Title removed as per user request
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)

        plt.tight_layout()
        plot_path_png = os.path.join(output_dir, f'progression_{metric_key}.png')
        plot_path_pdf = os.path.join(output_dir, f'progression_{metric_key}.pdf')
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        plt.close()

    # Create combined plots: sparsity vs metrics (what user wants to see!)
    if 'relative_sparsity' in df.columns:
        # Plot each metric vs sparsity
        metrics_for_sparsity_plot = [
            ('train_kl', 'KL Divergence', 'blue'),
            ('train_loss', 'Training Loss', 'red'),
            ('train_accuracy', 'Accuracy', 'green')
        ]

        for metric_key, metric_label, color in metrics_for_sparsity_plot:
            if metric_key not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['relative_sparsity'], df[metric_key],
                               c=df['step'], cmap='viridis',
                               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Step', fontsize=14, fontweight='bold')

            ax.set_xlabel('Relative Sparsity (%)', fontsize=18, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
            # Title removed as per user request
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'train_sparsity_vs_{metric_key}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f'train_sparsity_vs_{metric_key}.pdf'), bbox_inches='tight')
            plt.close()

    # Also create for full_sparsity
    if 'full_sparsity' in df.columns:
        for metric_key, metric_label, color in metrics_for_sparsity_plot:
            if metric_key not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['full_sparsity'], df[metric_key],
                               c=df['step'], cmap='plasma',
                               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Step', fontsize=14, fontweight='bold')

            ax.set_xlabel('Full Sparsity (%)', fontsize=18, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
            # Title removed as per user request
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'train_full_sparsity_vs_{metric_key}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f'train_full_sparsity_vs_{metric_key}.pdf'), bbox_inches='tight')
            plt.close()

    logger.info(f"Saved sparsity progression plots (PNG and PDF) to {output_dir}")


def create_sparsity_plots(metrics_history, output_dir, logger):
    """
    Create aesthetic plots for sparsity (X-axis) vs various metrics (Y-axis) with error bars.
    Shows test evaluation data with mean ± std.

    Args:
        metrics_history: List of dicts with metrics for each epoch
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if len(metrics_history) == 0:
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_history)

    # Define Y metrics to plot (with their std keys)
    y_metrics = [
        ('test_kl', 'test_kl_std', 'KL Divergence', 'blue'),
        ('test_logit_diff', 'test_logit_diff_std', 'Logit Difference', 'green'),
        ('test_exact_match', 'test_exact_match_std', 'Exact Match', 'purple'),
        ('test_masked_acc', 'test_masked_acc_std', 'Masked Accuracy', 'red')
    ]

    # Create plots for RELATIVE SPARSITY (X-axis) vs METRICS (Y-axis)
    for metric_key, std_key, metric_label, color in y_metrics:
        if metric_key not in df.columns or 'relative_sparsity' not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot with error bars showing mean ± std
        if std_key in df.columns:
            # Scatter plot with error bars
            ax.errorbar(df['relative_sparsity'], df[metric_key],
                       yerr=df[std_key],
                       fmt='o',  # Circle markers
                       markersize=10,
                       linewidth=2,
                       capsize=6,
                       capthick=2,
                       elinewidth=2,
                       alpha=0.8,
                       color=color,
                       ecolor=color,
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       label=f'{metric_label} (mean ± std)')

            # Add connecting line if multiple points
            if len(df) > 1:
                ax.plot(df['relative_sparsity'], df[metric_key],
                       linestyle='--', linewidth=1.5, alpha=0.5, color=color, zorder=1)
        else:
            # Just scatter plot without error bars
            ax.scatter(df['relative_sparsity'], df[metric_key],
                      s=100, alpha=0.8, color=color,
                      edgecolors='black', linewidth=1.5,
                      label=metric_label)

        ax.set_xlabel('Relative Sparsity (%)', fontsize=18, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
        # Title removed as per user request
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=1)
        ax.legend(fontsize=18)

        plt.tight_layout()
        # Save both PNG and PDF
        plot_path_png = os.path.join(output_dir, f'test_relative_sparsity_vs_{metric_key}.png')
        plot_path_pdf = os.path.join(output_dir, f'test_relative_sparsity_vs_{metric_key}.pdf')
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        plt.close()

    # Create plots for FULL SPARSITY (X-axis) vs METRICS (Y-axis)
    for metric_key, std_key, metric_label, color in y_metrics:
        if metric_key not in df.columns or 'full_sparsity' not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot with error bars showing mean ± std
        if std_key in df.columns:
            # Scatter plot with error bars
            ax.errorbar(df['full_sparsity'], df[metric_key],
                       yerr=df[std_key],
                       fmt='s',  # Square markers
                       markersize=10,
                       linewidth=2,
                       capsize=6,
                       capthick=2,
                       elinewidth=2,
                       alpha=0.8,
                       color='darkgreen',
                       ecolor='darkgreen',
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       label=f'{metric_label} (mean ± std)')

            # Add connecting line if multiple points
            if len(df) > 1:
                ax.plot(df['full_sparsity'], df[metric_key],
                       linestyle='--', linewidth=1.5, alpha=0.5, color='darkgreen', zorder=1)
        else:
            # Just scatter plot without error bars
            ax.scatter(df['full_sparsity'], df[metric_key],
                      s=100, alpha=0.8, color='darkgreen',
                      marker='s', edgecolors='black', linewidth=1.5,
                      label=metric_label)

        ax.set_xlabel('Full Sparsity (%)', fontsize=18, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=18, fontweight='bold')
        # Title removed as per user request
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=1)
        ax.legend(fontsize=18)

        plt.tight_layout()
        # Save both PNG and PDF
        plot_path_png = os.path.join(output_dir, f'test_full_sparsity_vs_{metric_key}.png')
        plot_path_pdf = os.path.join(output_dir, f'test_full_sparsity_vs_{metric_key}.pdf')
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved test sparsity plots with error bars (PNG and PDF) to {output_dir}")


def create_mask_value_plots(circuit, output_dir, data_type, logger):
    """
    Create plots of average mask values for QK and OV circuits.
    For IOI tasks, uses color coding based on head categories.

    Args:
        circuit: MaskedTransformerCircuit instance
        output_dir: Directory to save plots
        data_type: Type of data (e.g., 'ioi', 'gp')
        logger: Logger instance
    """
    import json

    is_ioi = (data_type.lower() == 'ioi')

    # Extract mask data for QK circuit
    qk_mask_data = []
    ov_mask_data = []

    n_layers = circuit.model.cfg.n_layers
    n_heads = circuit.model.cfg.n_heads

    for layer in range(n_layers):
        for head in range(n_heads):
            # Correct key format: 'differential_head_{layer}_{head}'
            head_key = f'differential_head_{layer}_{head}'

            # QK masks
            if head_key in circuit.qk_masks:
                qk_mask = circuit.qk_masks[head_key]
                # Apply sigmoid to get actual mask values
                qk_mask_values = torch.sigmoid(qk_mask)
                avg_mask_value = qk_mask_values.mean().item()

                if is_ioi:
                    color, category = get_head_color_ioi(layer, head)
                else:
                    color, category = '#1f77b4', 'All Heads'

                qk_mask_data.append({
                    'layer': layer,
                    'head': head,
                    'avg_mask': avg_mask_value,
                    'color': color,
                    'category': category,
                    'head_label': f'L{layer}H{head}'
                })

            # OV masks
            if head_key in circuit.ov_masks:
                ov_mask = circuit.ov_masks[head_key]
                # Apply sigmoid to get actual mask values
                ov_mask_values = torch.sigmoid(ov_mask)
                avg_mask_value = ov_mask_values.mean().item()

                if is_ioi:
                    color, category = get_head_color_ioi(layer, head)
                else:
                    color, category = '#ff7f0e', 'All Heads'

                ov_mask_data.append({
                    'layer': layer,
                    'head': head,
                    'avg_mask': avg_mask_value,
                    'color': color,
                    'category': category,
                    'head_label': f'L{layer}H{head}'
                })

    # Save data for later replotting
    mask_data_file = os.path.join(output_dir, 'mask_values_data.json')
    with open(mask_data_file, 'w') as f:
        json.dump({
            'qk_masks': qk_mask_data,
            'ov_masks': ov_mask_data,
            'is_ioi': is_ioi
        }, f, indent=2)
    logger.info(f"Saved mask values data to {mask_data_file}")

    # Create QK circuit plot
    if qk_mask_data:
        # Issue B fix: Increase height to accommodate bottom legend
        fig, ax = plt.subplots(figsize=(16, 9))

        # Sort by layer and head
        qk_mask_data_sorted = sorted(qk_mask_data, key=lambda x: (x['layer'], x['head']))

        x_pos = np.arange(len(qk_mask_data_sorted))
        colors = [d['color'] for d in qk_mask_data_sorted]
        avg_masks = [d['avg_mask'] for d in qk_mask_data_sorted]
        labels = [d['head_label'] for d in qk_mask_data_sorted]

        bars = ax.bar(x_pos, avg_masks, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Head', fontsize=18, fontweight='bold')
        ax.set_ylabel('Average Mask Value', fontsize=18, fontweight='bold')
        # Title removed as per user request
        ax.set_xticks(x_pos[::2])  # Show every other label to avoid crowding
        ax.set_xticklabels(labels[::2], rotation=90, ha='right', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Issue B fix: Create legend at bottom with bigger fonts
        if is_ioi:
            from matplotlib.patches import Patch
            legend_elements = []
            for category, info in IOI_HEAD_CATEGORIES.items():
                if any(d['category'] == category for d in qk_mask_data_sorted):
                    legend_elements.append(Patch(facecolor=info['color'],
                                                edgecolor='black',
                                                label=category))
            if legend_elements:
                ax.legend(handles=legend_elements,
                         loc='upper center',
                         bbox_to_anchor=(0.5, -0.12),
                         ncol=min(4, len(legend_elements)),
                         fontsize=15,
                         frameon=True,
                         fancybox=True,
                         shadow=True)

        plt.tight_layout()
        # Save both PNG and PDF
        plt.savefig(os.path.join(output_dir, 'qk_average_mask_values.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'qk_average_mask_values.pdf'), bbox_inches='tight')
        plt.close()
        logger.info(f"Saved QK average mask value plot")

    # Create OV circuit plot
    if ov_mask_data:
        # Issue B fix: Increase height to accommodate bottom legend
        fig, ax = plt.subplots(figsize=(16, 9))

        # Sort by layer and head
        ov_mask_data_sorted = sorted(ov_mask_data, key=lambda x: (x['layer'], x['head']))

        x_pos = np.arange(len(ov_mask_data_sorted))
        colors = [d['color'] for d in ov_mask_data_sorted]
        avg_masks = [d['avg_mask'] for d in ov_mask_data_sorted]
        labels = [d['head_label'] for d in ov_mask_data_sorted]

        bars = ax.bar(x_pos, avg_masks, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Head', fontsize=18, fontweight='bold')
        ax.set_ylabel('Average Mask Value', fontsize=18, fontweight='bold')
        # Title removed as per user request
        ax.set_xticks(x_pos[::2])  # Show every other label to avoid crowding
        ax.set_xticklabels(labels[::2], rotation=90, ha='right', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Issue B fix: Create legend at bottom with bigger fonts
        if is_ioi:
            from matplotlib.patches import Patch
            legend_elements = []
            for category, info in IOI_HEAD_CATEGORIES.items():
                if any(d['category'] == category for d in ov_mask_data_sorted):
                    legend_elements.append(Patch(facecolor=info['color'],
                                                edgecolor='black',
                                                label=category))
            if legend_elements:
                ax.legend(handles=legend_elements,
                         loc='upper center',
                         bbox_to_anchor=(0.5, -0.12),
                         ncol=min(4, len(legend_elements)),
                         fontsize=15,
                         frameon=True,
                         fancybox=True,
                         shadow=True)

        plt.tight_layout()
        # Save both PNG and PDF
        plt.savefig(os.path.join(output_dir, 'ov_average_mask_values.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'ov_average_mask_values.pdf'), bbox_inches='tight')
        plt.close()
        logger.info(f"Saved OV average mask value plot")


def run_test_evaluation(circuit, model, test_loader, config, device, logger):
    """
    Run evaluation on test dataset.

    Returns:
        dict with keys:
            - test_kl: KL divergence between masked and full model
            - test_loss: Same as test_kl
            - test_masked_acc: Accuracy of masked model on correct labels
            - test_full_acc: Accuracy of full model on correct labels
            - test_exact_match: How often masked and full model predictions agree
            - test_masked_exact_match: How often masked model predicts correctly (same as test_masked_acc)
            - test_logit_diff: Logit difference (correct - incorrect)
            - relative_sparsity, full_sparsity: Sparsity metrics
    """
    test_losses = []
    kl_losses = []
    masked_accuracies = []
    full_model_accuracies = []
    exact_matches = []
    masked_exact_matches = []
    logit_diffs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation", leave=False):
            clean_column_name, corrupted_column_name = get_data_column_names(config['data_type'])
            indirect_objects_column_name, subjects_column_name = get_indirect_objects_and_subjects(config['data_type'])

            # Tokenize inputs
            input_ids_clean = model.tokenizer(
                batch[clean_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            input_ids_corrupted = model.tokenizer(
                batch[corrupted_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            # Handle indirect objects - convert to list if tensor
            indirect_objs = batch[indirect_objects_column_name]
            if isinstance(indirect_objs, torch.Tensor):
                indirect_objs = indirect_objs.tolist()
            batch[indirect_objects_column_name] = [' ' + str(obj) for obj in indirect_objs]
            indirect_objects = model.tokenizer(
                batch[indirect_objects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            # Handle subjects - convert to list if tensor
            subjs = batch[subjects_column_name]
            if isinstance(subjs, torch.Tensor):
                subjs = subjs.tolist()
            batch[subjects_column_name] = [' ' + str(subj) for subj in subjs]
            subjects = model.tokenizer(
                batch[subjects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            # Calculate sequence lengths
            clean_lengths = (input_ids_clean != model.tokenizer.pad_token_id).sum(dim=1)
            corrupt_lengths = (input_ids_corrupted != model.tokenizer.pad_token_id).sum(dim=1)

            clean_last_idx = clean_lengths - 1
            corrupt_last_idx = corrupt_lengths - 1

            # Create attention masks
            attention_mask_clean = torch.arange(input_ids_clean.size(1), device=device)[None, :] < clean_lengths[:, None]
            attention_mask_corrupted = torch.arange(input_ids_corrupted.size(1), device=device)[None, :] < corrupt_lengths[:, None]

            # Run corrupted forward pass
            corrupted_logits, corrupted_cache = model.run_with_cache(
                input_ids_corrupted,
                attention_mask=attention_mask_corrupted
            )

            # Extract corrupted activations
            corrupted_activations = extract_corrupted_activations(
                model=model,
                corrupted_cache=corrupted_cache,
                corrupt_last_idx=corrupt_last_idx,
                device=device
            )

            # Forward pass with masked model
            masked_logits = circuit.forward_pass_through_model(
                input_ids_clean,
                attention_mask_clean,
                corrupted_activations=corrupted_activations,
                clean_last_idx=clean_last_idx
            )

            # Forward pass with full model for comparison
            full_logits = model(input_ids_clean, attention_mask=attention_mask_clean)

            # Get the position of the last valid token in each sequence
            batch_indices = torch.arange(input_ids_clean.size(0), device=device)

            # Extract logits for the last token position
            full_last_token_logits = full_logits[batch_indices, clean_last_idx]
            masked_last_token_logits = masked_logits[batch_indices, clean_last_idx]

            # Compute KL divergence
            log_probs_full = F.log_softmax(full_last_token_logits, dim=-1)
            probs_full = torch.exp(log_probs_full)
            log_probs_masked = F.log_softmax(masked_last_token_logits, dim=-1)

            kl_div = F.kl_div(log_probs_masked, probs_full, reduction='batchmean')

            # Compute accuracies
            masked_preds = masked_last_token_logits.argmax(dim=-1)
            full_preds = full_last_token_logits.argmax(dim=-1)
            correct_labels = indirect_objects[:, 0]
            incorrect_labels = subjects[:, 0]

            # Masked accuracy: How often the masked model predicts the correct label
            masked_accuracy = (masked_preds == correct_labels).float().mean().item()
            # Full model accuracy: How often the full model predicts the correct label
            full_model_accuracy = (full_preds == correct_labels).float().mean().item()
            # Exact match: How often masked and full model predictions agree
            exact_match = (masked_preds == full_preds).float().mean().item()
            # Masked exact match: How often the masked model makes the exact correct prediction (same as masked_accuracy)
            masked_exact_match = (masked_preds == correct_labels).float().mean().item()

            # Compute logit difference: logit[correct] - logit[incorrect]
            masked_logit_correct = masked_last_token_logits[batch_indices, correct_labels]
            masked_logit_incorrect = masked_last_token_logits[batch_indices, incorrect_labels]
            logit_diff = (masked_logit_correct - masked_logit_incorrect).mean().item()

            test_losses.append(kl_div.item())
            kl_losses.append(kl_div.item())
            masked_accuracies.append(masked_accuracy)
            full_model_accuracies.append(full_model_accuracy)
            exact_matches.append(exact_match)
            masked_exact_matches.append(masked_exact_match)
            logit_diffs.append(logit_diff)

    # Get sparsity metrics
    sparsity_threshold = config['masking'].get('sparsity_threshold', 1e-3)
    sparsity_metrics = circuit.get_relative_and_full_sparsity(threshold=sparsity_threshold)

    return {
        'test_kl': float(np.mean(kl_losses)),
        'test_kl_std': float(np.std(kl_losses)),
        'test_loss': float(np.mean(test_losses)),
        'test_masked_acc': float(np.mean(masked_accuracies)),
        'test_masked_acc_std': float(np.std(masked_accuracies)),
        'test_full_acc': float(np.mean(full_model_accuracies)),
        'test_full_acc_std': float(np.std(full_model_accuracies)),
        'test_exact_match': float(np.mean(exact_matches)),
        'test_exact_match_std': float(np.std(exact_matches)),
        'test_masked_exact_match': float(np.mean(masked_exact_matches)),
        'test_masked_exact_match_std': float(np.std(masked_exact_matches)),
        'test_logit_diff': float(np.mean(logit_diffs)),
        'test_logit_diff_std': float(np.std(logit_diffs)),
        'relative_sparsity': float(sparsity_metrics['relative_sparsity_pct']),
        'full_sparsity': float(sparsity_metrics['full_sparsity_pct']),
        'num_active_components': int(sparsity_metrics['num_active_components'])
    }

def run_validation(circuit, model, validation_loader, config, device, l1_weight, step, logger, run_dir, vis_dir,
                   metrics_history=None, metrics_history_file=None, plots_dir=None,
                   sparsity_progression=None, sparsity_progression_file=None):
    """
    Run validation and return metrics.

    Returns:
        dict with keys: avg_val_loss, avg_val_kl, current_l1_norm, should_stop, masked_acc, full_acc, exact_match
    """
    val_losses = []
    kl_losses = []
    masked_acc = []
    full_model_acc = []
    exact_match = []

    with torch.no_grad():
        for val_batch in tqdm(validation_loader, desc="Validation"):
            clean_column_name, corrupted_column_name = get_data_column_names(config['data_type'])
            indirect_objects_column_name, subjects_column_name = get_indirect_objects_and_subjects(config['data_type'])

            # Tokenize inputs
            input_ids_clean = model.tokenizer(
                val_batch[clean_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            input_ids_corrupted = model.tokenizer(
                val_batch[corrupted_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            # Add space before labels and convert to strings if needed
            # This handles both string labels (IOI, GP) and numeric labels (GT)
            # Handle indirect objects - convert to list if tensor
            indirect_objs = val_batch[indirect_objects_column_name]
            if isinstance(indirect_objs, torch.Tensor):
                indirect_objs = indirect_objs.tolist()
            val_batch[indirect_objects_column_name] = [' ' + str(obj) for obj in indirect_objs]
            indirect_objects = model.tokenizer(
                val_batch[indirect_objects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            # Handle subjects - convert to list if tensor
            subjs = val_batch[subjects_column_name]
            if isinstance(subjs, torch.Tensor):
                subjs = subjs.tolist()
            val_batch[subjects_column_name] = [' ' + str(subj) for subj in subjs]
            subjects = model.tokenizer(
                val_batch[subjects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            # Calculate sequence lengths
            clean_lengths = (input_ids_clean != model.tokenizer.pad_token_id).sum(dim=1)
            corrupt_lengths = (input_ids_corrupted != model.tokenizer.pad_token_id).sum(dim=1)

            clean_last_idx = clean_lengths - 1
            corrupt_last_idx = corrupt_lengths - 1

            # Create attention masks
            attention_mask_clean = torch.arange(input_ids_clean.size(1), device=device)[None, :] < clean_lengths[:, None]
            attention_mask_corrupted = torch.arange(input_ids_corrupted.size(1), device=device)[None, :] < corrupt_lengths[:, None]

            # Run corrupted forward pass
            corrupted_logits_val, corrupted_cache_val = model.run_with_cache(
                input_ids_corrupted,
                attention_mask=attention_mask_corrupted
            )

            # Extract corrupted activations
            corrupted_activations_val = extract_corrupted_activations(
                model=model,
                corrupted_cache=corrupted_cache_val,
                corrupt_last_idx=corrupt_last_idx,
                device=device
            )

            # Forward pass with masked model
            masked_logits = circuit.forward_pass_through_model(
                input_ids_clean,
                attention_mask_clean,
                corrupted_activations=corrupted_activations_val,
                clean_last_idx=clean_last_idx
            )

            # Forward pass with full model for comparison
            full_logits = model(input_ids_clean, attention_mask=attention_mask_clean)

            # Get the position of the last valid token in each sequence
            batch_indices = torch.arange(input_ids_clean.size(0), device=device)

            # Extract logits for the last token position (for next token prediction)
            full_last_token_logits = full_logits[batch_indices, clean_last_idx]
            masked_last_token_logits = masked_logits[batch_indices, clean_last_idx]

            # Compute KL divergence using the last token logits
            log_probs_full = F.log_softmax(full_last_token_logits, dim=-1)
            probs_full = torch.exp(log_probs_full)
            log_probs_masked = F.log_softmax(masked_last_token_logits, dim=-1)

            kl_div = F.kl_div(log_probs_masked, probs_full, reduction='batchmean')

            val_losses.append(kl_div.detach().cpu().item() + circuit.get_l1_penalty() * l1_weight)
            kl_losses.append(kl_div.detach().cpu().item())

            masked_accuracy = (masked_last_token_logits.argmax(dim=-1) == indirect_objects[:, 0]).float().mean().item()
            full_model_accuracy = (full_last_token_logits.argmax(dim=-1) == indirect_objects[:, 0]).float().mean().item()
            exact_match_value = (masked_last_token_logits.argmax(dim=-1) == full_last_token_logits.argmax(dim=-1)).float().mean().item()

            # Append to lists
            masked_acc.append(masked_accuracy)
            full_model_acc.append(full_model_accuracy)
            exact_match.append(exact_match_value)

    # Calculate average validation metrics
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_kl = np.mean(kl_losses)

    # Get L1 penalty (total number of active components)
    current_l1_norm = circuit.get_l1_penalty().item()

    logger.info(f"Validation loss: {avg_val_loss:.4f}, KL: {avg_val_kl:.4f}, L1 norm: {current_l1_norm:.1f}")

    if config['use_wandb']:
        wandb.log({
            'validation/kl_div': avg_val_kl,
            'validation/loss': avg_val_loss,
            'validation/l1_norm': current_l1_norm,
            'validation/masked_accuracy': np.mean(masked_acc),
            'validation/full_model_accuracy': np.mean(full_model_acc),
            'validation/exact_match': np.mean(exact_match)
        }, step=step)

    # Check early stopping criteria
    kl_opt = config['training'].get('kl_opt', 0.15)
    l1_opt = config['training'].get('l1_opt', 900)

    should_stop = False
    if avg_val_kl < kl_opt and current_l1_norm < l1_opt:
        logger.info(f"🎉 Early stopping criteria met!")
        logger.info(f"   KL divergence: {avg_val_kl:.4f} < {kl_opt}")
        logger.info(f"   L1 norm: {current_l1_norm:.1f} < {l1_opt}")

        # Get sparsity statistics
        sparsity_stats = circuit.get_sparsity_stats()
        sparsity_threshold = config['masking'].get('sparsity_threshold', 1e-3)
        sparsity_metrics = circuit.get_relative_and_full_sparsity(threshold=sparsity_threshold)

        logger.info(f"   Overall sparsity: {sparsity_stats['total_sparsity']:.2%}")
        logger.info(f"   Active components: {sparsity_metrics['num_active_components']}")
        logger.info(f"   Relative sparsity: {sparsity_metrics['relative_sparsity_pct']:.2f}% "
                   f"({sparsity_metrics['num_active_components']}/{sparsity_metrics['relative_denominator']})")
        logger.info(f"   Full sparsity: {sparsity_metrics['full_sparsity_pct']:.2f}% "
                   f"({sparsity_metrics['num_active_components']}/{sparsity_metrics['full_denominator']})")

        # Save optimal model to run directory
        optimal_path = os.path.join(run_dir, "model_optimal.pt")
        torch.save({
            'step': step,
            'qk_masks': circuit.qk_masks,
            'ov_masks': circuit.ov_masks,
            'mlp_in_masks': circuit.mlp_in_masks,
            'mlp_out_masks': circuit.mlp_out_masks,
            'sparsity_stats': sparsity_stats,
            'sparsity_metrics': sparsity_metrics,
            'val_kl': avg_val_kl,
            'val_loss': avg_val_loss,
            'l1_norm': current_l1_norm,
            'config': config,
        }, optimal_path)
        logger.info(f"✓ Saved optimal model to {optimal_path}")

        # Create final visualizations
        logger.info("Creating final visualizations...")
        circuit.visualize_masks(output_dir=vis_dir, data_type=config['data_type'])

        logger.info("Creating masked singular value visualizations...")
        circuit.visualize_masked_singular_values(output_dir=vis_dir)

        # Run test evaluation
        logger.info("Running test evaluation...")
        data_loader_fn = getattr(local_data_loader, f"load_{config['data_type']}_dataset")
        test_loader = data_loader_fn(
            batch_size=config['training']['batch_size'],
            train=False,
            validation=False,
            shuffle=False
        )
        test_metrics = run_test_evaluation(circuit, model, test_loader, config, device, logger)
        logger.info(f"Test KL: {test_metrics['test_kl']:.6f}, "
                   f"Test Logit Diff: {test_metrics['test_logit_diff']:.4f}, "
                   f"Test Masked Accuracy: {test_metrics['test_masked_acc']:.4f}, "
                   f"Test Exact Match: {test_metrics['test_exact_match']:.4f}, "
                   f"Test Masked Exact Match: {test_metrics['test_masked_exact_match']:.4f}")

        # Add to metrics history and create plots
        test_metrics['epoch'] = step  # or could use epoch number
        test_metrics['step'] = step
        metrics_history.append(test_metrics)

        # Save metrics history
        with open(metrics_history_file, 'w') as f:
            json.dump(list(metrics_history), f, indent=2)

        # Create plots
        if len(metrics_history) > 0:
            logger.info("Creating sparsity vs metrics plots...")
            create_sparsity_plots(list(metrics_history), plots_dir, logger)

        # Create sparsity progression plots
        if sparsity_progression is not None and len(sparsity_progression) > 0:
            logger.info("Creating sparsity progression plots...")
            with open(sparsity_progression_file, 'w') as f:
                json.dump(sparsity_progression, f, indent=2)
            create_sparsity_progression_plots(sparsity_progression, plots_dir, logger)

        # Create mask value plots
        logger.info("Creating average mask value plots for QK and OV circuits...")
        create_mask_value_plots(circuit, plots_dir, config['data_type'], logger)

        # Save statistics to JSON
        stats_path = os.path.join(run_dir, "run_summary.json")
        with open(stats_path, 'w') as f:
            json.dump({
                'run_name': os.path.basename(run_dir),
                'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                'optimal_conditions_met': True,
                'step': step,
                'validation_kl': float(avg_val_kl),
                'validation_loss': float(avg_val_loss),
                'l1_norm': float(current_l1_norm),
                'masked_accuracy': float(np.mean(masked_acc)),
                'full_model_accuracy': float(np.mean(full_model_acc)),
                'exact_match': float(np.mean(exact_match)),
                'test_evaluation': test_metrics,  # Add test metrics
                'sparsity_stats': sparsity_stats,
                'sparsity_metrics': {
                    'num_active_components': int(sparsity_metrics['num_active_components']),
                    'relative_sparsity': float(sparsity_metrics['relative_sparsity']),
                    'full_sparsity': float(sparsity_metrics['full_sparsity']),
                    'relative_sparsity_pct': float(sparsity_metrics['relative_sparsity_pct']),
                    'full_sparsity_pct': float(sparsity_metrics['full_sparsity_pct']),
                    'relative_denominator': int(sparsity_metrics['relative_denominator']),
                    'full_denominator': int(sparsity_metrics['full_denominator']),
                    'sparsity_threshold': float(sparsity_threshold),
                },
                'kl_threshold': kl_opt,
                'l1_threshold': l1_opt,
                'hyperparameters': {
                    'learning_rate': config['training']['learning_rate'],
                    'l1_weight': config['training']['l1_weight'],
                    'weight_decay': config['training']['weight_decay'],
                    'batch_size': config['training']['batch_size'],
                    # 'target_kl': config['training']['target_kl'],
                    'mask_init_value': config['masking']['mask_init_value'],
                    'sparsity_threshold': float(sparsity_threshold),
                },
                'config': config  # Add full config
            }, f, indent=2)
        logger.info(f"✓ Saved run summary to {stats_path}")

        # Write to central log file in svd_logs/
        central_log_path = os.path.join("svd_logs", "optimal_runs_log.txt")
        with open(central_log_path, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Optimal Model Found: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Run Directory: {run_dir}\n")
            f.write(f"Step: {step}\n")
            f.write(f"\n--- Optimal Criteria ---\n")
            f.write(f"KL Divergence: {avg_val_kl:.6f} (threshold: {kl_opt})\n")
            f.write(f"L1 Norm: {current_l1_norm:.2f} (threshold: {l1_opt})\n")
            f.write(f"\n--- Validation Metrics ---\n")
            f.write(f"Validation Loss: {avg_val_loss:.6f}\n")
            f.write(f"Val Masked Accuracy: {np.mean(masked_acc):.4f}\n")
            f.write(f"Val Full Model Accuracy: {np.mean(full_model_acc):.4f}\n")
            f.write(f"Val Exact Match: {np.mean(exact_match):.4f}\n")
            f.write(f"\n--- Test Metrics ---\n")
            f.write(f"Test KL Divergence: {test_metrics['test_kl']:.6f} ± {test_metrics['test_kl_std']:.6f}\n")
            f.write(f"Test Masked Accuracy: {test_metrics['test_masked_acc']:.4f} ± {test_metrics['test_masked_acc_std']:.4f}\n")
            f.write(f"Test Full Model Accuracy: {test_metrics['test_full_acc']:.4f} ± {test_metrics['test_full_acc_std']:.4f}\n")
            f.write(f"Test Exact Match (Masked vs Full): {test_metrics['test_exact_match']:.4f} ± {test_metrics['test_exact_match_std']:.4f}\n")
            f.write(f"Test Masked Exact Match (Correct Predictions): {test_metrics['test_masked_exact_match']:.4f} ± {test_metrics['test_masked_exact_match_std']:.4f}\n")
            f.write(f"\n--- Sparsity Statistics ---\n")
            f.write(f"Overall Sparsity: {sparsity_stats['total_sparsity']:.2%}\n")
            f.write(f"Active Components: {sparsity_metrics['num_active_components']}\n")
            f.write(f"Relative Sparsity: {sparsity_metrics['relative_sparsity_pct']:.2f}% "
                   f"({sparsity_metrics['num_active_components']}/{sparsity_metrics['relative_denominator']})\n")
            f.write(f"Full Sparsity: {sparsity_metrics['full_sparsity_pct']:.2f}% "
                   f"({sparsity_metrics['num_active_components']}/{sparsity_metrics['full_denominator']})\n")
            f.write("\n\n")

        logger.info(f"✓ Logged to central file: {central_log_path}")

        if config['use_wandb']:
            wandb.log({
                'optimal/kl_div': avg_val_kl,
                'optimal/l1_norm': current_l1_norm,
                'optimal/sparsity': sparsity_stats['total_sparsity'],
                'optimal/num_active_components': sparsity_metrics['num_active_components'],
                'optimal/relative_sparsity': sparsity_metrics['relative_sparsity'],
                'optimal/full_sparsity': sparsity_metrics['full_sparsity'],
                'test/kl_div': test_metrics['test_kl'],
                'test/masked_accuracy': test_metrics['test_masked_acc'],
                'test/full_accuracy': test_metrics['test_full_acc'],
                'test/exact_match': test_metrics['test_exact_match'],
            }, step=step)

        should_stop = True

    return {
        'avg_val_loss': avg_val_loss,
        'avg_val_kl': avg_val_kl,
        'current_l1_norm': current_l1_norm,
        'should_stop': should_stop,
        'masked_acc': np.mean(masked_acc),
        'full_acc': np.mean(full_model_acc),
        'exact_match': np.mean(exact_match)
    }


def train_circuit(config, logger):
    """Train the SVD-based circuit discovery model"""
    import json
    import datetime

    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create run-specific directory structure
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['experiment_name']}_{timestamp}"
    run_dir = os.path.join("svd_logs", run_name)

    # Create subdirectories
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Visualizations will be saved to: {vis_dir}")

    # Update config to use run-specific directories
    config['run_dir'] = run_dir
    config['vis_dir'] = vis_dir
    
    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    model = HookedTransformer.from_pretrained(
        config['model']['name'],
        cache_dir=config['model']['pretrained_cache_dir']
    )
    model = model.to(device)
    
    # Parse train_masks if provided
    train_masks = None
    if 'train_masks' in config and config['train_masks'] is not None:
        train_masks = config['train_masks']
        logger.info(f"Training masks: {train_masks}")

    # Create circuit analyzer
    logger.info("Initializing circuit analyzer")
    circuit = MaskedTransformerCircuit(
        model=model,
        device=device,
        cache_svd=config['masking']['cache_svd'],
        mask_init_value=config['masking']['mask_init_value'],
        train_masks=train_masks
    )
    
    # Load data
    logger.info(f"Loading {config['data_type']} dataset")
    data_loader_fn = getattr(local_data_loader, f"load_{config['data_type']}_dataset")
    train_loader = data_loader_fn(batch_size=config['training']['batch_size'], train=True)
    validation_loader = data_loader_fn(batch_size=config['training']['batch_size'], train=False, validation=True)
    
    logger.info(f"Training with {len(train_loader)} batches")
    
    # Initialize WandB if enabled
    if config['use_wandb']:
        wandb.init(
            project="svd_circuit_discovery",
            name=config['experiment_name'],
            config=config
        )
    
    # Create visualization directory if needed
    if config['visualization']['save_visualizations']:
        os.makedirs(os.path.join(config['log_dir'], config['visualization']['visualization_dir']), exist_ok=True)
    
    # Training loop
    logger.info("Starting training")
    step = 0
    best_val_loss = float('inf')
    last_val_metrics = None  # Store last validation metrics for final summary

    # Create metrics history for last 10 epochs (for plotting)
    metrics_history = deque(maxlen=10)
    metrics_history_file = os.path.join(run_dir, "metrics_history.json")

    # Create sparsity progression tracker (tracks all training steps)
    sparsity_progression = []
    sparsity_progression_file = os.path.join(run_dir, "sparsity_progression.json")
    sparsity_track_interval = 10  # Track sparsity every N batches

    # Create monitoring plots directory
    plots_dir = os.path.join(run_dir, "monitoring_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get validation interval from config (-1 means only at epoch end)
    validation_interval = config['training'].get('validation_interval', -1)
    logger.info(f"Validation interval: {'epoch end only' if validation_interval == -1 else f'every {validation_interval} steps'}")

    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        l1_weight = config['training']['l1_weight']

        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            
            clean_column_name, corrupted_column_name = get_data_column_names(config['data_type'])
            indirect_objects_column_name, subjects_column_name = get_indirect_objects_and_subjects(config['data_type'])
            
            # Tokenize clean inputs
            input_ids_clean = model.tokenizer(
                batch[clean_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)
            
            # Tokenize corrupted inputs
            input_ids_corrupted = model.tokenizer(
                batch[corrupted_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)
            
            # Add space before labels and convert to strings if needed
            # This handles both string labels (IOI, GP) and numeric labels (GT)
            # Handle indirect objects - convert to list if tensor
            indirect_objs = batch[indirect_objects_column_name]
            if isinstance(indirect_objs, torch.Tensor):
                indirect_objs = indirect_objs.tolist()
            batch[indirect_objects_column_name] = [' ' + str(obj) for obj in indirect_objs]

            # Handle subjects - convert to list if tensor
            subjs = batch[subjects_column_name]
            if isinstance(subjs, torch.Tensor):
                subjs = subjs.tolist()
            batch[subjects_column_name] = [' ' + str(subj) for subj in subjs]

            indirect_objects = model.tokenizer(
                batch[indirect_objects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            subjects = model.tokenizer(
                batch[subjects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)
            
            # Calculate sequence lengths
            clean_lengths = (input_ids_clean != model.tokenizer.pad_token_id).sum(dim=1)
            corrupt_lengths = (input_ids_corrupted != model.tokenizer.pad_token_id).sum(dim=1)
            
            # Get last token indices (before EOS/padding)
            clean_last_idx = clean_lengths - 1
            corrupt_last_idx = corrupt_lengths - 1
            
            # Create attention masks
            attention_mask_clean = torch.arange(input_ids_clean.size(1), device=device)[None, :] < clean_lengths[:, None]
            attention_mask_corrupted = torch.arange(input_ids_corrupted.size(1), device=device)[None, :] < corrupt_lengths[:, None]
            
            # Run corrupted forward pass and cache activations
            with torch.no_grad():
                corrupted_logits, corrupted_cache = model.run_with_cache(
                    input_ids_corrupted,
                    attention_mask=attention_mask_corrupted
                )
            
            # Extract corrupted activations at last token positions
            corrupted_activations = extract_corrupted_activations(
                model=model,
                corrupted_cache=corrupted_cache,
                corrupt_last_idx=corrupt_last_idx,
                device=device
            )
            
            # Train masks for this batch with corrupted activations
            history = circuit.train_masks(
                input_ids=input_ids_clean,
                attention_mask=attention_mask_clean,
                sequence_lengths=clean_lengths,
                num_iterations=config['masking']['max_iterations_per_batch'],
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                l1_weight=l1_weight,
                # target_kl=config['training']['target_kl'],
                temperature=config['training']['temperature'],
                eval_interval=config['training']['eval_interval'],
                patience=config['training']['patience'],
                min_lr=config['training']['min_lr'],
                lr_factor=config['training']['lr_factor'],
                indirect_objects=indirect_objects,
                subjects=subjects,
                corrupted_activations=corrupted_activations,  # NEW: Pass corrupted activations
                clean_last_idx=clean_last_idx  # NEW: Pass last token indices
            )
            
            # Log metrics (only if history is not empty)
            if len(history['loss']) > 0:
                metrics = {
                    'train/loss': history['loss'][-1],
                    'train/kl_div': history['kl_div'][-1],
                    'train/l1_penalty': history['l1_penalty'][-1],
                    'train/l1_coeff': l1_weight,
                    'train/accuracy': history['accuracy'][-1],
                    'train/exact_match': history['exact_match'][-1],
                    'train/masked_accuracy': history['masked_accuracy'][-1],
                }

                if config['use_wandb']:
                    wandb.log(metrics, step=step)

                # Log to console periodically
                if batch_idx % 10 == 0:
                    logger.info(f"Step {step}: Loss={metrics['train/loss']:.4f}, KL={metrics['train/kl_div']:.4f}")

                # Track sparsity progression periodically
                if step % sparsity_track_interval == 0:
                    sparsity_threshold = config['masking'].get('sparsity_threshold', 1e-3)
                    sparsity_metrics = circuit.get_relative_and_full_sparsity(threshold=sparsity_threshold)

                    sparsity_progression.append({
                        'step': step,
                        'epoch': epoch,
                        'batch': batch_idx,
                        'train_loss': history['loss'][-1],
                        'train_kl': history['kl_div'][-1],
                        'train_accuracy': history['masked_accuracy'][-1],
                        'l1_penalty': history['l1_penalty'][-1],
                        'relative_sparsity': float(sparsity_metrics['relative_sparsity_pct']),
                        'full_sparsity': float(sparsity_metrics['full_sparsity_pct']),
                        'num_active_components': int(sparsity_metrics['num_active_components'])
                    })

                    # Save progression periodically
                    if step % (sparsity_track_interval * 10) == 0:
                        with open(sparsity_progression_file, 'w') as f:
                            json.dump(sparsity_progression, f, indent=2)
                        # Create progression plots
                        create_sparsity_progression_plots(sparsity_progression, plots_dir, logger)

            else:
                logger.warning(f"Step {step}: Batch {batch_idx} - No training history recorded (likely due to early NaN detection)")

            # Create visualizations periodically (skip intermediate ones, only save final)
            # Intermediate visualizations are commented out to reduce overhead

            # Increment step counter
            step += 1

            # Step-based validation (if configured)
            if validation_interval > 0 and step % validation_interval == 0 and validation_loader is not None:
                logger.info(f"Running validation at step {step}")
                result = run_validation(
                    circuit, model, validation_loader, config, device,
                    l1_weight, step, logger, run_dir, vis_dir,
                    metrics_history, metrics_history_file, plots_dir,
                    sparsity_progression, sparsity_progression_file
                )
                last_val_metrics = result  # Store for final summary

                # Check if optimal conditions met
                if result['should_stop']:
                    logger.info("Training stopped early - optimal model found!")
                    if config['use_wandb']:
                        wandb.finish()
                    return circuit

                # Update best validation loss
                if result['avg_val_loss'] < best_val_loss and config['training']['save_checkpoints']:
                    best_val_loss = result['avg_val_loss']
                    checkpoint_path = os.path.join(config['log_dir'], f"{config['experiment_name']}_best.pt")
                    torch.save({
                        'step': step,
                        'qk_masks': circuit.qk_masks,
                        'ov_masks': circuit.ov_masks,
                        'mlp_in_masks': circuit.mlp_in_masks,
                        'mlp_out_masks': circuit.mlp_out_masks,
                        'sparsity_stats': circuit.get_sparsity_stats(),
                        'val_loss': best_val_loss,
                        'config': config,
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Validation at the end of each epoch (if not using step-based validation, or in addition to it)
        if validation_loader is not None:
            logger.info(f"Running end-of-epoch validation (Epoch {epoch+1})")
            result = run_validation(
                circuit, model, validation_loader, config, device,
                l1_weight, step, logger, run_dir, vis_dir,
                metrics_history, metrics_history_file, plots_dir,
                sparsity_progression, sparsity_progression_file
            )
            last_val_metrics = result  # Store for final summary

            # Check if optimal conditions met
            if result['should_stop']:
                logger.info("Training stopped early - optimal model found!")
                if config['use_wandb']:
                    wandb.finish()
                return circuit

            # Save checkpoint if validation loss improved
            if result['avg_val_loss'] < best_val_loss and config['training']['save_checkpoints']:
                best_val_loss = result['avg_val_loss']
                checkpoint_path = os.path.join(config['log_dir'], f"{config['experiment_name']}_best.pt")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'qk_masks': circuit.qk_masks,
                    'ov_masks': circuit.ov_masks,
                    'mlp_in_masks': circuit.mlp_in_masks,
                    'mlp_out_masks': circuit.mlp_out_masks,
                    'sparsity_stats': circuit.get_sparsity_stats(),
                    'val_loss': best_val_loss,
                    'config': config,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation (training completed without meeting optimal conditions)
    logger.info("Training complete, running final evaluation")

    # Get sparsity statistics
    sparsity_stats = circuit.get_sparsity_stats()
    sparsity_threshold = config['masking'].get('sparsity_threshold', 1e-3)
    sparsity_metrics = circuit.get_relative_and_full_sparsity(threshold=sparsity_threshold)

    logger.info(f"Overall sparsity: {sparsity_stats['total_sparsity']:.2%}")
    logger.info(f"Active components: {sparsity_metrics['num_active_components']}")
    logger.info(f"Relative sparsity: {sparsity_metrics['relative_sparsity_pct']:.2f}% "
               f"({sparsity_metrics['num_active_components']}/{sparsity_metrics['relative_denominator']})")
    logger.info(f"Full sparsity: {sparsity_metrics['full_sparsity_pct']:.2f}% "
               f"({sparsity_metrics['num_active_components']}/{sparsity_metrics['full_denominator']})")

    # Save final model to run directory
    final_path = os.path.join(config['run_dir'], "model_final.pt")

    torch.save({
        'step': step,
        'mlp_in_masks': circuit.mlp_in_masks,
        'mlp_out_masks': circuit.mlp_out_masks,
        'qk_masks': circuit.qk_masks,
        'ov_masks': circuit.ov_masks,
        'sparsity_stats': sparsity_stats,
        'config': config,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    # Final visualizations
    logger.info("Creating final mask visualizations...")
    circuit.visualize_masks(output_dir=config['vis_dir'], data_type=config['data_type'])

    logger.info("Creating masked singular value visualizations...")
    circuit.visualize_masked_singular_values(output_dir=config['vis_dir'])

    # Run test evaluation
    logger.info("Running test evaluation...")
    data_loader_fn = getattr(local_data_loader, f"load_{config['data_type']}_dataset")
    test_loader = data_loader_fn(
        batch_size=config['training']['batch_size'],
        train=False,
        validation=False,
        shuffle=False
    )
    test_metrics = run_test_evaluation(circuit, model, test_loader, config, device, logger)
    logger.info(f"Test KL: {test_metrics['test_kl']:.6f}, "
               f"Test Logit Diff: {test_metrics['test_logit_diff']:.4f}, "
               f"Test Masked Accuracy: {test_metrics['test_masked_acc']:.4f}, "
               f"Test Exact Match (Masked vs Full): {test_metrics['test_exact_match']:.4f}, "
               f"Test Masked Exact Match (Correct): {test_metrics['test_masked_exact_match']:.4f}")

    # Add to metrics history and create plots
    test_metrics['epoch'] = step  # or could use epoch number
    test_metrics['step'] = step
    metrics_history.append(test_metrics)

    # Save metrics history
    with open(metrics_history_file, 'w') as f:
        json.dump(list(metrics_history), f, indent=2)

    # Create final plots
    if len(metrics_history) > 0:
        logger.info("Creating final sparsity vs metrics plots...")
        create_sparsity_plots(list(metrics_history), plots_dir, logger)

    # Create final sparsity progression plots
    if len(sparsity_progression) > 0:
        logger.info("Creating final sparsity progression plots...")
        with open(sparsity_progression_file, 'w') as f:
            json.dump(sparsity_progression, f, indent=2)
        create_sparsity_progression_plots(sparsity_progression, plots_dir, logger)

    # Create final mask value plots
    logger.info("Creating final average mask value plots for QK and OV circuits...")
    create_mask_value_plots(circuit, plots_dir, config['data_type'], logger)

    # Save final run summary
    final_summary_path = os.path.join(config['run_dir'], "run_summary.json")

    summary_data = {
        'run_name': run_name,
        'timestamp': timestamp,
        'optimal_conditions_met': False,
        'training_completed': True,
        'final_step': step,
        'sparsity_stats': sparsity_stats,
        'sparsity_metrics': {
            'num_active_components': int(sparsity_metrics['num_active_components']),
            'relative_sparsity': float(sparsity_metrics['relative_sparsity']),
            'full_sparsity': float(sparsity_metrics['full_sparsity']),
            'relative_sparsity_pct': float(sparsity_metrics['relative_sparsity_pct']),
            'full_sparsity_pct': float(sparsity_metrics['full_sparsity_pct']),
            'relative_denominator': int(sparsity_metrics['relative_denominator']),
            'full_denominator': int(sparsity_metrics['full_denominator']),
            'sparsity_threshold': float(sparsity_threshold),
        },
        'hyperparameters': {
            'learning_rate': config['training']['learning_rate'],
            'l1_weight': config['training']['l1_weight'],
            'weight_decay': config['training']['weight_decay'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
            'sparsity_threshold': float(sparsity_threshold),
        },
        'config': config  # Add full config
    }

    # Add last validation metrics if available
    if last_val_metrics is not None:
        summary_data.update({
            'validation_kl': float(last_val_metrics['avg_val_kl']),
            'validation_loss': float(last_val_metrics['avg_val_loss']),
            'l1_norm': float(last_val_metrics['current_l1_norm']),
            'masked_accuracy': float(last_val_metrics['masked_acc']),
            'full_model_accuracy': float(last_val_metrics['full_acc']),
            'exact_match': float(last_val_metrics['exact_match']),
            'kl_threshold': config['training'].get('kl_opt', 0.15),
            'l1_threshold': config['training'].get('l1_opt', 900),
        })

    # Add test metrics
    summary_data['test_evaluation'] = test_metrics

    with open(final_summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved final run summary to {final_summary_path}")

    # Close wandb
    if config['use_wandb']:
        wandb.finish()

    return circuit

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.log_dir:
        config['log_dir'] = args.log_dir
    if args.use_wandb:
        config['use_wandb'] = True
    if args.train_masks:
        # Parse comma-separated list into Python list
        config['train_masks'] = [mask.strip() for mask in args.train_masks.split(',')]
    
    # Setup logging
    logger = setup_logging(config['log_dir'], config['experiment_name'])
    
    # Log configuration
    logger.info("Configuration:")
    for section, params in config.items():
        if isinstance(params, dict):
            logger.info(f"  {section}:")
            for key, value in params.items():
                logger.info(f"    {key}: {value}")
        else:
            logger.info(f"  {section}: {params}")
    
    # Train circuit
    train_circuit(config, logger)

if __name__ == "__main__":
    main()