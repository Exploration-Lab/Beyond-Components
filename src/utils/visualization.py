"""
Visualization utilities for masked transformer circuits.

This module contains functions to visualize learned masks and training history
for masked transformer circuits with improved aesthetics and readability.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
from typing import Dict, Optional, Any


# Set seaborn style for better aesthetics with darker, crisper look
sns.set_style("whitegrid", {
    'grid.linewidth': 1.2,
    'grid.alpha': 0.4,
    'axes.edgecolor': '0.2',
    'axes.linewidth': 1.5
})
sns.set_context("paper", font_scale=1.3)

# Darker, bolder color palette for improved contrast
COLORS = {
    'qk': '#1a5490',      # Dark Blue
    'ov': '#c0392b',      # Dark Red
    'mlp_in': '#1e7e34',  # Dark Green
    'mlp_out': '#6c3483', # Dark Purple
    'original': '#5d6d7e', # Dark Gray
    'loss': '#1c2833',    # Very Dark Blue-gray
    'kl': '#d35400',      # Dark Orange
    'l1': '#117a65'       # Dark Teal
}

# IOI head categories with colors (for Indirect Object Identification task)
# Using a vibrant, publication-quality color palette
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
    """Get color and category for a head based on IOI circuit categories"""
    for category, info in IOI_HEAD_CATEGORIES.items():
        if (layer, head) in info['heads']:
            return info['color'], category
    return IOI_HEAD_CATEGORIES['Other Heads']['color'], 'Other Heads'


def visualize_masks(circuit, output_dir=None, mask_fn=None, dpi=300, data_type='gp'):
    """
    Visualize learned masks for all attention heads and MLP layers.

    Args:
        circuit: MaskedTransformerCircuit instance
        output_dir: Directory to save plots (optional)
        mask_fn: Mask function to apply (default: torch.sigmoid)
        dpi: Resolution for saved images (default: 300)
    """
    if mask_fn is None:
        mask_fn = torch.sigmoid

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Check if IOI task for color coding
    is_ioi = (data_type.lower() == 'ioi')

    # Create subplots for QK masks with improved aesthetics
    # Add extra space at bottom for legend if IOI
    fig_height_qk = circuit.n_layers * 3 + (1.5 if is_ioi else 0)
    fig_qk, axes_qk = plt.subplots(
        circuit.n_layers, circuit.n_heads,
        figsize=(circuit.n_heads * 4, fig_height_qk),
        squeeze=False
    )
    # Remove title as per user request

    # Create subplots for OV masks
    fig_height_ov = circuit.n_layers * 3 + (1.5 if is_ioi else 0)
    fig_ov, axes_ov = plt.subplots(
        circuit.n_layers, circuit.n_heads,
        figsize=(circuit.n_heads * 4, fig_height_ov),
        squeeze=False
    )
    # Remove title as per user request

    # Plot masks for each head
    for layer in range(circuit.n_layers):
        for head in range(circuit.n_heads):
            head_key = f'differential_head_{layer}_{head}'

            # Get mask values
            if circuit.l1_reg:
                qk_mask = mask_fn(circuit.qk_masks[head_key]).detach().cpu().numpy()
                ov_mask = mask_fn(circuit.ov_masks[head_key]).detach().cpu().numpy()
            else:
                qk_mask, ov_mask = circuit.sample_hard_concrete_masks()
                qk_mask = qk_mask[head_key].detach().cpu().numpy()
                ov_mask = ov_mask[head_key].detach().cpu().numpy()

            # Determine color based on IOI category if applicable
            if is_ioi:
                head_color, head_category = get_head_color_ioi(layer, head)
            else:
                head_color = COLORS['qk']
                head_category = None

            # Plot QK mask with improved styling
            ax_qk = axes_qk[layer, head]
            x_positions = np.arange(len(qk_mask))
            # bars_qk = ax_qk.bar(x_positions, qk_mask,
            #                     color=head_color, alpha=0.9,
            #                     edgecolor='black', linewidth=1.2)
            bars_qk = ax_qk.bar(x_positions, qk_mask,
                    color=head_color, alpha=0.7,  
                    edgecolor='none')

            # Add subtle gradient effect
            for i, bar in enumerate(bars_qk):
                bar.set_alpha(0.7 + 0.3 * qk_mask[i])

            # Add circular markers at the top of each bar
            ax_qk.plot(x_positions, qk_mask, 'o',
                      color=head_color, markersize=7,
                      markeredgecolor='black', markeredgewidth=1.5,
                      zorder=5)

            # Add IOI category to title if applicable
            title_str = f'Layer {layer}, Head {head}'
            if is_ioi and head_category and head_category != 'Other Heads':
                title_str += f'\n({head_category})'
            ax_qk.set_title(title_str,
                           fontsize=16, fontweight='bold', pad=10)
            ax_qk.set_ylim(0, 1.05)
            ax_qk.axhline(y=0.5, color='gray', linestyle='--',
                         alpha=0.4, linewidth=1.5)

            if head == 0:
                ax_qk.set_ylabel('Mask Value', fontsize=16, fontweight='bold')
            if layer == circuit.n_layers - 1:
                ax_qk.set_xlabel('Singular Value Index',
                               fontsize=16, fontweight='bold')

            # Improve grid
            ax_qk.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_qk.set_axisbelow(True)

            # Plot OV mask with improved styling (use same color as QK for consistency)
            ax_ov = axes_ov[layer, head]
            x_positions_ov = np.arange(len(ov_mask))
            # Use slightly darker version of the same IOI color for OV
            # bars_ov = ax_ov.bar(x_positions_ov, ov_mask,
            #                     color=head_color, alpha=0.9,
            #                     edgecolor='black', linewidth=1.2)
            
            bars_ov = ax_ov.bar(x_positions_ov, ov_mask,
                    color=head_color, alpha=0.7,  
                    edgecolor='none')   

            # Add subtle gradient effect
            for i, bar in enumerate(bars_ov):
                bar.set_alpha(0.7 + 0.3 * ov_mask[i])

            # Add circular markers at the top of each bar
            ax_ov.plot(x_positions_ov, ov_mask, 'o',
                      color=head_color, markersize=7,
                      markeredgecolor='black', markeredgewidth=1.5,
                      zorder=5)

            # Add IOI category to title if applicable
            title_str_ov = f'Layer {layer}, Head {head}'
            if is_ioi and head_category and head_category != 'Other Heads':
                title_str_ov += f'\n({head_category})'
            ax_ov.set_title(title_str_ov,
                           fontsize=16, fontweight='bold', pad=10)
            ax_ov.set_ylim(0, 1.05)
            ax_ov.axhline(y=0.5, color='gray', linestyle='--',
                         alpha=0.4, linewidth=1.5)

            if head == 0:
                ax_ov.set_ylabel('Mask Value', fontsize=16, fontweight='bold')
            if layer == circuit.n_layers - 1:
                ax_ov.set_xlabel('Singular Value Index',
                               fontsize=16, fontweight='bold')

            # Improve grid
            ax_ov.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_ov.set_axisbelow(True)

    # Adjust layout (leave space at bottom for legend if IOI)
    if is_ioi:
        fig_qk.tight_layout(rect=[0, 0.04, 1, 0.99])
        fig_ov.tight_layout(rect=[0, 0.04, 1, 0.99])
    else:
        fig_qk.tight_layout(rect=[0, 0, 1, 0.99])
        fig_ov.tight_layout(rect=[0, 0, 1, 0.99])

    # Issue A fix: Add legend at bottom for IOI plots
    if is_ioi:
        from matplotlib.patches import Patch
        legend_elements = []
        for category, info in IOI_HEAD_CATEGORIES.items():
            if category != 'Other Heads':  # Skip 'Other Heads' in legend
                legend_elements.append(Patch(facecolor=info['color'],
                                            edgecolor='black',
                                            label=category))

        # Add legend at bottom center of QK figure
        fig_qk.legend(handles=legend_elements,
                     loc='lower center',
                     bbox_to_anchor=(0.5, 0.0),
                     ncol=min(4, len(legend_elements)),
                     fontsize=36,
                     frameon=True,
                     fancybox=True,
                     shadow=True)

        # Add legend at bottom center of OV figure
        fig_ov.legend(handles=legend_elements,
                     loc='lower center',
                     bbox_to_anchor=(0.5, 0.0),
                     ncol=min(4, len(legend_elements)),
                     fontsize=36,
                     frameon=True,
                     fancybox=True,
                     shadow=True)

    # Save if output directory provided
    if output_dir is not None:
        fig_qk.savefig(os.path.join(output_dir, 'qk_masks.pdf'),
                       bbox_inches='tight', dpi=dpi)
        fig_qk.savefig(os.path.join(output_dir, 'qk_masks.png'),
                       bbox_inches='tight', dpi=dpi)
        fig_ov.savefig(os.path.join(output_dir, 'ov_masks.pdf'),
                       bbox_inches='tight', dpi=dpi)
        fig_ov.savefig(os.path.join(output_dir, 'ov_masks.png'),
                       bbox_inches='tight', dpi=dpi)

    # Visualize MLP masks if enabled
    if circuit.mask_mlp:
        fig_mlp, axes_mlp = plt.subplots(
            circuit.n_layers, 2,
            figsize=(10, circuit.n_layers * 3),
            squeeze=False
        )
        # Remove title as per user request

        if circuit.l1_reg:
            mlp_in_masks = circuit.mlp_in_masks
            mlp_out_masks = circuit.mlp_out_masks
        else:
            mlp_in_masks, mlp_out_masks = circuit.sample_mlp_hard_concrete_masks()

        for layer in range(circuit.n_layers):
            mlp_key = f'mlp_{layer}'

            if circuit.l1_reg:
                mlp_in_mask = mask_fn(mlp_in_masks[mlp_key]).detach().cpu().numpy()
                mlp_out_mask = mask_fn(mlp_out_masks[mlp_key]).detach().cpu().numpy()
            else:
                mlp_in_mask = mlp_in_masks[mlp_key].detach().cpu().numpy()
                mlp_out_mask = mlp_out_masks[mlp_key].detach().cpu().numpy()

            # Plot MLP input mask
            ax_in = axes_mlp[layer, 0]
            x_positions_in = np.arange(len(mlp_in_mask))
            bars_in = ax_in.bar(x_positions_in, mlp_in_mask,
                               color=COLORS['mlp_in'], alpha=0.9,
                               edgecolor='black', linewidth=1.2)

            # Add gradient effect
            for i, bar in enumerate(bars_in):
                bar.set_alpha(0.7 + 0.3 * mlp_in_mask[i])

            # Add circular markers at the top of each bar
            ax_in.plot(x_positions_in, mlp_in_mask, 'o',
                      color=COLORS['mlp_in'], markersize=7,
                      markeredgecolor='black', markeredgewidth=1.5,
                      zorder=5)

            ax_in.set_title(f'Layer {layer} - Input Projection',
                           fontsize=16, fontweight='bold', pad=10)
            ax_in.set_ylim(0, 1.05)
            ax_in.axhline(y=0.5, color='gray', linestyle='--',
                         alpha=0.4, linewidth=1.5)
            ax_in.set_ylabel('Mask Value', fontsize=16, fontweight='bold')

            if layer == circuit.n_layers - 1:
                ax_in.set_xlabel('Singular Value Index',
                               fontsize=16, fontweight='bold')

            ax_in.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_in.set_axisbelow(True)

            # Plot MLP output mask
            ax_out = axes_mlp[layer, 1]
            x_positions_out = np.arange(len(mlp_out_mask))
            bars_out = ax_out.bar(x_positions_out, mlp_out_mask,
                                 color=COLORS['mlp_out'], alpha=0.9,
                                 edgecolor='black', linewidth=1.2)

            # Add gradient effect
            for i, bar in enumerate(bars_out):
                bar.set_alpha(0.7 + 0.3 * mlp_out_mask[i])

            # Add circular markers at the top of each bar
            ax_out.plot(x_positions_out, mlp_out_mask, 'o',
                       color=COLORS['mlp_out'], markersize=7,
                       markeredgecolor='black', markeredgewidth=1.5,
                       zorder=5)

            ax_out.set_title(f'Layer {layer} - Output Projection',
                            fontsize=16, fontweight='bold', pad=10)
            ax_out.set_ylim(0, 1.05)
            ax_out.axhline(y=0.5, color='gray', linestyle='--',
                          alpha=0.4, linewidth=1.5)

            if layer == circuit.n_layers - 1:
                ax_out.set_xlabel('Singular Value Index',
                                fontsize=16, fontweight='bold')

            ax_out.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_out.set_axisbelow(True)

        fig_mlp.tight_layout(rect=[0, 0, 1, 0.99])

        if output_dir is not None:
            fig_mlp.savefig(os.path.join(output_dir, 'mlp_masks.pdf'),
                           bbox_inches='tight', dpi=dpi)
            fig_mlp.savefig(os.path.join(output_dir, 'mlp_masks.png'),
                           bbox_inches='tight', dpi=dpi)

    # Close figures if saving to avoid blocking
    if output_dir is not None:
        plt.close('all')
    else:
        plt.show()


def visualize_masked_singular_values(circuit, output_dir=None, mask_fn=None, dpi=300):
    """
    Visualize mask_value * singular_value for each component.

    This shows the effective strength of each singular value component after masking.

    Args:
        circuit: MaskedTransformerCircuit instance
        output_dir: Directory to save plots (optional)
        mask_fn: Mask function to apply (default: torch.sigmoid)
        dpi: Resolution for saved images (default: 300)
    """
    if mask_fn is None:
        mask_fn = torch.sigmoid

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Create subplots for QK masked singular values
    fig_qk, axes_qk = plt.subplots(
        circuit.n_layers, circuit.n_heads,
        figsize=(circuit.n_heads * 4, circuit.n_layers * 3),
        squeeze=False
    )
    # Remove title as per user request

    # Create subplots for OV masked singular values
    fig_ov, axes_ov = plt.subplots(
        circuit.n_layers, circuit.n_heads,
        figsize=(circuit.n_heads * 4, circuit.n_layers * 3),
        squeeze=False
    )
    # Remove title as per user request

    # Plot masked singular values for each head
    for layer in range(circuit.n_layers):
        for head in range(circuit.n_heads):
            head_key = f'differential_head_{layer}_{head}'

            # Get cache keys
            qk_cache_key = f"{head_key}_qk"
            ov_cache_key = f"{head_key}_ov"

            # Get singular values from SVD cache
            U_qk, S_qk, Vh_qk, _ = circuit.svd_cache[qk_cache_key]
            U_ov, S_ov, Vh_ov, _ = circuit.svd_cache[ov_cache_key]

            # Get mask values
            qk_mask = mask_fn(circuit.qk_masks[head_key]).detach().cpu().numpy()
            ov_mask = mask_fn(circuit.ov_masks[head_key]).detach().cpu().numpy()

            # Get singular values (truncated to d_head or d_head+1)
            S_qk_trunc = S_qk[:circuit.d_head].detach().cpu().numpy()
            S_ov_trunc = S_ov[:circuit.d_head+1].detach().cpu().numpy()

            # Compute masked singular values
            masked_S_qk = qk_mask * S_qk_trunc
            masked_S_ov = ov_mask * S_ov_trunc

            # Plot QK masked singular values with improved styling
            ax_qk = axes_qk[layer, head]
            indices_qk = np.arange(len(masked_S_qk))

            # Plot masked values as bars
            bars_qk = ax_qk.bar(indices_qk, masked_S_qk,
                               color=COLORS['qk'], alpha=0.85,
                               edgecolor='black', linewidth=1.2,
                               label='Masked σ')

            # Add circular markers at the top of each bar
            ax_qk.plot(indices_qk, masked_S_qk, 'o',
                      color=COLORS['qk'], markersize=6,
                      markeredgecolor='black', markeredgewidth=1.2,
                      zorder=5)

            # Plot original singular values as line
            ax_qk.plot(indices_qk, S_qk_trunc,
                      color=COLORS['original'], linestyle='--',
                      linewidth=3, alpha=0.8, marker='D',
                      markersize=5, markeredgecolor='black',
                      markeredgewidth=0.8, label='Original σ')

            ax_qk.set_title(f'Layer {layer}, Head {head}',
                           fontsize=16, fontweight='bold', pad=10)

            if head == 0:
                ax_qk.set_ylabel('Singular Value',
                               fontsize=16, fontweight='bold')
            if layer == circuit.n_layers - 1:
                ax_qk.set_xlabel('Index', fontsize=16, fontweight='bold')
            if layer == 0 and head == circuit.n_heads - 1:
                ax_qk.legend(fontsize=10, loc='upper right',
                            framealpha=0.95, edgecolor='black', fancybox=False)

            ax_qk.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_qk.set_axisbelow(True)

            # Plot OV masked singular values with improved styling
            ax_ov = axes_ov[layer, head]
            indices_ov = np.arange(len(masked_S_ov))

            # Plot masked values as bars
            bars_ov = ax_ov.bar(indices_ov, masked_S_ov,
                               color=COLORS['ov'], alpha=0.85,
                               edgecolor='black', linewidth=1.2,
                               label='Masked σ')

            # Add circular markers at the top of each bar
            ax_ov.plot(indices_ov, masked_S_ov, 'o',
                      color=COLORS['ov'], markersize=6,
                      markeredgecolor='black', markeredgewidth=1.2,
                      zorder=5)

            # Plot original singular values as line
            ax_ov.plot(indices_ov, S_ov_trunc,
                      color=COLORS['original'], linestyle='--',
                      linewidth=3, alpha=0.8, marker='D',
                      markersize=5, markeredgecolor='black',
                      markeredgewidth=0.8, label='Original σ')

            ax_ov.set_title(f'Layer {layer}, Head {head}',
                           fontsize=16, fontweight='bold', pad=10)

            if head == 0:
                ax_ov.set_ylabel('Singular Value',
                               fontsize=16, fontweight='bold')
            if layer == circuit.n_layers - 1:
                ax_ov.set_xlabel('Index', fontsize=16, fontweight='bold')
            if layer == 0 and head == circuit.n_heads - 1:
                ax_ov.legend(fontsize=10, loc='upper right',
                            framealpha=0.95, edgecolor='black', fancybox=False)

            ax_ov.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_ov.set_axisbelow(True)

    # Adjust layout
    fig_qk.tight_layout(rect=[0, 0, 1, 0.99])
    fig_ov.tight_layout(rect=[0, 0, 1, 0.99])

    # Save if output directory provided
    if output_dir is not None:
        fig_qk.savefig(os.path.join(output_dir, 'qk_masked_singular_values.pdf'),
                       bbox_inches='tight', dpi=dpi)
        fig_qk.savefig(os.path.join(output_dir, 'qk_masked_singular_values.png'),
                       bbox_inches='tight', dpi=dpi)
        fig_ov.savefig(os.path.join(output_dir, 'ov_masked_singular_values.pdf'),
                       bbox_inches='tight', dpi=dpi)
        fig_ov.savefig(os.path.join(output_dir, 'ov_masked_singular_values.png'),
                       bbox_inches='tight', dpi=dpi)

    # Visualize MLP masked singular values if enabled
    if circuit.mask_mlp:
        fig_mlp, axes_mlp = plt.subplots(
            circuit.n_layers, 2,
            figsize=(12, circuit.n_layers * 3.5),
            squeeze=False
        )
        # Remove title as per user request

        for layer in range(circuit.n_layers):
            mlp_key = f'mlp_{layer}'

            # Get cache keys
            mlp_in_cache_key = f"mlp_{layer}_in"
            mlp_out_cache_key = f"mlp_{layer}_out"

            # Get singular values from SVD cache
            U_in, S_in, Vh_in, _ = circuit.svd_cache[mlp_in_cache_key]
            U_out, S_out, Vh_out, _ = circuit.svd_cache[mlp_out_cache_key]

            # Get mask values
            mlp_in_mask = mask_fn(circuit.mlp_in_masks[mlp_key]).detach().cpu().numpy()
            mlp_out_mask = mask_fn(circuit.mlp_out_masks[mlp_key]).detach().cpu().numpy()

            # Get singular values
            S_in_np = S_in.detach().cpu().numpy()
            S_out_np = S_out.detach().cpu().numpy()

            # Compute masked singular values
            masked_S_in = mlp_in_mask * S_in_np[:len(mlp_in_mask)]
            masked_S_out = mlp_out_mask * S_out_np[:len(mlp_out_mask)]

            # Plot MLP_in masked singular values
            ax_in = axes_mlp[layer, 0]
            indices_in = np.arange(len(masked_S_in))

            bars_in = ax_in.bar(indices_in, masked_S_in,
                               color=COLORS['mlp_in'], alpha=0.85,
                               edgecolor='black', linewidth=1.2,
                               label='Masked σ')

            # Add circular markers at the top of each bar
            ax_in.plot(indices_in, masked_S_in, 'o',
                      color=COLORS['mlp_in'], markersize=6,
                      markeredgecolor='black', markeredgewidth=1.2,
                      zorder=5)

            ax_in.plot(indices_in, S_in_np[:len(mlp_in_mask)],
                      color=COLORS['original'], linestyle='--',
                      linewidth=3, alpha=0.8, marker='D',
                      markersize=5, markeredgecolor='black',
                      markeredgewidth=0.8, label='Original σ')

            ax_in.set_title(f'Layer {layer} - Input Projection',
                           fontsize=16, fontweight='bold', pad=10)
            ax_in.set_ylabel('Singular Value',
                            fontsize=16, fontweight='bold')
            ax_in.set_xlabel('Index', fontsize=16, fontweight='bold')

            if layer == 0:
                ax_in.legend(fontsize=10, loc='upper right',
                            framealpha=0.95, edgecolor='black', fancybox=False)

            ax_in.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_in.set_axisbelow(True)

            # Plot MLP_out masked singular values
            ax_out = axes_mlp[layer, 1]
            indices_out = np.arange(len(masked_S_out))

            bars_out = ax_out.bar(indices_out, masked_S_out,
                                 color=COLORS['mlp_out'], alpha=0.85,
                                 edgecolor='black', linewidth=1.2,
                                 label='Masked σ')

            # Add circular markers at the top of each bar
            ax_out.plot(indices_out, masked_S_out, 'o',
                       color=COLORS['mlp_out'], markersize=6,
                       markeredgecolor='black', markeredgewidth=1.2,
                       zorder=5)

            ax_out.plot(indices_out, S_out_np[:len(mlp_out_mask)],
                       color=COLORS['original'], linestyle='--',
                       linewidth=3, alpha=0.8, marker='D',
                       markersize=5, markeredgecolor='black',
                       markeredgewidth=0.8, label='Original σ')

            ax_out.set_title(f'Layer {layer} - Output Projection',
                            fontsize=16, fontweight='bold', pad=10)
            ax_out.set_ylabel('Singular Value',
                             fontsize=16, fontweight='bold')
            ax_out.set_xlabel('Index', fontsize=16, fontweight='bold')

            if layer == 0:
                ax_out.legend(fontsize=10, loc='upper right',
                             framealpha=0.95, edgecolor='black', fancybox=False)

            ax_out.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
            ax_out.set_axisbelow(True)

        fig_mlp.tight_layout(rect=[0, 0, 1, 0.99])

        if output_dir is not None:
            fig_mlp.savefig(os.path.join(output_dir, 'mlp_masked_singular_values.pdf'),
                           bbox_inches='tight', dpi=dpi)
            fig_mlp.savefig(os.path.join(output_dir, 'mlp_masked_singular_values.png'),
                           bbox_inches='tight', dpi=dpi)

    # Close figures if saving to avoid blocking
    if output_dir is not None:
        plt.close('all')
    else:
        plt.show()


def plot_training_history(history, output_dir=None, dpi=300):
    """
    Plot training history with improved aesthetics.

    Args:
        history: Dictionary containing training metrics
        output_dir: Directory to save plots (optional)
        dpi: Resolution for saved images (default: 300)
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Create figure with improved layout
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    # Remove title as per user request

    iterations = history['iteration']

    # Plot total loss with darker, bolder styling
    ax_loss = axes[0]
    ax_loss.plot(iterations, history['loss'],
                color=COLORS['loss'], linewidth=3.5, alpha=0.9,
                marker='o', markersize=3, markevery=max(1, len(iterations)//20))
    ax_loss.fill_between(iterations, history['loss'], alpha=0.3,
                         color=COLORS['loss'])
    ax_loss.set_ylabel('Total Loss', fontsize=16, fontweight='bold')
    ax_loss.set_title('Training Loss', fontsize=18, fontweight='bold', pad=12)
    ax_loss.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
    ax_loss.set_axisbelow(True)

    # Add min/max annotations with bold marker
    min_loss_idx = np.argmin(history['loss'])
    ax_loss.plot(iterations[min_loss_idx], history['loss'][min_loss_idx],
                marker='*', color='#c0392b', markersize=18,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'Min: {history["loss"][min_loss_idx]:.4f}', zorder=10)
    ax_loss.legend(fontsize=11, loc='upper right', framealpha=0.95,
                  edgecolor='black', fancybox=False)

    # Plot KL divergence with darker, bolder styling
    ax_kl = axes[1]
    ax_kl.plot(iterations, history['kl_div'],
              color=COLORS['kl'], linewidth=3.5, alpha=0.9,
              marker='s', markersize=3, markevery=max(1, len(iterations)//20))
    ax_kl.fill_between(iterations, history['kl_div'], alpha=0.3,
                       color=COLORS['kl'])
    ax_kl.set_ylabel('KL Divergence', fontsize=16, fontweight='bold')
    ax_kl.set_title('KL Divergence from Original Model', fontsize=18,
                   fontweight='bold', pad=12)
    ax_kl.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
    ax_kl.set_axisbelow(True)

    # Add final value annotation with bolder line
    final_kl = history['kl_div'][-1]
    ax_kl.axhline(y=final_kl, color='#5d6d7e', linestyle='--',
                  alpha=0.6, linewidth=2)
    ax_kl.text(iterations[-1], final_kl, f' Final: {final_kl:.4f}',
              verticalalignment='bottom', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='black', alpha=0.8))

    # Plot L1 penalty with darker, bolder styling
    ax_l1 = axes[2]
    ax_l1.plot(iterations, history['l1_penalty'],
              color=COLORS['l1'], linewidth=3.5, alpha=0.9,
              marker='^', markersize=3, markevery=max(1, len(iterations)//20))
    ax_l1.fill_between(iterations, history['l1_penalty'], alpha=0.3,
                       color=COLORS['l1'])
    ax_l1.set_ylabel('L1 Penalty', fontsize=16, fontweight='bold')
    ax_l1.set_xlabel('Iteration', fontsize=16, fontweight='bold')
    ax_l1.set_title('Mask Sparsity (L1 Penalty)', fontsize=18,
                   fontweight='bold', pad=12)
    ax_l1.grid(True, alpha=0.35, linestyle='-', linewidth=0.8)
    ax_l1.set_axisbelow(True)

    # Add final value annotation with bolder line
    final_l1 = history['l1_penalty'][-1]
    ax_l1.axhline(y=final_l1, color='#5d6d7e', linestyle='--',
                  alpha=0.6, linewidth=2)
    ax_l1.text(iterations[-1], final_l1, f' Final: {final_l1:.4f}',
              verticalalignment='top', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='black', alpha=0.8))

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save if output directory provided
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'training_history.pdf'),
                   bbox_inches='tight', dpi=dpi)
        plt.savefig(os.path.join(output_dir, 'training_history.png'),
                   bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        plt.show()
