#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Range-Based Ablation Study

Instead of arbitrary multipliers, we swap activations to their empirically
observed ranges for the opposite gender.

For example:
- L9.H7.SV1 (masculine direction):
  - On "he" prompts: V'·u ≈ +0.115
  - On "she" prompts: V'·u ≈ -0.453
  - Intervention: Swap these values!

This is more interpretable than arbitrary multipliers.
"""

import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer
from tqdm import tqdm
import json

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import data_loader as local_data_loader
from src.utils.utils import get_data_column_names, get_indirect_objects_and_subjects
from src.models.masked_transformer_circuit import MaskedTransformerCircuit


# Empirically measured activation ranges from deep_analysis_report.txt
DIRECTION_RANGES = {
    # Masculine directions
    'L9.H7.SV1': {
        'layer': 9, 'head': 7, 'sv_idx': 1,
        'type': 'masculine',
        'he_mean': +0.115, 'he_std': 0.180,
        'she_mean': -0.453, 'she_std': 0.291,
    },
    'L11.H8.SV6': {
        'layer': 11, 'head': 8, 'sv_idx': 6,
        'type': 'masculine',
        'he_mean': +0.203, 'he_std': 0.140,
        'she_mean': -0.121, 'she_std': 0.184,
    },

    # Feminine directions
    'L10.H9.SV0': {
        'layer': 10, 'head': 9, 'sv_idx': 0,
        'type': 'feminine',
        'he_mean': -0.273, 'he_std': 0.259,
        'she_mean': +0.652, 'she_std': 0.420,
    },
    'L11.H8.SV9': {
        'layer': 11, 'head': 8, 'sv_idx': 9,
        'type': 'feminine',
        'he_mean': -0.159, 'he_std': 0.171,
        'she_mean': +0.134, 'she_std': 0.199,
    },

    # Plural direction (for comparison)
    'L9.H7.SV0': {
        'layer': 9, 'head': 7, 'sv_idx': 0,
        'type': 'plural',
        'he_mean': -0.334, 'he_std': 0.137,
        'she_mean': -0.174, 'she_std': 0.162,
    },
}


def load_model_and_circuit(model_path, config):
    """Load the model and circuit with trained masks"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HookedTransformer.from_pretrained(
            config['model']['name'],
            cache_dir=config['model']['pretrained_cache_dir']
    )
    model = model.to(device)

    circuit = MaskedTransformerCircuit(
        model=model,
        device=device,
        cache_svd=True,
        mask_init_value=config['masking']['mask_init_value']
    )

    checkpoint = torch.load(model_path, map_location=device, 
                            weights_only=False)
    circuit.qk_masks = checkpoint['qk_masks']
    circuit.ov_masks = checkpoint['ov_masks']
    circuit.mlp_in_masks = checkpoint['mlp_in_masks']
    circuit.mlp_out_masks = checkpoint['mlp_out_masks']

    return model, circuit, device


def compute_baseline_metrics(model, circuit, data_loader, device, max_batches=20):
    """Compute baseline without intervention"""
    he_token = model.tokenizer.encode(' he', add_special_tokens=False)[0]
    she_token = model.tokenizer.encode(' she', add_special_tokens=False)[0]
    they_token = model.tokenizer.encode(' they', add_special_tokens=False)[0]

    results = {
        'he': {'correct': [], 'logit_diff': [], 'they_logit': []},
        'she': {'correct': [], 'logit_diff': [], 'they_logit': []}
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Computing baseline")):
            if max_batches and batch_idx >= max_batches:
                break

            clean_column_name, _ = get_data_column_names('gp')
            indirect_objects_column_name, _ = get_indirect_objects_and_subjects('gp')

            input_ids_clean = model.tokenizer(
                batch[clean_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            indirect_objs = batch[indirect_objects_column_name]
            if isinstance(indirect_objs, torch.Tensor):
                indirect_objs = indirect_objs.tolist()
            batch[indirect_objects_column_name] = [' ' + str(obj) for obj in indirect_objs]
            indirect_objects = model.tokenizer(
                batch[indirect_objects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            clean_lengths = (input_ids_clean != model.tokenizer.pad_token_id).sum(dim=1)
            clean_last_idx = clean_lengths - 1
            attention_mask_clean = torch.arange(input_ids_clean.size(1), device=device)[None, :] < clean_lengths[:, None]

            logits = model(input_ids_clean, attention_mask=attention_mask_clean)
            batch_size = input_ids_clean.size(0)
            batch_indices = torch.arange(batch_size, device=device)

            last_logits = logits[batch_indices, clean_last_idx, :]
            predictions = last_logits.argmax(dim=-1)
            labels = indirect_objects[:, 0]

            he_logits = last_logits[:, he_token]
            she_logits = last_logits[:, she_token]
            they_logits = last_logits[:, they_token]

            he_mask = labels == he_token
            she_mask = labels == she_token

            if he_mask.sum() > 0:
                results['he']['correct'].extend((predictions[he_mask] == labels[he_mask]).cpu().tolist())
                logit_diff = (he_logits[he_mask] - she_logits[he_mask]).cpu().tolist()
                results['he']['logit_diff'].extend(logit_diff)
                results['he']['they_logit'].extend(they_logits[he_mask].cpu().tolist())

            if she_mask.sum() > 0:
                results['she']['correct'].extend((predictions[she_mask] == labels[she_mask]).cpu().tolist())
                logit_diff = (she_logits[she_mask] - he_logits[she_mask]).cpu().tolist()
                results['she']['logit_diff'].extend(logit_diff)
                results['she']['they_logit'].extend(they_logits[she_mask].cpu().tolist())

    summary = {}
    for gender in ['he', 'she']:
        summary[gender] = {
            'accuracy': np.mean(results[gender]['correct']) * 100,
            'mean_logit_diff': np.mean(results[gender]['logit_diff']),
            'std_logit_diff': np.std(results[gender]['logit_diff']),
            'mean_they_logit': np.mean(results[gender]['they_logit']),
            'std_they_logit': np.std(results[gender]['they_logit']),
            'n_samples': len(results[gender]['correct'])
        }

    return summary


def apply_range_swap_intervention(model, circuit, data_loader, device,
                                  target_gender, directions_to_swap,
                                  max_batches=20, sigma_amplification=1.0):
    """
    Apply range-swap intervention

    For each direction, replace V'·u_i with the value it has for opposite gender

    Args:
        sigma_amplification: Factor to multiply sigma values by (default 1.0 = no amplification)
    """
    he_token = model.tokenizer.encode(' he', add_special_tokens=False)[0]
    she_token = model.tokenizer.encode(' she', add_special_tokens=False)[0]
    they_token = model.tokenizer.encode(' they', add_special_tokens=False)[0]

    results = {
        'correct': [],
        'logit_diff': [],
        'they_logit': [],
        'baseline_predictions': [],  # Track original predictions
        'intervened_predictions': []  # Track post-intervention predictions
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Range swap: {target_gender}")):
            if max_batches and batch_idx >= max_batches:
                break

            clean_column_name, _ = get_data_column_names('gp')
            indirect_objects_column_name, _ = get_indirect_objects_and_subjects('gp')

            input_ids_clean = model.tokenizer(
                batch[clean_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            indirect_objs = batch[indirect_objects_column_name]
            if isinstance(indirect_objs, torch.Tensor):
                indirect_objs = indirect_objs.tolist()
            batch[indirect_objects_column_name] = [' ' + str(obj) for obj in indirect_objs]
            indirect_objects = model.tokenizer(
                batch[indirect_objects_column_name],
                return_tensors='pt',
                padding=True
            )['input_ids'].to(device)

            clean_lengths = (input_ids_clean != model.tokenizer.pad_token_id).sum(dim=1)
            clean_last_idx = clean_lengths - 1
            attention_mask_clean = torch.arange(input_ids_clean.size(1), device=device)[None, :] < clean_lengths[:, None]

            _, cache = model.run_with_cache(input_ids_clean, attention_mask=attention_mask_clean)

            batch_size = input_ids_clean.size(0)
            batch_indices = torch.arange(batch_size, device=device)

            labels = indirect_objects[:, 0]
            if target_gender == 'he':
                target_mask = labels == he_token
                correct_token = he_token
                incorrect_token = she_token
            elif target_gender == 'she':
                target_mask = labels == she_token
                correct_token = she_token
                incorrect_token = he_token

            if target_mask.sum() == 0:
                continue

            # First, get BASELINE predictions (without intervention)
            baseline_resid_unnorm = cache['blocks.11.hook_resid_post']
            baseline_last_token_resid = baseline_resid_unnorm[batch_indices, clean_last_idx, :]
            baseline_resid_norm = model.ln_final(baseline_last_token_resid)
            baseline_logits = torch.matmul(baseline_resid_norm, model.W_U)
            baseline_predictions = baseline_logits.argmax(dim=-1)

            # Store baseline predictions for target samples
            results['baseline_predictions'].extend(baseline_predictions[target_mask].cpu().tolist())

            # Compute total intervention
            total_intervention = torch.zeros(batch_size, model.cfg.d_model, device=device)

            for dir_name, dir_info in directions_to_swap.items():
                layer = dir_info['layer']
                head = dir_info['head']
                sv_idx = dir_info['sv_idx']

                # Get attention pattern and input
                attn_pattern = cache[f'blocks.{layer}.attn.hook_pattern']
                attn_in = cache[f'blocks.{layer}.ln1.hook_normalized']
                attn_weights_last = attn_pattern[batch_indices, :, clean_last_idx, :]

                attn_w = attn_weights_last[:, head, :]
                context_standard = torch.matmul(attn_w.unsqueeze(1), attn_in).squeeze(1)
                ones = torch.ones(batch_size, 1, device=device)
                context = torch.cat([ones, context_standard], dim=1)  # V'

                # Get SVD components
                head_key = f'differential_head_{layer}_{head}'
                ov_cache_key = f"{head_key}_ov"

                if ov_cache_key not in circuit.svd_cache:
                    continue

                U_ov, S_ov, Vt_ov, _ = circuit.svd_cache[ov_cache_key]
                U_ov = U_ov.to(device)
                V_ov = Vt_ov.T.to(device)

                u_i = U_ov[:, sv_idx:sv_idx+1]  # [d_aug, 1]
                v_i = V_ov[:, sv_idx:sv_idx+1]  # [d_model, 1]
                sigma_i = S_ov[sv_idx]

                # Current activation: V' · u_i
                current_activation = torch.matmul(context, u_i)  # [batch, 1]

                # Target activation (from opposite gender's range)
                if target_gender == 'he':
                    # We're on "he" prompts, set to "she" range
                    target_activation_value = dir_info['she_mean']
                else:
                    # We're on "she" prompts, set to "he" range
                    target_activation_value = dir_info['he_mean']

                target_activation = torch.full_like(current_activation, target_activation_value)

                # Compute the change in activation
                delta_activation = target_activation - current_activation

                # Convert activation change to output change
                # Apply sigma amplification if specified
                amplified_sigma = sigma_i * sigma_amplification
                intervention = delta_activation * amplified_sigma * v_i.T  # [batch, d_model]
                total_intervention += intervention

            # Apply intervention
            # CORRECTED: Get the UNnormalized residual stream before ln_final
            final_resid_unnorm = cache['blocks.11.hook_resid_post']
            intervened_resid_unnorm = final_resid_unnorm.clone()
            intervened_resid_unnorm[batch_indices, clean_last_idx, :] += total_intervention

            # Get last token positions
            last_token_resid = intervened_resid_unnorm[batch_indices, clean_last_idx, :]

            # Apply layer normalization to the modified residual
            intervened_resid_norm = model.ln_final(last_token_resid)

            # Get new logits
            last_logits = torch.matmul(intervened_resid_norm, model.W_U)

            predictions = last_logits.argmax(dim=-1)
            he_logits = last_logits[:, he_token]
            she_logits = last_logits[:, she_token]
            they_logits = last_logits[:, they_token]

            # Store intervened predictions for target samples
            results['intervened_predictions'].extend(predictions[target_mask].cpu().tolist())

            results['correct'].extend((predictions[target_mask] == labels[target_mask]).cpu().tolist())

            if target_gender == 'he':
                logit_diff = (he_logits[target_mask] - she_logits[target_mask]).cpu().tolist()
            else:
                logit_diff = (she_logits[target_mask] - he_logits[target_mask]).cpu().tolist()

            results['logit_diff'].extend(logit_diff)
            results['they_logit'].extend(they_logits[target_mask].cpu().tolist())

    # Calculate prediction flip statistics
    he_token = model.tokenizer.encode(' he', add_special_tokens=False)[0]
    she_token = model.tokenizer.encode(' she', add_special_tokens=False)[0]

    baseline_preds = np.array(results['baseline_predictions'])
    intervened_preds = np.array(results['intervened_predictions'])

    # Count predictions
    n_baseline_he = (baseline_preds == he_token).sum()
    n_baseline_she = (baseline_preds == she_token).sum()
    n_intervened_he = (intervened_preds == he_token).sum()
    n_intervened_she = (intervened_preds == she_token).sum()

    # Count flips
    flipped_to_he = ((baseline_preds == she_token) & (intervened_preds == he_token)).sum()
    flipped_to_she = ((baseline_preds == he_token) & (intervened_preds == she_token)).sum()

    # Calculate flip percentages (based on baseline predictions of target gender)
    # flip_to_he_pct = % of baseline 'she' predictions that flipped to 'he'
    # flip_to_she_pct = % of baseline 'he' predictions that flipped to 'she'
    flip_to_he_pct = (flipped_to_he / n_baseline_she * 100) if n_baseline_she > 0 else 0
    flip_to_she_pct = (flipped_to_she / n_baseline_he * 100) if n_baseline_he > 0 else 0

    summary = {
        'accuracy': np.mean(results['correct']) * 100,
        'mean_logit_diff': np.mean(results['logit_diff']),
        'std_logit_diff': np.std(results['logit_diff']),
        'mean_they_logit': np.mean(results['they_logit']),
        'std_they_logit': np.std(results['they_logit']),
        'n_samples': len(results['correct']),
        # Prediction counts
        'n_baseline_he': int(n_baseline_he),
        'n_baseline_she': int(n_baseline_she),
        'n_intervened_he': int(n_intervened_he),
        'n_intervened_she': int(n_intervened_she),
        # Flip statistics
        'flipped_to_he': int(flipped_to_he),
        'flipped_to_she': int(flipped_to_she),
        'flip_to_he_pct': float(flip_to_he_pct),
        'flip_to_she_pct': float(flip_to_she_pct)
    }

    return summary


def main():
    run_dir = "/home/areeb/CircDisk/svd_logs/gp_20251017_123045"
    model_path = os.path.join(run_dir, "model_final.pt")
    config_path = os.path.join(run_dir, "run_summary.json")
    output_dir = os.path.join(run_dir, "smart_ablation")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        summary = json.load(f)
    config = summary['config']

    print("=" * 100)
    print("SMART RANGE-BASED ABLATION STUDY")
    print("=" * 100)
    print("\nStrategy: Swap V'·u_i to empirically observed ranges for opposite gender")
    print()
    print("Directions and their ranges:")
    for dir_name, dir_info in DIRECTION_RANGES.items():
        print(f"  {dir_name} ({dir_info['type']}):")
        print(f"    he:  {dir_info['he_mean']:+.3f} ± {dir_info['he_std']:.3f}")
        print(f"    she: {dir_info['she_mean']:+.3f} ± {dir_info['she_std']:.3f}")
    print()

    print("[1/4] Loading model and circuit...")
    model, circuit, device = load_model_and_circuit(model_path, config)

    print("\n[2/4] Loading test data...")
    test_loader = local_data_loader.load_gp_dataset(
        batch_size=32,
        train=False,
        validation=False,
        shuffle=False
    )

    print("\n[3/4] Computing baseline metrics...")
    baseline_metrics = compute_baseline_metrics(model, circuit, test_loader, device, max_batches=20)

    print("\nBaseline Results:")
    for gender in ['he', 'she']:
        print(f"  {gender.upper()}:")
        print(f"    Accuracy: {baseline_metrics[gender]['accuracy']:.2f}%")
        print(f"    Logit Diff: {baseline_metrics[gender]['mean_logit_diff']:.4f}")
        print(f"    They Logit: {baseline_metrics[gender]['mean_they_logit']:.4f}")

    print("\n[4/4] Running range-swap interventions...")

    # Experiment 1: Masculine prompts - swap all directions
    print("\n  Exp 1: Masculine prompts → Swap all gender directions")
    masculine_dirs = {k: v for k, v in DIRECTION_RANGES.items() if v['type'] == 'masculine'}
    feminine_dirs = {k: v for k, v in DIRECTION_RANGES.items() if v['type'] == 'feminine'}
    all_gender_dirs = {**masculine_dirs, **feminine_dirs}

    exp1_results = apply_range_swap_intervention(
        model, circuit, test_loader, device,
        target_gender='he',
        directions_to_swap=all_gender_dirs,
        max_batches=20
    )

    # Experiment 2: Feminine prompts - swap all directions
    print("\n  Exp 2: Feminine prompts → Swap all gender directions")
    exp2_results = apply_range_swap_intervention(
        model, circuit, test_loader, device,
        target_gender='she',
        directions_to_swap=all_gender_dirs,
        max_batches=20
    )

    # Experiment 3: Masculine prompts - swap only masculine
    print("\n  Exp 3: Masculine prompts → Swap only masculine directions")
    exp3_results = apply_range_swap_intervention(
        model, circuit, test_loader, device,
        target_gender='he',
        directions_to_swap=masculine_dirs,
        max_batches=20
    )

    # Experiment 4: Feminine prompts - swap only feminine
    print("\n  Exp 4: Feminine prompts → Swap only feminine directions")
    exp4_results = apply_range_swap_intervention(
        model, circuit, test_loader, device,
        target_gender='she',
        directions_to_swap=feminine_dirs,
        max_batches=20
    )

    # SIGMA AMPLIFICATION EXPERIMENTS
    print("\n" + "=" * 100)
    print("SIGMA AMPLIFICATION EXPERIMENTS")
    print("=" * 100)
    print("\nTesting if amplifying sigma increases logit difference...")
    print()

    sigma_scales = [1.0, 1.5, 2.0, 3.0, 5.0]
    sigma_exp_results = {}

    for sigma_scale in sigma_scales:
        print(f"\n  Testing σ × {sigma_scale}:")

        # Test on masculine prompts with all directions
        print(f"    - Masc prompts, swap all (σ × {sigma_scale})")
        result = apply_range_swap_intervention(
            model, circuit, test_loader, device,
            target_gender='he',
            directions_to_swap=all_gender_dirs,
            max_batches=20,
            sigma_amplification=sigma_scale
        )
        sigma_exp_results[f'masc_all_sigma_{sigma_scale}'] = result
        print(f"      Logit Diff: {result['mean_logit_diff']:.4f} (baseline: {baseline_metrics['he']['mean_logit_diff']:.4f})")

    # Save results
    results = {
        'baseline': baseline_metrics,
        'exp1_masc_swap_all': exp1_results,
        'exp2_fem_swap_all': exp2_results,
        'exp3_masc_swap_masc_only': exp3_results,
        'exp4_fem_swap_fem_only': exp4_results,
        'sigma_amplification_experiments': sigma_exp_results,
        'direction_ranges': DIRECTION_RANGES
    }

    results_path = os.path.join(output_dir, "smart_ablation_results.json")
    with open(results_path, 'w') as f:
        # Convert to serializable format
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj

        json.dump(make_serializable(results), f, indent=2)

    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Experiment':<40} {'Accuracy':<12} {'Logit Diff':<12} {'They Logit':<12}")
    print("-" * 80)
    print(f"{'Baseline (he)':<40} {baseline_metrics['he']['accuracy']:<12.2f} {baseline_metrics['he']['mean_logit_diff']:<12.4f} {baseline_metrics['he']['mean_they_logit']:<12.4f}")
    print(f"{'Baseline (she)':<40} {baseline_metrics['she']['accuracy']:<12.2f} {baseline_metrics['she']['mean_logit_diff']:<12.4f} {baseline_metrics['she']['mean_they_logit']:<12.4f}")
    print("-" * 80)
    print(f"{'Exp1: Masc → Swap all':<40} {exp1_results['accuracy']:<12.2f} {exp1_results['mean_logit_diff']:<12.4f} {exp1_results['mean_they_logit']:<12.4f}")
    print(f"{'Exp2: Fem → Swap all':<40} {exp2_results['accuracy']:<12.2f} {exp2_results['mean_logit_diff']:<12.4f} {exp2_results['mean_they_logit']:<12.4f}")
    print(f"{'Exp3: Masc → Swap masc only':<40} {exp3_results['accuracy']:<12.2f} {exp3_results['mean_logit_diff']:<12.4f} {exp3_results['mean_they_logit']:<12.4f}")
    print(f"{'Exp4: Fem → Swap fem only':<40} {exp4_results['accuracy']:<12.2f} {exp4_results['mean_logit_diff']:<12.4f} {exp4_results['mean_they_logit']:<12.4f}")
    print()
    print(f"Results saved to: {results_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
