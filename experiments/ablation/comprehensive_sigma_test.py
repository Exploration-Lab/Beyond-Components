#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Sigma Amplification Test
Test sigma amplification on all 4 experiments systematically
"""

import os
import sys
import torch
import numpy as np
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from experiments.ablation.intervention import (
    load_model_and_circuit,
    compute_baseline_metrics,
    apply_range_swap_intervention,
    DIRECTION_RANGES
)
from src.data import data_loader as local_data_loader

def main():
    # Use checkpoints from repository
    run_dir = os.path.join(project_root, "checkpoints", "gp")
    model_path = os.path.join(run_dir, "model_final.pt")
    config_path = os.path.join(run_dir, "run_summary.json")
    output_dir = os.path.join(project_root, "results", "comprehensive_sigma_test")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        summary = json.load(f)
    config = summary['config']

    print("=" * 120)
    print("COMPREHENSIVE SIGMA AMPLIFICATION TEST")
    print("=" * 120)
    print("\nTesting sigma amplification on ALL 4 experiments")
    print("Sigma values to test: [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]")
    print()

    # Load model and circuit
    print("[1/3] Loading model and circuit...")
    model, circuit, device = load_model_and_circuit(model_path, config)

    # Load test data
    print("[2/3] Loading test data...")
    test_loader = local_data_loader.load_gp_dataset(
        batch_size=32,
        train=False,
        validation=False,
        shuffle=False
    )

    # Compute baseline
    print("[3/3] Computing baseline metrics...")
    baseline_metrics = compute_baseline_metrics(model, circuit, test_loader, device, max_batches=20)

    print("\nBaseline Results:")
    print(f"  HE:  Accuracy: {baseline_metrics['he']['accuracy']:.2f}%, Logit Diff: {baseline_metrics['he']['mean_logit_diff']:.4f}")
    print(f"  SHE: Accuracy: {baseline_metrics['she']['accuracy']:.2f}%, Logit Diff: {baseline_metrics['she']['mean_logit_diff']:.4f}")

    # Define experiments
    masculine_dirs = {k: v for k, v in DIRECTION_RANGES.items() if v['type'] == 'masculine'}
    feminine_dirs = {k: v for k, v in DIRECTION_RANGES.items() if v['type'] == 'feminine'}
    all_gender_dirs = {**masculine_dirs, **feminine_dirs}

    experiments = [
        {
            'name': 'E.1: Swap ALL dirs',
            'target_gender': 'he',
            'directions': all_gender_dirs,
            'baseline': baseline_metrics['he']
        },
        {
            'name': 'E.2: Swap ALL dirs',
            'target_gender': 'she',
            'directions': all_gender_dirs,
            'baseline': baseline_metrics['she']
        },
        {
            'name': 'E.3: Swap Masc. only',
            'target_gender': 'he',
            'directions': masculine_dirs,
            'baseline': baseline_metrics['he']
        },
        {
            'name': 'E.4: Swap Fem. only',
            'target_gender': 'she',
            'directions': feminine_dirs,
            'baseline': baseline_metrics['she']
        }
    ]

    # Sigma values to test
    sigma_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]

    # Run experiments
    all_results = {}

    print("\n" + "=" * 120)
    print("RUNNING SIGMA AMPLIFICATION EXPERIMENTS")
    print("=" * 120)

    for exp in experiments:
        exp_name = exp['name']
        print(f"\n{exp_name} (target: {exp['target_gender']})")
        print("-" * 120)

        exp_results = {}

        for sigma in sigma_values:
            print(f"  σ × {sigma:5.1f}...", end=" ", flush=True)

            result = apply_range_swap_intervention(
                model, circuit, test_loader, device,
                target_gender=exp['target_gender'],
                directions_to_swap=exp['directions'],
                max_batches=20,
                sigma_amplification=sigma
            )

            exp_results[f'sigma_{sigma}'] = result

            # Calculate change from baseline
            baseline_logit_diff = exp['baseline']['mean_logit_diff']
            interv_logit_diff = result['mean_logit_diff']
            delta_logit_diff = interv_logit_diff - baseline_logit_diff

            print(f"∆Logit: {interv_logit_diff:+7.2f} (Δ from baseline: {delta_logit_diff:+7.2f}), "
                  f"Flip: {result['flip_to_she_pct'] + result['flip_to_he_pct']:.1f}%")

        all_results[exp_name] = {
            'baseline': exp['baseline'],
            'sigma_experiments': exp_results
        }

    # Save detailed results
    results_path = os.path.join(output_dir, "comprehensive_sigma_results.json")
    with open(results_path, 'w') as f:
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\n\nDetailed results saved to: {results_path}")

    # Create comprehensive table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE RESULTS TABLE")
    print("=" * 120)
    print()

    # Header
    print(f"{'Experiment':<25} {'σ':<6} {'Prompt':<8} {'n':<5} "
          f"{'Baseline ∆Logit':<20} {'Interv. ∆Logit':<20} {'∆(∆Logit)':<12} "
          f"{'Acc%':<8} {'Flip→she%':<12} {'Flip→he%':<12}")
    print("-" * 160)

    for exp_name, data in all_results.items():
        baseline = data['baseline']
        sigma_exps = data['sigma_experiments']

        # Determine prompt context
        if 'he' in exp_name.lower() or data['baseline']['mean_logit_diff'] == baseline_metrics['he']['mean_logit_diff']:
            prompt = '"he"'
        else:
            prompt = '"she"'

        n = baseline['n_samples']
        baseline_logit = baseline['mean_logit_diff']
        baseline_std = baseline['std_logit_diff']

        # Print each sigma value
        for sigma_key in sorted(sigma_exps.keys(), key=lambda x: float(x.split('_')[1])):
            sigma_val = float(sigma_key.split('_')[1])
            result = sigma_exps[sigma_key]

            interv_logit = result['mean_logit_diff']
            interv_std = result['std_logit_diff']
            delta_logit = interv_logit - baseline_logit
            acc = result['accuracy']
            flip_she = result['flip_to_she_pct']
            flip_he = result['flip_to_he_pct']

            print(f"{exp_name:<25} {sigma_val:<6.1f} {prompt:<8} {n:<5} "
                  f"{baseline_logit:+6.2f} ± {baseline_std:5.2f}     "
                  f"{interv_logit:+6.2f} ± {interv_std:5.2f}     "
                  f"{delta_logit:+7.2f}      "
                  f"{acc:6.1f}%   {flip_she:6.1f}%       {flip_he:6.1f}%")

        print()

    print("=" * 120)

if __name__ == "__main__":
    main()
