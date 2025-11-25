#!/usr/bin/env python3
"""
Generate comprehensive sigma amplification table from JSON results.
Ensures exact reproducibility by reading directly from comprehensive_sigma_results.json
"""

import json
import sys

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_table(results):
    """Generate formatted table from results."""

    print("=" * 160)
    print("COMPREHENSIVE SIGMA AMPLIFICATION RESULTS")
    print("All values directly from: comprehensive_sigma_results.json")
    print("=" * 160)
    print()

    # Header
    print(f"{'Experiment':<25} {'σ':<6} {'Prompt':<8} {'n':<5} "
          f"{'Baseline ∆Logit':<20} {'Interv. ∆Logit':<20} {'∆(∆Logit)':<12} "
          f"{'Acc%':<8} {'Flip→she%':<12} {'Flip→he%':<12}")
    print("-" * 160)

    # Process each experiment
    for exp_name in ['E.1: Swap ALL dirs', 'E.2: Swap ALL dirs',
                     'E.3: Swap Masc. only', 'E.4: Swap Fem. only']:

        data = results[exp_name]
        baseline = data['baseline']
        sigma_exps = data['sigma_experiments']

        # Determine prompt type based on experiment name
        if 'E.1' in exp_name or 'E.3' in exp_name:
            prompt = '"he"'
        else:
            prompt = '"she"'

        n = baseline['n_samples']
        baseline_logit = baseline['mean_logit_diff']
        baseline_std = baseline['std_logit_diff']

        # Process each sigma value
        sigma_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
        for sigma_val in sigma_values:
            result = sigma_exps[f'sigma_{sigma_val}']

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

        print()  # Blank line between experiments

    print("=" * 160)
    print()
    print("Column Definitions:")
    print("  σ: Sigma amplification factor")
    print("  Prompt: Prompt context (prompts expecting 'he' or 'she')")
    print("  n: Number of test samples")
    print("  Baseline ∆Logit: logit(target) - logit(other) before intervention (mean ± std)")
    print("  Interv. ∆Logit: logit(target) - logit(other) after intervention (mean ± std)")
    print("  ∆(∆Logit): Change in logit difference (Interv. - Baseline)")
    print("  Acc%: Accuracy after intervention")
    print("  Flip→she%: % of baseline 'he' predictions that flipped to 'she'")
    print("  Flip→he%: % of baseline 'she' predictions that flipped to 'he'")
    print()
    print("Flip Percentage Formulas:")
    print("  Flip→she% = (flipped_to_she / n_baseline_he) × 100")
    print("  Flip→he% = (flipped_to_he / n_baseline_she) × 100")
    print()

def verify_flip_calculations(results):
    """Verify that flip percentages in JSON are computed correctly."""
    print("=" * 160)
    print("VERIFICATION: Flip Percentage Calculations")
    print("=" * 160)
    print()

    all_correct = True

    for exp_name, data in results.items():
        sigma_exps = data['sigma_experiments']

        for sigma_key, result in sigma_exps.items():
            n_baseline_he = result['n_baseline_he']
            n_baseline_she = result['n_baseline_she']
            flipped_to_he = result['flipped_to_he']
            flipped_to_she = result['flipped_to_she']
            flip_to_he_pct_json = result['flip_to_he_pct']
            flip_to_she_pct_json = result['flip_to_she_pct']

            # Calculate expected values
            flip_to_he_pct_expected = (flipped_to_he / n_baseline_she * 100) if n_baseline_she > 0 else 0
            flip_to_she_pct_expected = (flipped_to_she / n_baseline_he * 100) if n_baseline_he > 0 else 0

            # Check if they match (within floating point tolerance)
            he_match = abs(flip_to_he_pct_json - flip_to_he_pct_expected) < 0.01
            she_match = abs(flip_to_she_pct_json - flip_to_she_pct_expected) < 0.01

            if not (he_match and she_match):
                all_correct = False
                print(f"❌ {exp_name}, {sigma_key}:")
                if not she_match:
                    print(f"   Flip→she%: JSON={flip_to_she_pct_json:.2f}, Expected={flip_to_she_pct_expected:.2f}")
                if not he_match:
                    print(f"   Flip→he%: JSON={flip_to_he_pct_json:.2f}, Expected={flip_to_he_pct_expected:.2f}")

    if all_correct:
        print("✓ All flip percentages are calculated correctly!")
        print("  Formula verified: flip_to_X_pct = (flipped_to_X / n_baseline_opposite) × 100")

    print()
    print("=" * 160)
    print()

if __name__ == "__main__":
    # Default path
    json_path = "/home/areeb/CircDisk/svd_logs/gp_20251017_123045/comprehensive_sigma_test/comprehensive_sigma_results.json"

    # Allow custom path from command line
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    print(f"Loading results from: {json_path}\n")

    # Load results
    results = load_results(json_path)

    # Verify calculations
    verify_flip_calculations(results)

    # Generate table
    generate_table(results)
