import json
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load run summaries with test evaluation data
print("Loading test evaluation data from checkpoints:")
print("- IOI: checkpoints/ioi")
print("- GT:  checkpoints/gt")
print("- GP:  checkpoints/gp")
print()

with open(os.path.join(project_root, 'checkpoints/ioi/run_summary.json'), 'r') as f:
    ioi_summary = json.load(f)

with open(os.path.join(project_root, 'checkpoints/gt/run_summary.json'), 'r') as f:
    gt_summary = json.load(f)

with open(os.path.join(project_root, 'checkpoints/gp/run_summary_metrics.json'), 'r') as f:
    gp_summary = json.load(f)

print("=" * 140)
print("TEST EVALUATION METRICS: Sparsity vs KL Divergence vs Accuracy vs Exact Match")
print("All metrics are from test data evaluation (mean ± std)")
print("=" * 140)
print()

# Main comparison table with both sparsity types
print("## Final Test Metrics (mean ± std)")
print()
print("| Dataset | Relative Sparsity (%) | Full Sparsity (%) | KL Divergence | Accuracy | Exact Match |")
print("|---------|----------------------|-------------------|---------------|----------|-------------|")

# IOI
if 'test_evaluation' in ioi_summary:
    te = ioi_summary['test_evaluation']
    print(f"| IOI     | {te['relative_sparsity']:.2f} | "
          f"{te['full_sparsity']:.2f} | "
          f"{te['test_kl']:.4f} ± {te['test_kl_std']:.4f} | "
          f"{te['test_masked_acc']:.4f} ± {te['test_masked_acc_std']:.4f} | "
          f"{te['test_exact_match']:.4f} ± {te['test_exact_match_std']:.4f} |")

# GT
if 'test_evaluation' in gt_summary:
    te = gt_summary['test_evaluation']
    print(f"| GT      | {te['relative_sparsity']:.2f} | "
          f"{te['full_sparsity']:.2f} | "
          f"{te['test_kl']:.4f} ± {te['test_kl_std']:.4f} | "
          f"{te['test_masked_acc']:.4f} ± {te['test_masked_acc_std']:.4f} | "
          f"{te['test_exact_match']:.4f} ± {te['test_exact_match_std']:.4f} |")

# GP
if 'test_evaluation' in gp_summary:
    te = gp_summary['test_evaluation']
    print(f"| GP      | {te['relative_sparsity']:.2f} | "
          f"{te['full_sparsity']:.2f} | "
          f"{te['test_kl']:.4f} ± {te['test_kl_std']:.4f} | "
          f"{te['test_masked_acc']:.4f} ± {te['test_masked_acc_std']:.4f} | "
          f"{te['test_exact_match']:.4f} ± {te['test_exact_match_std']:.4f} |")

print()
print("---")
print()

# Additional details table
print("## Detailed Test Metrics")
print()
print("| Dataset | Num Active | Test Loss | Logit Diff | Training Steps |")
print("|---------|------------|-----------|------------|----------------|")

if 'test_evaluation' in ioi_summary:
    te = ioi_summary['test_evaluation']
    print(f"| IOI     | {te['num_active_components']:>10} | "
          f"{te['test_loss']:.4f} ± {te['test_kl_std']:.4f} | "
          f"{te['test_logit_diff']:.4f} ± {te['test_logit_diff_std']:.4f} | "
          f"{te['step']:>14} |")

if 'test_evaluation' in gt_summary:
    te = gt_summary['test_evaluation']
    print(f"| GT      | {te['num_active_components']:>10} | "
          f"{te['test_loss']:.4f} ± {te['test_kl_std']:.4f} | "
          f"{te['test_logit_diff']:.4f} ± {te['test_logit_diff_std']:.4f} | "
          f"{te['step']:>14} |")

if 'test_evaluation' in gp_summary:
    te = gp_summary['test_evaluation']
    print(f"| GP      | {te['num_active_components']:>10} | "
          f"{te['test_loss']:.4f} ± {te['test_kl_std']:.4f} | "
          f"{te['test_logit_diff']:.4f} ± {te['test_logit_diff_std']:.4f} | "
          f"{te['step']:>14} |")

print()
print("---")
print()

# Comparison at similar relative sparsity levels
print("## Comparison at Target Relative Sparsity Levels")
print()
print("Showing which dataset achieves what metrics at similar sparsity levels:")
print()
print("| Relative Sparsity | Dataset | Full Sparsity | KL Div | Accuracy | Exact Match |")
print("|-------------------|---------|---------------|--------|----------|-------------|")

# Group by approximate sparsity levels
sparsity_data = []
if 'test_evaluation' in ioi_summary:
    te = ioi_summary['test_evaluation']
    sparsity_data.append({
        'rel_sparsity': te['relative_sparsity'],
        'dataset': 'IOI',
        'full_sparsity': te['full_sparsity'],
        'kl': te['test_kl'],
        'kl_std': te['test_kl_std'],
        'acc': te['test_masked_acc'],
        'acc_std': te['test_masked_acc_std'],
        'em': te['test_exact_match'],
        'em_std': te['test_exact_match_std']
    })

if 'test_evaluation' in gt_summary:
    te = gt_summary['test_evaluation']
    sparsity_data.append({
        'rel_sparsity': te['relative_sparsity'],
        'dataset': 'GT',
        'full_sparsity': te['full_sparsity'],
        'kl': te['test_kl'],
        'kl_std': te['test_kl_std'],
        'acc': te['test_masked_acc'],
        'acc_std': te['test_masked_acc_std'],
        'em': te['test_exact_match'],
        'em_std': te['test_exact_match_std']
    })

if 'test_evaluation' in gp_summary:
    te = gp_summary['test_evaluation']
    sparsity_data.append({
        'rel_sparsity': te['relative_sparsity'],
        'dataset': 'GP',
        'full_sparsity': te['full_sparsity'],
        'kl': te['test_kl'],
        'kl_std': te['test_kl_std'],
        'acc': te['test_masked_acc'],
        'acc_std': te['test_masked_acc_std'],
        'em': te['test_exact_match'],
        'em_std': te['test_exact_match_std']
    })

# Sort by relative sparsity
sparsity_data.sort(key=lambda x: x['rel_sparsity'])

for data in sparsity_data:
    print(f"| {data['rel_sparsity']:17.2f} | "
          f"{data['dataset']:>7} | "
          f"{data['full_sparsity']:13.2f} | "
          f"{data['kl']:.4f} | "
          f"{data['acc']:.4f} | "
          f"{data['em']:.4f} |")

print()
print("=" * 140)
print()
print("**Key Definitions:**")
print("- **Relative Sparsity**: Percentage of attention head components pruned (out of 37,020 total)")
print("- **Full Sparsity**: Percentage of all model components pruned (out of 239,772 total, includes MLPs)")
print("- **KL Divergence**: KL divergence between pruned and original model outputs (lower is better)")
print("- **Accuracy**: Masked accuracy on the task")
print("- **Exact Match**: Proportion of exactly correct predictions")
print("- **Num Active**: Number of active (non-pruned) components")
print("- **Logit Diff**: Difference in logits between correct and incorrect answers (higher is better)")
print()
print("**Notes:**")
print("- All metrics are from test set evaluation with standard deviation across multiple evaluation runs")
print("- GT uses exact match as primary metric (accuracy=0 is expected)")
print("- IOI data from IOI_20251022_213530 (has test evaluation)")
print("- GT data from GT_20251023_012757")
print("- GP data from GP_20251023_012746")
print()
print("=" * 140)
