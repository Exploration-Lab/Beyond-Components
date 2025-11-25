# Circuit Subspace: SVD-based Transformer Circuit Discovery

This repository implements a method for discovering interpretable circuits in transformer language models using Singular Value Decomposition (SVD) and learnable masks on the singular value components.

## Overview

The method decomposes each attention head's OV (Output-Value) and QK (Query-Key) matrices using SVD, then learns sparse masks over the singular value directions. This enables:

1. **Identifying minimal circuits** - Finding the smallest set of singular value directions that preserve model behavior
2. **Intervening on specific directions** - Modifying activations along identified directions to control model predictions
3. **Mathematical interpretability** - Each direction has a clear mathematical interpretation

## Installation

**Prerequisites:**
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

```bash
# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
Circuit_Subspace/
├── src/                          # Core source code
│   ├── models/
│   │   └── masked_transformer_circuit.py  # Main circuit discovery model
│   ├── data/
│   │   └── data_loader.py        # Dataset loaders for IOI, GP, GT tasks
│   └── utils/
│       ├── utils.py              # Utility functions
│       └── visualization.py      # Plotting and visualization
├── experiments/
│   ├── train.py                  # Training script
│   ├── ablation/
│   │   ├── intervention.py              # Activation swap interventions
│   │   └── comprehensive_sigma_test.py  # Sigma amplification experiments
│   └── evaluation/
│       ├── comprehensive_metrics_table.py  # Sparsity vs accuracy evaluation
│       └── generate_sigma_table.py         # Generate results tables
├── configs/
│   └── gp_config.yaml            # Configuration for GP task
├── data/                         # Place datasets here
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Prepare Data

Place your dataset CSV files in the `data/` directory. Expected format:

**For Gender Pronoun (GP) task:**
- `train_1k_gp.csv`, `val_gp.csv`, `test_gp.csv`
- Columns: `prefix`, `pronoun`, `name`, `corr_prefix`, `corr_pronoun`, `corr_name`

**For Indirect Object Identification (IOI) task:**
- `train_1k_ioi.csv`, `val_ioi.csv`, `test_1k_ioi.csv`
- Columns: `ioi_sentences_input`, `ioi_sentences_labels`, `corr_ioi_sentences_input`, etc.

### 2. Train a Circuit

```bash
python experiments/train.py --config configs/gp_config.yaml
```

This will:
- Load GPT-2 small and compute SVD for all attention heads
- Learn sparse masks over singular value directions
- Save the trained model and visualizations to `logs/`

### 3. Run Intervention Experiments

After training, run intervention experiments to test the discovered circuit:

```bash
python experiments/ablation/intervention.py
```

This swaps activations along identified directions to their empirically observed values for the opposite gender.

### 4. Generate Results Tables

```bash
python experiments/evaluation/generate_sigma_table.py
```

## Configuration Options

Key configuration parameters in `configs/gp_config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `training.learning_rate` | Mask learning rate | 2.0e-2 |
| `training.l1_weight` | Sparsity penalty weight | 1.95e-4 |
| `training.kl_opt` | Target KL divergence | 0.10 |
| `training.l1_opt` | Target L1 norm | 3500 |
| `masking.mask_init_value` | Initial mask values | 0.99 |
| `masking.sparsity_threshold` | Threshold for "active" | 1e-3 |

## Citation

If you use this code in your research, please cite the associated paper.

## License

MIT License

## Acknowledgments

This work builds on:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for transformer interpretability
- Research on mechanistic interpretability from Anthropic and others
