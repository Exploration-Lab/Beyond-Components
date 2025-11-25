# Circuit Subspace: SVD-based Transformer Circuit Discovery

This repository implements a method for discovering interpretable circuits in transformer language models using Singular Value Decomposition (SVD) and learnable masks on the singular value components.

## Overview

The method decomposes each attention head's OV (Output-Value) and QK (Query-Key) matrices using SVD, then learns sparse masks over the singular value directions. This enables:

1. **Identifying minimal circuits** - Finding the smallest set of singular value directions that preserve model behavior
2. **Intervening on specific directions** - Modifying activations along identified directions to control model predictions
3. **Mathematical interpretability** - Each direction has a clear mathematical interpretation

<p align="center">
  <img src="images/intervention.png" alt="Intervention Method Illustration" width="800"/>
</p>

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
Beyond-Components/
├── src/                          # Core source code
│   ├── models/
│   │   └── masked_transformer_circuit.py
│   │       # Main circuit discovery model with SVD decomposition
│   │       # Implements learnable masks on singular value components
│   │       # Handles QK (Query-Key) and OV (Output-Value) matrices
│   │
│   ├── data/
│   │   ├── data_loader.py        # Dataset loaders for all tasks
│   │   │   # - load_gp_dataset(): Gender Pronoun task
│   │   │   # - load_ioi_dataset(): Indirect Object Identification
│   │   │   # - load_gt_dataset(): Greater-Than task
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── utils.py              # Core utility functions
│       │   # - Model loading and initialization
│       │   # - Data column name helpers
│       │   # - Seed setting for reproducibility
│       ├── visualization.py      # Plotting and visualization
│       │   # - visualize_masks(): Heatmap visualization
│       │   # - plot_training_history(): Loss curves
│       │   # - visualize_masked_singular_values()
│       ├── constants.py          # Project-wide constants
│       └── __init__.py
│
├── experiments/
│   ├── train.py                  # Main training script
│   │   # Trains circuit discovery model with mask learning
│   │   # Supports W&B logging, checkpointing, visualization
│   │   # Usage: python experiments/train.py --config configs/gp_config.yaml
│   │
│   ├── ablation/                 # Intervention experiments
│   │   ├── intervention.py
│   │   │   # Swap activations to empirically observed values
│   │   │   # Tests discovered circuits on gender flip task
│   │   └── comprehensive_sigma_test.py
│   │       # Systematic sigma amplification testing
│   │       # Tests effect of different sigma multipliers
│   │
│   └── evaluation/               # Metrics and analysis
│       ├── comprehensive_metrics_table.py
│       │   # Generate sparsity vs accuracy tables
│       │   # Analyze trade-offs in circuit discovery
│       └── generate_sigma_table.py
│           # Generate results tables for interventions
│
├── configs/                      # YAML configuration files
│   ├── gp_config.yaml            # Gender Pronoun task config
│   ├── ioi_config.yaml           # Indirect Object Identification config
│   └── gt_config.yaml            # Greater-Than task config
│
├── data/                         # Datasets directory
│   ├── data_main.zip             # Complete dataset archive (48MB)
│   │   # Contains train/val/test splits for all tasks
│   │   # Extract: unzip data/data_main.zip -d data/
│   └── .gitkeep
│
├── checkpoints/                  # Trained model checkpoints (created during training)
│   └── .gitkeep
│
├── run_train.py                  # Convenience wrapper for training
├── run_ablation.py               # Convenience wrapper for ablation experiments
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation script
├── .gitignore                    # Git ignore patterns
├── LICENSE                       # MIT License
└── README.md                     # This file
```

### Key Components

**Core Model**: `MaskedTransformerCircuit` performs SVD decomposition on attention matrices and learns sparse masks to identify minimal circuits.

**Tasks Supported**:
- **GP (Gender Pronoun)**: Predict gender pronouns in context
- **IOI (Indirect Object Identification)**: Identify indirect objects in sentences
- **GT (Greater-Than)**: Compare numerical values

**Experiment Pipeline**:
1. Train circuit discovery model (`experiments/train.py`)
2. Run interventions to test circuits (`experiments/ablation/intervention.py`)
3. Generate evaluation metrics (`experiments/evaluation/`)


## Quick Start

### 1. Prepare Data

Extract the provided dataset archive:

```bash
cd data/
unzip data_main.zip
cd ..
```

This will create a `data/data_main/` directory with all required datasets.

**Dataset Structure:**

**Gender Pronoun (GP) task:**
- Files: `train_1k_gp.csv`, `val_gp.csv`, `test_gp.csv`
- Columns: `prefix`, `pronoun`, `name`, `corr_prefix`, `corr_pronoun`, `corr_name`

**Indirect Object Identification (IOI) task:**
- Files: `train_1k_ioi.csv`, `train_5k_ioi.csv`, `val_ioi.csv`, `test_ioi.csv`
- Columns: `ioi_sentences_input`, `ioi_sentences_labels`, `corr_ioi_sentences_input`, etc.

**Greater-Than (GT) task:**
- Files: `train_gt_1k.csv`, `train_gt_2k.csv`, `train_gt_3k.csv`, `val_gt.csv`, `test_gt.csv`

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
| `masking.mask_init_value` | Initial mask values | 0.99 |
| `masking.sparsity_threshold` | Threshold for "active" | 1e-3 |

## Citation

If you use this code in your research, please cite the associated paper.

## License

This work is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) - Creative Commons Attribution-ShareAlike 4.0 International License.

## Acknowledgments

This work builds on:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for transformer interpretability
- Research on mechanistic interpretability from Anthropic and others
