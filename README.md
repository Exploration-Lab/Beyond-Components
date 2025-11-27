# Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits


This repository contains the official release of the following paper:
> **Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits**<br>
> **Paper:** https://arxiv.org/abs/2511.20273

> **Authors:** Areeb Ahmad*, Abhinav Joshi*, Ashutosh Modi<br>
>
> **Abstract:** *Transformer-based language models exhibit complex and distributed behavior, yet their internal computations remain poorly understood. Existing mechanistic interpretability methods typically treat attention heads and multilayer perceptron layers (MLPs) (the building blocks of a transformer architecture) as indivisible units, overlooking possibilities of functional substructure learned within them. In this work, we introduce a more fine-grained perspective that decomposes these components into orthogonal singular directions, revealing superposed and independent computations within a single head or MLP. We validate our perspective on widely used standard tasks like Indirect Object Identification (IOI), Gender Pronoun (GP), and Greater Than (GT), showing that previously identified canonical functional heads, such as the “name mover,” encode multiple overlapping subfunctions aligned with distinct singular directions. Nodes in a computational graph, that are previously identified as circuit elements show strong activation along specific low-rank directions, suggesting that meaningful computations reside in compact subspaces. While some directions remain challenging to interpret fully, our results highlight that transformer computations are more distributed, structured, and compositional than previously assumed. This perspective opens new avenues for fine-grained mechanistic interpretability and a deeper understanding of model internals.*

![Teaser image](images/intervention.png)
**Picture:** *The Figure shows an illustration of the intervention process. Attention heads produce value vectors whose OV projections decompose into singular directions representing fixed logit receptors. By swapping the activation coefficients of gender-sensitive directions with opposite-gender mean values, the intervention modifies only the targeted subspaces in the residual stream, leading to predictable shifts in ‘he’/‘she’ output logits.*


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


```

@inproceedings{
ahmad2025beyond,
title={Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits},
author={Areeb Ahmad and Abhinav Joshi and Ashutosh Modi},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=7UbXEQNny7}
}
```

## License

This work is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) - Creative Commons Attribution-ShareAlike 4.0 International License.

