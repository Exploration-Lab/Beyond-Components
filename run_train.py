#!/usr/bin/env python
"""
Convenience script to run training with proper imports.

Usage:
    python run_train.py --config configs/gp_config.yaml
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set data directory
os.environ['CIRCUIT_SUBSPACE_DATA_DIR'] = os.path.join(project_root, 'data')

# Import and run the training script
from experiments.train import main

if __name__ == "__main__":
    main()
