"""
Utility functions for Circuit Subspace project
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from . import constants
import numpy as np
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Set the random seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, cache_dir):
    """
    Load a HookedTransformer model and tokenizer.

    Args:
        model_name: Name of the model (e.g., 'gpt2-small')
        cache_dir: Directory to cache model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    constants.HUGGINGFACE_HUB_CACHE = cache_dir
    model = HookedTransformer.from_pretrained(model_name, cache_dir=cache_dir, device=device)
    tokenizer = model.tokenizer
    return model, tokenizer


def get_label_column_names(data_type='ioi'):
    """
    Get the column names for correct and corrupted labels based on the data type.

    Args:
        data_type (str): The type of data being processed. Can be 'ioi', 'ioi_t1', 'gt', or 'gp'.

    Returns:
        tuple: A tuple containing (correct_label_column, corrupted_label_column).
            - For IOI: indirect object names
            - For GP: pronouns
            - For GT: century numbers
    """
    if data_type == 'ioi' or data_type == 'ioi_t1':
        return 'ioi_sentences_labels', 'ioi_sentences_labels_wrong'
    elif data_type == 'gp':
        return 'pronoun', 'corr_pronoun'
    elif data_type == 'gt':
        return 'century', 'corr_century'
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Expected 'ioi', 'ioi_t1', 'gt', or 'gp'.")


def get_indirect_objects_and_subjects(data_type='ioi'):
    """
    Get column names for labels. Alias for get_label_column_names for backwards compatibility.
    """
    return get_label_column_names(data_type)


def get_data_column_names(data_type='ioi'):
    """
    Get column names for clean and corrupted input data.

    Args:
        data_type: Type of dataset ('ioi', 'ioi_t1', 'gt', 'gp')

    Returns:
        Tuple of (clean_column, corrupted_column)
    """
    if data_type == 'ioi' or data_type == 'ioi_t1':
        column_clean, column_corrupted = 'ioi_sentences_input', 'corr_ioi_sentences_input'
    elif data_type == 'gt' or data_type == 'gp':
        column_clean = 'prefix'
        column_corrupted = 'corr_prefix'
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return column_clean, column_corrupted


def get_data_label_column_names(data_type='ioi'):
    """
    Get column names for clean and wrong labels.

    Args:
        data_type: Type of dataset

    Returns:
        Tuple of (clean_label_column, wrong_label_column)
    """
    if data_type == 'ioi' or data_type == 'ioi_t1':
        column_clean_label, column_wrong_label = 'ioi_sentences_labels', 'ioi_sentences_labels_wrong'
    elif data_type == 'gp':
        column_clean_label = 'pronoun'
        column_wrong_label = 'corr_pronoun'
    elif data_type == 'gt':
        raise ValueError('GT task requires special handling for labels')
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return column_clean_label, column_wrong_label
