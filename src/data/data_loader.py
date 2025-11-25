import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformer_lens
from ..utils.utils import get_data_column_names, get_label_column_names

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from torch.optim import AdamW



class IOIDataset(Dataset):
    """Custom Dataset for IOI sentences data"""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file
        """
        self.data = pd.read_csv(csv_file)
        # model , tokenizer = get_model(model_name='gpt2-small',cache_dir='cache_dir')
        # self.tokenizer = tokenizer
        
        tokenizer= AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        
        # Create a copy of the data first to avoid modifying the original during iteration
        data_copy = self.data.copy()
        rows_to_drop = []
        clean_idx_freq = {}
        corrupted_idx_freq = {}

        # for i, row in data_copy.iterrows():
        #     clean_input_ids = tokenizer(row['ioi_sentences_input'], return_tensors="pt")['input_ids']
        #     corrupted_input_ids = tokenizer(row['corr_ioi_sentences_input'], return_tensors="pt")['input_ids']
            
        #     if clean_input_ids.shape != corrupted_input_ids.shape:
        #         print(f"dropping row {i} - shapes: {clean_input_ids.shape} vs {corrupted_input_ids.shape}")
        #         #collect the frequency of the mismatched shapes
        #         if clean_input_ids.shape[1] not in clean_idx_freq:
        #             clean_idx_freq[clean_input_ids.shape[1]] = 1
        #         else:
        #             clean_idx_freq[clean_input_ids.shape[1]] += 1
                    
        #         if corrupted_input_ids.shape[1] not in corrupted_idx_freq:
        #             corrupted_idx_freq[corrupted_input_ids.shape[1]] = 1
                    
        #         else:
        #             corrupted_idx_freq[corrupted_input_ids.shape[1]] += 1
                
                
        #         # print the tokenized sentences
        #         # print(f"clean: {tokenizer.decode(clean_input_ids[0])}")
        #         # print(f"corrupted: {tokenizer.decode(corrupted_input_ids[0])}")
        #         rows_to_drop.append(i)
        #         # breakpoint()

        # # Drop all mismatched rows at once
        # if rows_to_drop:
        #     self.data = self.data.drop(rows_to_drop)
        #     self.data = self.data.reset_index(drop=True)  # Reset index after dropping rows
        #sort keys by their magnitude
        clean_idx_freq = dict(sorted(clean_idx_freq.items(), key=lambda item: item[0]))
        corrupted_idx_freq = dict(sorted(corrupted_idx_freq.items(), key=lambda item: item[0]))
        # breakpoint()   
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        
            
        # Get only the input sentences columns
        sample = {
            'ioi_sentences_input': self.data.iloc[idx]['ioi_sentences_input'],
            'ioi_sentences_labels': self.data.iloc[idx]['ioi_sentences_labels'],
            'corr_ioi_sentences_input': self.data.iloc[idx]['corr_ioi_sentences_input'],
            'corr_ioi_sentences_labels': self.data.iloc[idx]['corr_ioi_sentences_labels'],
            'ioi_sentences_labels_wrong': self.data.iloc[idx]['ioi_sentences_labels_wrong'],
            'corr_ioi_sentences_labels_wrong': self.data.iloc[idx]['corr_ioi_sentences_labels_wrong']
            
        }
        
        return sample

def load_ioi_dataset(data_dir=None, batch_size=32,
                        full_batch=False,
                        shuffle=True, num_workers=4,
                        validation=False,
                        train=False):
                            
        """
        Load the IOI dataset with batching
        
        Args:
            data_dir (str): Directory containing the dataset files
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of workers for data loading
            
        Returns:
            DataLoader: PyTorch DataLoader object containing the batched dataset with text pairs
        """
        if data_dir is None:
            # Look for data in project root/data/data_main
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'data', 'data_main')
            
        if train:
            # data_path = os.path.join(data_dir, 'train_60k_ioi.csv')
            # data_path = os.path.join(data_dir, 'train_400_ioi.csv')
            data_path = os.path.join(data_dir, 'train_1k_ioi.csv')
            # data_path = os.path.join(data_dir, 'train_5k_ioi.csv')
            
            
        elif validation:
            data_path = os.path.join(data_dir, 'val_ioi.csv')
            # data_path = os.path.join(data_dir, 'synthetic_val_ioi_dob.csv')
            # data_path = os.path.join(data_dir, 'synthetic_val_ioi.csv')
            
        else:
            data_path = os.path.join(data_dir, 'test_1k_ioi.csv')
            # data_path = os.path.join(data_dir, 'test_ioi.csv')
        
        # Create dataset
        dataset = IOIDataset(data_path)
        
        if full_batch:
            batch_size = len(dataset)
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return loader


def load_ioi_t1_dataset(data_dir='data_main/', batch_size=50,
                        full_batch=False,
                        shuffle=True, num_workers=4,
                        validation=False,
                        train=False):
                        
    """
    Load the IOI dataset with batching
    
    Args:
        data_dir (str): Directory containing the dataset files
        batch_size (int): Size of each batch
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader object containing the batched dataset with text pairs
    """
    if train:
        data_path = os.path.join(data_dir, 'train_ioi-t1.csv')
        
    elif validation:
        data_path = os.path.join(data_dir, 'val_ioi-t1.csv')
        
    else:
        data_path = os.path.join(data_dir, 'test_ioi-t1.csv')
    
    # Create dataset
    dataset = IOIDataset(data_path)
    
    if full_batch:
        batch_size = len(dataset)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return loader

# # Example usage:
# if __name__ == "__main__":
#     # Load the dataset with batching
#     train_loader = load_ioi_t1_dataset(batch_size=32)
    
#     # Example of iterating through batches
#     for batch in train_loader:
#         # Each batch will be a dictionary with:
#         # batch['ioi_sentences_input'] - list of strings (size: batch_size)
#         # batch['corr_ioi_sentences_input'] - list of strings (size: batch_size)
#         pass



class GPDataset(Dataset):
    #prefix,pronoun,template,name,corr_prefix,corr_pronoun,corr_template,corr_name
    """Custom Dataset for Gender pronoun data"""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file
        """
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get only the input sentences columns
        sample = {
            'prefix': self.data.iloc[idx]['prefix'],
            'pronoun': self.data.iloc[idx]['pronoun'],
            # 'template': self.data.iloc[idx]['template'],
            'name': self.data.iloc[idx]['name'],
            'corr_prefix': self.data.iloc[idx]['corr_prefix'],
            'corr_pronoun': self.data.iloc[idx]['corr_pronoun'],
            # 'corr_template': self.data.iloc[idx]['corr_template'],
            'corr_name': self.data.iloc[idx]['corr_name']
        }
        
        return sample
    











class GTDataset(Dataset):
    """Custom Dataset for Greater than sentences data"""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file
        """
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get only the input sentences columns
        sample = {
            # 'template': self.data.iloc[idx]['template'],
            'century': self.data.iloc[idx]['century'],
            'noun': self.data.iloc[idx]['noun'],
            'digits': self.data.iloc[idx]['digits'],
            'prefix': self.data.iloc[idx]['prefix'],
            # 'corr_template': self.data.iloc[idx]['corr_template'],
            'corr_century': self.data.iloc[idx]['corr_century'],
            'corr_noun': self.data.iloc[idx]['corr_noun'],
            'corr_digits': self.data.iloc[idx]['corr_digits'],
            'corr_prefix': self.data.iloc[idx]['corr_prefix']
        }
        
        return sample
    
def load_gt_dataset(data_dir=None, batch_size=2048,
                    full_batch=False,
                    shuffle=True, num_workers=4,
                    validation=False,
                    train=False):
                        
    """
    Load the Greater than dataset with batching
    
    Args:
        data_dir (str): Directory containing the dataset files
        batch_size (int): Size of each batch
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader object containing the batched dataset with text pairs
    """
    
    if data_dir is None:
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, 'data_main')
            
    if train:
        data_path = os.path.join(data_dir, 'train_gt_2k.csv')
        
    elif validation:
        data_path = os.path.join(data_dir, 'val_gt_500.csv')
        
    else:
        data_path = os.path.join(data_dir, 'test_gt.csv')
    
    # Create dataset
    dataset = GTDataset(data_path)
    
    if full_batch:
        batch_size = len(dataset)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    
    return loader


def load_gp_dataset(data_dir=None, batch_size=2048,
                    full_batch=False,
                    shuffle=True, num_workers=4,
                    validation=False,
                    train=False):
                            
        """
        Load the Greater than dataset with batching
        
        Args:
            data_dir (str): Directory containing the dataset files
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of workers for data loading
            
        Returns:
            DataLoader: PyTorch DataLoader object containing the batched dataset with text pairs
            
        """
        
        if data_dir is None:
            # Look for data in project root/data/data_main
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'data', 'data_main')
            
        if train:
            data_path = os.path.join(data_dir, 'train_1k_gp.csv')
            
        elif validation:
            data_path = os.path.join(data_dir, 'val_gp.csv')
            
        else:
            data_path = os.path.join(data_dir, 'test_gp.csv')
        
        # Create dataset
        dataset = GPDataset(data_path)
        
        if full_batch:
            batch_size = len(dataset)
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return loader   