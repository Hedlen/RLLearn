"""Data utilities for RL training"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
from pathlib import Path
import pandas as pd

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    datasets = None
    HAS_DATASETS = False

if TYPE_CHECKING:
    import datasets as datasets_typing
    from transformers import PreTrainedTokenizer as PreTrainedTokenizer_typing

try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS_UTILS = True
except ImportError:
    PreTrainedTokenizer = None
    HAS_TRANSFORMERS_UTILS = False

import torch
import numpy as np
from tqdm import tqdm


def load_dataset(dataset_name_or_path: str,
                dataset_config: Optional[str] = None,
                split: Optional[str] = None,
                cache_dir: Optional[str] = None,
                streaming: bool = False) -> Any:
    """Load dataset from HuggingFace or local path
    
    Args:
        dataset_name_or_path: Dataset name or local path
        dataset_config: Dataset configuration name
        split: Dataset split to load
        cache_dir: Cache directory
        streaming: Whether to use streaming mode
        
    Returns:
        Loaded dataset
    """
    try:
        # Try loading from HuggingFace
        dataset = datasets.load_dataset(
            dataset_name_or_path,
            dataset_config,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming
        )
        return dataset
    except Exception as e:
        # Try loading from local path
        if os.path.exists(dataset_name_or_path):
            if os.path.isdir(dataset_name_or_path):
                dataset = datasets.load_from_disk(dataset_name_or_path)
                if split and split in dataset:
                    return dataset[split]
                return dataset
            else:
                # Load from file
                return load_dataset_from_file(dataset_name_or_path)
        else:
            raise ValueError(f"Cannot load dataset: {e}")


def load_dataset_from_file(file_path: str) -> Any:
    """Load dataset from local file
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Loaded dataset
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_ext == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = [{'text': line.strip()} for line in lines if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return datasets.Dataset.from_list(data)


def preprocess_text(text: str, 
                   max_length: Optional[int] = None,
                   remove_special_chars: bool = False,
                   normalize_whitespace: bool = True) -> str:
    """Preprocess text for training
    
    Args:
        text: Input text
        max_length: Maximum text length
        remove_special_chars: Whether to remove special characters
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = ' '.join(text.split())
    
    # Remove special characters if requested
    if remove_special_chars:
        import re
        text = re.sub(r'[^\w\s]', '', text)
    
    # Truncate if necessary
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def tokenize_batch(texts: List[str],
                   tokenizer: Any,
                   max_length: int = 512,
                   padding: str = 'max_length',
                   truncation: bool = True,
                   return_tensors: str = 'pt') -> Dict[str, torch.Tensor]:
    """Tokenize a batch of texts
    
    Args:
        texts: List of texts to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        return_tensors: Format of returned tensors
        
    Returns:
        Tokenized batch
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )


def create_preference_pairs(dataset: Any,
                           response_column: str = 'response',
                           score_column: str = 'score',
                           prompt_column: str = 'prompt') -> Any:
    """Create preference pairs from scored responses
    
    Args:
        dataset: Dataset with scored responses
        response_column: Column containing responses
        score_column: Column containing scores
        prompt_column: Column containing prompts
        
    Returns:
        Dataset with preference pairs
    """
    # Group by prompt
    prompt_groups = {}
    for example in dataset:
        prompt = example[prompt_column]
        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append(example)
    
    # Create preference pairs
    preference_pairs = []
    for prompt, responses in prompt_groups.items():
        if len(responses) < 2:
            continue
        
        # Sort by score
        responses.sort(key=lambda x: x[score_column], reverse=True)
        
        # Create pairs
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                chosen = responses[i]
                rejected = responses[j]
                
                preference_pairs.append({
                    'prompt': prompt,
                    'chosen': chosen[response_column],
                    'rejected': rejected[response_column],
                    'chosen_score': chosen[score_column],
                    'rejected_score': rejected[score_column]
                })
    
    return datasets.Dataset.from_list(preference_pairs)


def split_dataset(dataset: Any,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 seed: int = 42) -> Tuple[Any, ...]:
    """Split dataset into train/validation/test sets
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Split dataset
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    return train_dataset, val_dataset, test_dataset


def filter_by_length(dataset: Any,
                    tokenizer: Any,
                    min_length: int = 10,
                    max_length: int = 512,
                    text_column: str = 'text') -> Any:
    """Filter dataset by text length
    
    Args:
        dataset: Dataset to filter
        tokenizer: Tokenizer for length calculation
        min_length: Minimum token length
        max_length: Maximum token length
        text_column: Column containing text
        
    Returns:
        Filtered dataset
    """
    def filter_fn(example):
        text = example[text_column]
        tokens = tokenizer.encode(text)
        return min_length <= len(tokens) <= max_length
    
    return dataset.filter(filter_fn)


def balance_dataset(dataset: Any,
                   label_column: str = 'label',
                   max_samples_per_class: Optional[int] = None) -> Any:
    """Balance dataset by class labels
    
    Args:
        dataset: Dataset to balance
        label_column: Column containing labels
        max_samples_per_class: Maximum samples per class
        
    Returns:
        Balanced dataset
    """
    # Group by label
    label_groups = {}
    for i, example in enumerate(dataset):
        label = example[label_column]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(i)
    
    # Determine target size
    if max_samples_per_class is None:
        target_size = min(len(indices) for indices in label_groups.values())
    else:
        target_size = max_samples_per_class
    
    # Sample indices
    balanced_indices = []
    for label, indices in label_groups.items():
        if len(indices) > target_size:
            sampled_indices = random.sample(indices, target_size)
        else:
            sampled_indices = indices
        balanced_indices.extend(sampled_indices)
    
    # Shuffle and return
    random.shuffle(balanced_indices)
    return dataset.select(balanced_indices)


def augment_text(text: str,
                augmentation_type: str = 'paraphrase',
                num_augmentations: int = 1) -> List[str]:
    """Augment text data
    
    Args:
        text: Input text
        augmentation_type: Type of augmentation
        num_augmentations: Number of augmentations to generate
        
    Returns:
        List of augmented texts
    """
    augmented_texts = [text]  # Include original
    
    if augmentation_type == 'paraphrase':
        # Simple paraphrasing (placeholder - would use actual paraphrasing model)
        for i in range(num_augmentations):
            # This is a placeholder - implement actual paraphrasing
            augmented_text = text.replace('.', f' (version {i+1}).')
            augmented_texts.append(augmented_text)
    
    elif augmentation_type == 'synonym':
        # Synonym replacement (placeholder)
        for i in range(num_augmentations):
            # This is a placeholder - implement actual synonym replacement
            augmented_text = text.replace('good', 'excellent' if i % 2 == 0 else 'great')
            augmented_texts.append(augmented_text)
    
    return augmented_texts[1:]  # Return only augmented versions


def create_conversation_dataset(conversations: List[List[Dict[str, str]]],
                              max_turns: int = 10) -> Any:
    """Create dataset from conversation data
    
    Args:
        conversations: List of conversations (each conversation is a list of turns)
        max_turns: Maximum number of turns per conversation
        
    Returns:
        Conversation dataset
    """
    processed_conversations = []
    
    for conversation in conversations:
        if len(conversation) > max_turns:
            conversation = conversation[:max_turns]
        
        processed_conversations.append({
            'conversation': conversation,
            'num_turns': len(conversation)
        })
    
    return datasets.Dataset.from_list(processed_conversations)


def compute_dataset_stats(dataset: Any,
                         text_columns: List[str] = None,
                         tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """Compute statistics for dataset
    
    Args:
        dataset: Dataset to analyze
        text_columns: Columns containing text
        tokenizer: Tokenizer for token-level stats
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_examples': len(dataset),
        'columns': dataset.column_names
    }
    
    if text_columns is None:
        text_columns = ['text', 'prompt', 'response', 'chosen', 'rejected']
        text_columns = [col for col in text_columns if col in dataset.column_names]
    
    for col in text_columns:
        if col in dataset.column_names:
            texts = dataset[col]
            
            # Character-level stats
            char_lengths = [len(text) for text in texts]
            stats[f'{col}_char_length'] = {
                'mean': np.mean(char_lengths),
                'std': np.std(char_lengths),
                'min': np.min(char_lengths),
                'max': np.max(char_lengths),
                'median': np.median(char_lengths)
            }
            
            # Word-level stats
            word_lengths = [len(text.split()) for text in texts]
            stats[f'{col}_word_length'] = {
                'mean': np.mean(word_lengths),
                'std': np.std(word_lengths),
                'min': np.min(word_lengths),
                'max': np.max(word_lengths),
                'median': np.median(word_lengths)
            }
            
            # Token-level stats (if tokenizer provided)
            if tokenizer:
                token_lengths = []
                for text in tqdm(texts[:1000], desc=f"Tokenizing {col}"):  # Sample for efficiency
                    tokens = tokenizer.encode(text)
                    token_lengths.append(len(tokens))
                
                stats[f'{col}_token_length'] = {
                    'mean': np.mean(token_lengths),
                    'std': np.std(token_lengths),
                    'min': np.min(token_lengths),
                    'max': np.max(token_lengths),
                    'median': np.median(token_lengths)
                }
    
    return stats


def save_dataset_info(dataset: Any,
                     save_path: str,
                     include_samples: int = 5) -> None:
    """Save dataset information to file
    
    Args:
        dataset: Dataset to save info for
        save_path: Path to save info file
        include_samples: Number of sample examples to include
    """
    info = {
        'num_examples': len(dataset),
        'columns': dataset.column_names,
        'features': str(dataset.features),
        'samples': []
    }
    
    # Add sample examples
    for i in range(min(include_samples, len(dataset))):
        info['samples'].append(dataset[i])
    
    # Save to file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def merge_datasets(datasets_list: List[Any],
                  axis: int = 0) -> Any:
    """Merge multiple datasets
    
    Args:
        datasets_list: List of datasets to merge
        axis: Axis to merge along (0 for concatenation)
        
    Returns:
        Merged dataset
    """
    if axis == 0:
        return datasets.concatenate_datasets(datasets_list)
    else:
        raise ValueError("Only axis=0 (concatenation) is currently supported")