"""Data processing utilities for RL training"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    datasets = None
    HAS_DATASETS = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

if TYPE_CHECKING:
    import datasets as datasets_typing
    import transformers as transformers_typing

from ..utils.logger import get_logger


class DataProcessor:
    """Main data processor for RL training datasets"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 max_length: int = 512):
        
        # Check dependencies
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        if config is not None:
            # Initialize from config (for main.py usage)
            self.config = config
            self.data_config = config.get('data', {})
            self.model_config = config.get('model', {})
            self.logger = get_logger()
            
            # Initialize tokenizer from config
            tokenizer_name = (
                self.model_config.get('tokenizer_name_or_path') or 
                self.model_config.get('model_name_or_path')
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Get max_length from config
            self.max_length = self.data_config.get('max_length', 512)
        else:
            # Initialize from direct parameters (for example_training.py usage)
            if tokenizer is None:
                raise ValueError("Either config or tokenizer must be provided")
            
            self.config = {}
            self.data_config = {}
            self.model_config = {}
            self.logger = get_logger()
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def process_data(self) -> None:
        """Main data processing pipeline"""
        # Check dependencies
        if not HAS_DATASETS:
            self.logger.warning("datasets library not available. Data processing will be skipped.")
            self.logger.info("To enable data processing, install datasets: pip install datasets")
            return
            
        self.logger.info("Starting data processing pipeline")
        
        # Load raw datasets
        train_dataset = self._load_dataset('train')
        eval_dataset = self._load_dataset('validation')
        
        if train_dataset is None:
            raise ValueError("No training dataset found")
        
        # Process datasets based on task type
        processed_train = self._process_dataset(train_dataset, 'train')
        processed_eval = self._process_dataset(eval_dataset, 'validation') if eval_dataset else None
        
        # Save processed datasets
        self._save_processed_dataset(processed_train, 'train')
        if processed_eval:
            self._save_processed_dataset(processed_eval, 'validation')
        
        self.logger.info("Data processing completed")
    
    def _load_dataset(self, split: str) -> Optional[Any]:
        """Load dataset for given split"""
        if not HAS_DATASETS:
            raise ImportError("datasets library is required. Install with: pip install datasets")
            
        dataset_name = self.data_config.get('dataset_name')
        dataset_config = self.data_config.get('dataset_config')
        
        # Try loading from HuggingFace datasets
        if dataset_name:
            try:
                dataset = datasets.load_dataset(
                    dataset_name, 
                    dataset_config, 
                    split=split,
                    cache_dir=self.data_config.get('cache_dir')
                )
                self.logger.info(f"Loaded {split} dataset from HuggingFace: {len(dataset)} samples")
                return dataset
            except Exception as e:
                self.logger.warning(f"Failed to load dataset from HuggingFace: {e}")
        
        # Try loading from local files
        file_key = f"{split}_file" if split != 'validation' else 'validation_file'
        file_path = self.data_config.get(file_key)
        
        if file_path and os.path.exists(file_path):
            dataset = self._load_local_file(file_path)
            self.logger.info(f"Loaded {split} dataset from local file: {len(dataset)} samples")
            return dataset
        
        self.logger.warning(f"No {split} dataset found")
        return None
    
    def _load_local_file(self, file_path: str) -> Any:
        """Load dataset from local file"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_ext == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return datasets.Dataset.from_list(data)
    
    def _process_dataset(self, dataset: Any, split: str) -> Any:
        """Process dataset based on task requirements"""
        self.logger.info(f"Processing {split} dataset")
        
        # Apply preprocessing
        dataset = dataset.map(
            self._preprocess_example,
            batched=True,
            batch_size=1000,
            num_proc=self.data_config.get('preprocessing_num_workers', 4),
            remove_columns=dataset.column_names,
            desc=f"Preprocessing {split} dataset"
        )
        
        # Filter out examples that are too long
        original_size = len(dataset)
        dataset = dataset.filter(
            lambda x: len(x['input_ids']) <= self.max_length,
            desc="Filtering long examples"
        )
        filtered_size = len(dataset)
        
        if filtered_size < original_size:
            self.logger.info(
                f"Filtered out {original_size - filtered_size} examples "
                f"({(original_size - filtered_size) / original_size * 100:.1f}%) "
                f"that exceed max length {self.max_length}"
            )
        
        # Limit dataset size if specified
        max_samples_key = f"max_{split}_samples" if split != 'validation' else 'max_eval_samples'
        max_samples = self.data_config.get(max_samples_key)
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            self.logger.info(f"Limited {split} dataset to {max_samples} samples")
        
        return dataset
    
    def _preprocess_example(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess a batch of examples"""
        # This is a base implementation - should be overridden for specific tasks
        processed = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for i in range(len(examples[list(examples.keys())[0]])):
            # Extract text based on common field names
            text = self._extract_text_from_example(examples, i)
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            
            processed['input_ids'].append(tokenized['input_ids'])
            processed['attention_mask'].append(tokenized['attention_mask'])
            processed['labels'].append(tokenized['input_ids'].copy())  # For language modeling
        
        return processed
    
    def _extract_text_from_example(self, examples: Dict[str, List], index: int) -> str:
        """Extract text from example based on common field names"""
        # Try common field names
        text_fields = ['text', 'prompt', 'input', 'question', 'instruction']
        
        for field in text_fields:
            if field in examples:
                return examples[field][index]
        
        # If no common field found, concatenate all string fields
        text_parts = []
        for key, values in examples.items():
            if isinstance(values[index], str):
                text_parts.append(f"{key}: {values[index]}")
        
        return " ".join(text_parts)
    
    def _save_processed_dataset(self, dataset: Any, split: str) -> None:
        """Save processed dataset"""
        output_dir = os.path.join(self.config['training']['output_dir'], 'processed_data')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"{split}_dataset")
        dataset.save_to_disk(save_path)
        
        self.logger.info(f"Saved processed {split} dataset to {save_path}")
    
    def load_dataset(self, file_path: str, dataset_type: str = "sft") -> Any:
        """Load and process dataset from file
        
        Args:
            file_path: Path to the dataset file
            dataset_type: Type of dataset ("sft" or "preference")
            
        Returns:
            Processed dataset
        """
        # Check dependencies
        if not HAS_DATASETS:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required. Install with: pip install transformers")
            
        self.logger.info(f"Loading {dataset_type} dataset from: {file_path}")
        
        # Load raw data
        if file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path).to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Create dataset
        dataset = datasets.Dataset.from_list(data)
        
        # Process based on dataset type
        if dataset_type == "sft":
            # Process for SFT training
            def process_sft_example(example):
                # Extract text and tokenize
                if 'text' in example:
                    text = example['text']
                elif 'prompt' in example and 'response' in example:
                    text = f"{example['prompt']}{self.tokenizer.eos_token}{example['response']}"
                else:
                    raise ValueError("SFT dataset must have 'text' or 'prompt'+'response' fields")
                
                # Tokenize
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=self.max_length,
                    return_tensors=None
                )
                
                return {
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask'],
                    'labels': tokens['input_ids'].copy()  # For causal LM
                }
            
            dataset = dataset.map(process_sft_example, remove_columns=dataset.column_names)
            
        elif dataset_type == "preference":
            # Process for preference learning (DPO, etc.)
            def process_preference_example(example):
                # Extract prompt, chosen, and rejected
                prompt = example.get('prompt', '')
                chosen = example.get('chosen', '')
                rejected = example.get('rejected', '')
                
                # Tokenize each part
                prompt_tokens = self.tokenizer(
                    prompt,
                    truncation=True,
                    padding=False,
                    max_length=self.max_length // 3,  # Reserve space for responses
                    return_tensors=None
                )
                
                chosen_tokens = self.tokenizer(
                    chosen,
                    truncation=True,
                    padding=False,
                    max_length=self.max_length // 3,
                    return_tensors=None
                )
                
                rejected_tokens = self.tokenizer(
                    rejected,
                    truncation=True,
                    padding=False,
                    max_length=self.max_length // 3,
                    return_tensors=None
                )
                
                return {
                    'prompt_input_ids': prompt_tokens['input_ids'],
                    'prompt_attention_mask': prompt_tokens['attention_mask'],
                    'chosen_input_ids': chosen_tokens['input_ids'],
                    'chosen_attention_mask': chosen_tokens['attention_mask'],
                    'rejected_input_ids': rejected_tokens['input_ids'],
                    'rejected_attention_mask': rejected_tokens['attention_mask']
                }
            
            dataset = dataset.map(process_preference_example, remove_columns=dataset.column_names)
        
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        self.logger.info(f"Dataset loaded and processed. Size: {len(dataset)}")
        return dataset


class PreferenceDataProcessor(DataProcessor):
    """Data processor for preference learning (DPO, etc.)"""
    
    def _preprocess_example(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess preference examples"""
        processed = {
            'prompt_input_ids': [],
            'prompt_attention_mask': [],
            'chosen_input_ids': [],
            'chosen_attention_mask': [],
            'rejected_input_ids': [],
            'rejected_attention_mask': []
        }
        
        for i in range(len(examples[list(examples.keys())[0]])):
            # Extract prompt, chosen, and rejected responses
            prompt = examples.get('prompt', [''])[i]
            chosen = examples.get('chosen', [''])[i]
            rejected = examples.get('rejected', [''])[i]
            
            # Tokenize prompt
            prompt_tokenized = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=self.max_length // 2,  # Leave space for response
                return_tensors=None
            )
            
            # Tokenize chosen response
            chosen_full = prompt + chosen
            chosen_tokenized = self.tokenizer(
                chosen_full,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            
            # Tokenize rejected response
            rejected_full = prompt + rejected
            rejected_tokenized = self.tokenizer(
                rejected_full,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            
            processed['prompt_input_ids'].append(prompt_tokenized['input_ids'])
            processed['prompt_attention_mask'].append(prompt_tokenized['attention_mask'])
            processed['chosen_input_ids'].append(chosen_tokenized['input_ids'])
            processed['chosen_attention_mask'].append(chosen_tokenized['attention_mask'])
            processed['rejected_input_ids'].append(rejected_tokenized['input_ids'])
            processed['rejected_attention_mask'].append(rejected_tokenized['attention_mask'])
        
        return processed


class RLHFDataProcessor(DataProcessor):
    """Data processor for RLHF training"""
    
    def _preprocess_example(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Preprocess RLHF examples"""
        processed = {
            'query_input_ids': [],
            'query_attention_mask': [],
            'response_input_ids': [],
            'response_attention_mask': [],
            'rewards': []
        }
        
        for i in range(len(examples[list(examples.keys())[0]])):
            # Extract query and response
            query = examples.get('query', examples.get('prompt', ['']))[i]
            response = examples.get('response', [''])[i]
            reward = examples.get('reward', [0.0])[i]
            
            # Tokenize query
            query_tokenized = self.tokenizer(
                query,
                truncation=True,
                padding=False,
                max_length=self.max_length // 2,
                return_tensors=None
            )
            
            # Tokenize response
            response_tokenized = self.tokenizer(
                response,
                truncation=True,
                padding=False,
                max_length=self.max_length // 2,
                return_tensors=None
            )
            
            processed['query_input_ids'].append(query_tokenized['input_ids'])
            processed['query_attention_mask'].append(query_tokenized['attention_mask'])
            processed['response_input_ids'].append(response_tokenized['input_ids'])
            processed['response_attention_mask'].append(response_tokenized['attention_mask'])
            processed['rewards'].append(float(reward))
        
        return processed