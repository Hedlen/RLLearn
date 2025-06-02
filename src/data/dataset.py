"""Dataset classes for RL training"""

from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
import torch
from torch.utils.data import Dataset

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    datasets = None
    HAS_DATASETS = False

try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    PreTrainedTokenizer = None
    HAS_TRANSFORMERS = False

if TYPE_CHECKING:
    import datasets as datasets_typing
    from transformers import PreTrainedTokenizer as PreTrainedTokenizer_typing


class RLDataset(Dataset):
    """Base dataset class for RL training"""
    
    def __init__(self, 
                 dataset: Any,
                 tokenizer: Any,
                 max_length: int = 512):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        return self._process_item(item)
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single item - to be implemented by subclasses"""
        raise NotImplementedError


class SFTDataset(RLDataset):
    """Dataset for Supervised Fine-Tuning (SFT)"""
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process SFT item"""
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class PreferenceDataset(RLDataset):
    """Dataset for preference learning (DPO, etc.)"""
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process preference item"""
        return {
            'prompt_input_ids': torch.tensor(item['prompt_input_ids'], dtype=torch.long),
            'prompt_attention_mask': torch.tensor(item['prompt_attention_mask'], dtype=torch.long),
            'chosen_input_ids': torch.tensor(item['chosen_input_ids'], dtype=torch.long),
            'chosen_attention_mask': torch.tensor(item['chosen_attention_mask'], dtype=torch.long),
            'rejected_input_ids': torch.tensor(item['rejected_input_ids'], dtype=torch.long),
            'rejected_attention_mask': torch.tensor(item['rejected_attention_mask'], dtype=torch.long)
        }


class PPODataset(RLDataset):
    """Dataset for PPO training"""
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process PPO item"""
        return {
            'query_input_ids': torch.tensor(item['query_input_ids'], dtype=torch.long),
            'query_attention_mask': torch.tensor(item['query_attention_mask'], dtype=torch.long),
            'response_input_ids': torch.tensor(item['response_input_ids'], dtype=torch.long),
            'response_attention_mask': torch.tensor(item['response_attention_mask'], dtype=torch.long),
            'rewards': torch.tensor(item['rewards'], dtype=torch.float)
        }


class RLHFDataset(RLDataset):
    """Dataset for RLHF training with human feedback"""
    
    def __init__(self, 
                 dataset: Any,
                 tokenizer: Any,
                 max_length: int = 512,
                 include_rewards: bool = True):
        super().__init__(dataset, tokenizer, max_length)
        self.include_rewards = include_rewards
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process RLHF item"""
        result = {
            'query_input_ids': torch.tensor(item['query_input_ids'], dtype=torch.long),
            'query_attention_mask': torch.tensor(item['query_attention_mask'], dtype=torch.long),
            'response_input_ids': torch.tensor(item['response_input_ids'], dtype=torch.long),
            'response_attention_mask': torch.tensor(item['response_attention_mask'], dtype=torch.long)
        }
        
        if self.include_rewards and 'rewards' in item:
            result['rewards'] = torch.tensor(item['rewards'], dtype=torch.float)
        
        return result


class ConversationDataset(RLDataset):
    """Dataset for multi-turn conversation training"""
    
    def __init__(self, 
                 dataset: Any,
                 tokenizer: Any,
                 max_length: int = 512,
                 max_turns: int = 10):
        super().__init__(dataset, tokenizer, max_length)
        self.max_turns = max_turns
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process conversation item"""
        # Assume conversation is stored as a list of turns
        conversation = item.get('conversation', [])
        
        # Truncate to max turns
        if len(conversation) > self.max_turns:
            conversation = conversation[:self.max_turns]
        
        # Concatenate conversation turns
        full_text = ""
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            full_text += f"{role}: {content}\n"
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': tokenized['input_ids'].squeeze(0).clone()
        }


class RewardModelDataset(RLDataset):
    """Dataset for training reward models"""
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process reward model item"""
        # For reward model training, we need pairs of (input, score)
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        score = torch.tensor(item.get('score', 0.0), dtype=torch.float)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': score
        }


class InstructionDataset(RLDataset):
    """Dataset for instruction following training"""
    
    def __init__(self, 
                 dataset: Any,
                 tokenizer: Any,
                 max_length: int = 512,
                 instruction_template: str = "Instruction: {instruction}\nResponse: {response}"):
        super().__init__(dataset, tokenizer, max_length)
        self.instruction_template = instruction_template
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process instruction item"""
        instruction = item.get('instruction', '')
        response = item.get('response', item.get('output', ''))
        
        # Format using template
        full_text = self.instruction_template.format(
            instruction=instruction,
            response=response
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        
        # For instruction following, we typically mask the instruction part in labels
        instruction_only = self.tokenizer(
            f"Instruction: {instruction}\nResponse: ",
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        instruction_length = instruction_only['input_ids'].shape[1]
        labels = input_ids.clone()
        labels[:instruction_length] = -100  # Ignore instruction tokens in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class DynamicDataset(RLDataset):
    """Dynamic dataset that can switch between different processing modes"""
    
    def __init__(self, 
                 dataset: Any,
                 tokenizer: Any,
                 max_length: int = 512,
                 mode: str = 'sft'):
        super().__init__(dataset, tokenizer, max_length)
        self.mode = mode
        self._processors = {
            'sft': self._process_sft,
            'preference': self._process_preference,
            'ppo': self._process_ppo,
            'instruction': self._process_instruction
        }
    
    def set_mode(self, mode: str):
        """Change processing mode"""
        if mode not in self._processors:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self._processors.keys())}")
        self.mode = mode
    
    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process item based on current mode"""
        return self._processors[self.mode](item)
    
    def _process_sft(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """SFT processing"""
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _process_preference(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preference processing"""
        return {
            'prompt_input_ids': torch.tensor(item['prompt_input_ids'], dtype=torch.long),
            'prompt_attention_mask': torch.tensor(item['prompt_attention_mask'], dtype=torch.long),
            'chosen_input_ids': torch.tensor(item['chosen_input_ids'], dtype=torch.long),
            'chosen_attention_mask': torch.tensor(item['chosen_attention_mask'], dtype=torch.long),
            'rejected_input_ids': torch.tensor(item['rejected_input_ids'], dtype=torch.long),
            'rejected_attention_mask': torch.tensor(item['rejected_attention_mask'], dtype=torch.long)
        }
    
    def _process_ppo(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """PPO processing"""
        return {
            'query_input_ids': torch.tensor(item['query_input_ids'], dtype=torch.long),
            'query_attention_mask': torch.tensor(item['query_attention_mask'], dtype=torch.long),
            'response_input_ids': torch.tensor(item['response_input_ids'], dtype=torch.long),
            'response_attention_mask': torch.tensor(item['response_attention_mask'], dtype=torch.long),
            'rewards': torch.tensor(item['rewards'], dtype=torch.float)
        }
    
    def _process_instruction(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Instruction processing"""
        # Similar to instruction dataset processing
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }