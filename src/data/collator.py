"""Data collators for RL training"""

import torch
from typing import Dict, List, Any, Optional, Union
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class DataCollatorForRL:
    """Base data collator for RL training"""
    
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate features into a batch"""
        return self._collate_features(features)
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Base collation method - to be overridden by subclasses"""
        raise NotImplementedError
    
    def _pad_sequence(self, sequences: List[torch.Tensor], 
                     padding_value: int = 0) -> torch.Tensor:
        """Pad sequences to the same length"""
        # Convert lists to tensors if necessary
        tensor_sequences = []
        for seq in sequences:
            if isinstance(seq, list):
                seq = torch.tensor(seq, dtype=torch.long)
            tensor_sequences.append(seq)
        
        max_len = max(len(seq) for seq in tensor_sequences)
        
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // 
                      self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_sequences = []
        for seq in tensor_sequences:
            if len(seq) > max_len:
                seq = seq[:max_len]
            
            padding_length = max_len - len(seq)
            if padding_length > 0:
                padding = torch.full((padding_length,), padding_value, dtype=seq.dtype)
                seq = torch.cat([seq, padding])
            
            padded_sequences.append(seq)
        
        return torch.stack(padded_sequences)


@dataclass
class DataCollatorForSFT(DataCollatorForRL):
    """Data collator for Supervised Fine-Tuning"""
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate SFT features"""
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad sequences
        input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        attention_mask = self._pad_sequence(attention_mask, 0)
        labels = self._pad_sequence(labels, -100)  # -100 is ignored in loss computation
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


@dataclass
class DataCollatorForPreference(DataCollatorForRL):
    """Data collator for preference learning (DPO, etc.)"""
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate preference features"""
        # Extract sequences
        prompt_input_ids = [f['prompt_input_ids'] for f in features]
        prompt_attention_mask = [f['prompt_attention_mask'] for f in features]
        chosen_input_ids = [f['chosen_input_ids'] for f in features]
        chosen_attention_mask = [f['chosen_attention_mask'] for f in features]
        rejected_input_ids = [f['rejected_input_ids'] for f in features]
        rejected_attention_mask = [f['rejected_attention_mask'] for f in features]
        
        # Pad sequences
        prompt_input_ids = self._pad_sequence(prompt_input_ids, self.tokenizer.pad_token_id)
        prompt_attention_mask = self._pad_sequence(prompt_attention_mask, 0)
        chosen_input_ids = self._pad_sequence(chosen_input_ids, self.tokenizer.pad_token_id)
        chosen_attention_mask = self._pad_sequence(chosen_attention_mask, 0)
        rejected_input_ids = self._pad_sequence(rejected_input_ids, self.tokenizer.pad_token_id)
        rejected_attention_mask = self._pad_sequence(rejected_attention_mask, 0)
        
        return {
            'prompt_input_ids': prompt_input_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'chosen_input_ids': chosen_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected_input_ids,
            'rejected_attention_mask': rejected_attention_mask
        }


@dataclass
class DataCollatorForPPO(DataCollatorForRL):
    """Data collator for PPO training"""
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate PPO features"""
        query_input_ids = [f['query_input_ids'] for f in features]
        query_attention_mask = [f['query_attention_mask'] for f in features]
        response_input_ids = [f['response_input_ids'] for f in features]
        response_attention_mask = [f['response_attention_mask'] for f in features]
        rewards = [f['rewards'] for f in features]
        
        # Pad sequences
        query_input_ids = self._pad_sequence(query_input_ids, self.tokenizer.pad_token_id)
        query_attention_mask = self._pad_sequence(query_attention_mask, 0)
        response_input_ids = self._pad_sequence(response_input_ids, self.tokenizer.pad_token_id)
        response_attention_mask = self._pad_sequence(response_attention_mask, 0)
        
        # Stack rewards
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.tensor(rewards)
        
        return {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'response_input_ids': response_input_ids,
            'response_attention_mask': response_attention_mask,
            'rewards': rewards
        }


@dataclass
class DataCollatorForRLHF(DataCollatorForRL):
    """Data collator for RLHF training"""
    
    include_rewards: bool = True
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate RLHF features"""
        query_input_ids = [f['query_input_ids'] for f in features]
        query_attention_mask = [f['query_attention_mask'] for f in features]
        response_input_ids = [f['response_input_ids'] for f in features]
        response_attention_mask = [f['response_attention_mask'] for f in features]
        
        # Pad sequences
        query_input_ids = self._pad_sequence(query_input_ids, self.tokenizer.pad_token_id)
        query_attention_mask = self._pad_sequence(query_attention_mask, 0)
        response_input_ids = self._pad_sequence(response_input_ids, self.tokenizer.pad_token_id)
        response_attention_mask = self._pad_sequence(response_attention_mask, 0)
        
        result = {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'response_input_ids': response_input_ids,
            'response_attention_mask': response_attention_mask
        }
        
        # Add rewards if available
        if self.include_rewards and 'rewards' in features[0]:
            rewards = [f['rewards'] for f in features]
            rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.tensor(rewards)
            result['rewards'] = rewards
        
        return result


@dataclass
class DataCollatorForRewardModel(DataCollatorForRL):
    """Data collator for reward model training"""
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate reward model features"""
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad sequences
        input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        attention_mask = self._pad_sequence(attention_mask, 0)
        
        # Stack labels (scores)
        labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


@dataclass
class DataCollatorForConversation(DataCollatorForRL):
    """Data collator for conversation training"""
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate conversation features"""
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad sequences
        input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        attention_mask = self._pad_sequence(attention_mask, 0)
        labels = self._pad_sequence(labels, -100)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


@dataclass
class DataCollatorForInstruction(DataCollatorForRL):
    """Data collator for instruction following training"""
    
    def _collate_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate instruction features"""
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad sequences
        input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        attention_mask = self._pad_sequence(attention_mask, 0)
        labels = self._pad_sequence(labels, -100)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class DynamicDataCollator:
    """Dynamic data collator that can switch between different collation modes"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, mode: str = 'sft', **kwargs):
        self.tokenizer = tokenizer
        self.mode = mode
        self.kwargs = kwargs
        
        self._collators = {
            'sft': DataCollatorForSFT,
            'preference': DataCollatorForPreference,
            'ppo': DataCollatorForPPO,
            'rlhf': DataCollatorForRLHF,
            'reward': DataCollatorForRewardModel,
            'conversation': DataCollatorForConversation,
            'instruction': DataCollatorForInstruction
        }
        
        self._current_collator = self._create_collator(mode)
    
    def _create_collator(self, mode: str):
        """Create collator for given mode"""
        if mode not in self._collators:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self._collators.keys())}")
        
        collator_class = self._collators[mode]
        return collator_class(tokenizer=self.tokenizer, **self.kwargs)
    
    def set_mode(self, mode: str, **kwargs):
        """Change collation mode"""
        self.mode = mode
        self.kwargs.update(kwargs)
        self._current_collator = self._create_collator(mode)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate features using current mode"""
        return self._current_collator(features)


def get_collator(mode: str, tokenizer: PreTrainedTokenizer, **kwargs) -> DataCollatorForRL:
    """Factory function to get appropriate collator"""
    collators = {
        'sft': DataCollatorForSFT,
        'preference': DataCollatorForPreference,
        'ppo': DataCollatorForPPO,
        'rlhf': DataCollatorForRLHF,
        'reward': DataCollatorForRewardModel,
        'conversation': DataCollatorForConversation,
        'instruction': DataCollatorForInstruction
    }
    
    if mode not in collators:
        raise ValueError(f"Unknown collator mode: {mode}. Available: {list(collators.keys())}")
    
    return collators[mode](tokenizer=tokenizer, **kwargs)