"""Policy model implementation for reinforcement learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)


@dataclass
class PolicyModelConfig:
    """Configuration for policy model"""
    model_name_or_path: str
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True
    max_new_tokens: int = 1024
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    use_cache: bool = True
    

class PolicyModel(nn.Module):
    """Policy model for generating text sequences in RL
    
    This model is used as the policy in reinforcement learning,
    generating text sequences and computing action probabilities.
    """
    
    def __init__(self, 
                 config: PolicyModelConfig,
                 base_model: Optional[PreTrainedModel] = None):
        super().__init__()
        
        self.config = config
        
        # Load base model
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        
        # Generation config
        self.generation_config = GenerationConfig(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
            max_new_tokens=config.max_new_tokens,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            use_cache=config.use_cache
        )
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of policy model
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            labels: Labels for computing loss (optional)
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs (logits, loss, etc.)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict
        )
        
        return outputs
    
    def generate(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                generation_config: Optional[GenerationConfig] = None,
                **kwargs) -> torch.Tensor:
        """Generate text sequences
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            generation_config: Generation configuration
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        if generation_config is None:
            generation_config = self.generation_config
        
        # Update generation config with kwargs
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)
        
        with torch.no_grad():
            generated = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs
            )
        
        return generated
    
    def compute_log_probs(self,
                         input_ids: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log probabilities for sequences
        
        Args:
            input_ids: Full input token IDs (prompt + response)
            attention_mask: Attention mask
            target_ids: Target token IDs (response only)
            
        Returns:
            Log probabilities for target sequences
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        if target_ids is not None:
            # Compute log probs for specific target sequences
            # Assume target_ids are the response part
            prompt_length = input_ids.shape[1] - target_ids.shape[1]
            response_logits = logits[:, prompt_length-1:-1, :]  # Shift for next token prediction
            
            # Compute log probabilities
            log_probs = F.log_softmax(response_logits, dim=-1)
            
            # Gather log probabilities for actual tokens
            gathered_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask out padding tokens
            if attention_mask is not None:
                response_mask = attention_mask[:, prompt_length:].float()
            else:
                response_mask = (target_ids != self.generation_config.pad_token_id).float()
            
            masked_log_probs = gathered_log_probs * response_mask
            
            # Sum log probabilities over sequence length
            sequence_log_probs = masked_log_probs.sum(dim=-1)
            
            return sequence_log_probs
        else:
            # Return log probs for all tokens
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs
    
    def compute_action_log_probs(self,
                                states: torch.Tensor,
                                actions: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log probabilities of actions given states
        
        Args:
            states: State token IDs (prompts)
            actions: Action token IDs (responses)
            attention_mask: Attention mask
            
        Returns:
            Log probabilities of actions
        """
        # Combine states and actions
        full_input_ids = torch.cat([states, actions], dim=1)
        
        if attention_mask is not None:
            # Extend attention mask
            action_mask = (actions != self.generation_config.pad_token_id).long()
            full_attention_mask = torch.cat([attention_mask, action_mask], dim=1)
        else:
            full_attention_mask = None
        
        return self.compute_log_probs(
            full_input_ids, full_attention_mask, actions
        )
    
    def sample_actions(self,
                      states: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      max_new_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      top_k: Optional[int] = None,
                      top_p: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions (responses) given states (prompts)
        
        Args:
            states: State token IDs (prompts)
            attention_mask: Attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Tuple of (actions, log_probs)
        """
        generation_kwargs = {}
        if max_new_tokens is not None:
            generation_kwargs['max_new_tokens'] = max_new_tokens
        if temperature is not None:
            generation_kwargs['temperature'] = temperature
        if top_k is not None:
            generation_kwargs['top_k'] = top_k
        if top_p is not None:
            generation_kwargs['top_p'] = top_p
        
        # Generate responses
        generated = self.generate(
            states, 
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        # Extract actions (responses)
        prompt_length = states.shape[1]
        actions = generated[:, prompt_length:]
        
        # Compute log probabilities
        log_probs = self.compute_log_probs(generated, None, actions)
        
        return actions, log_probs
    
    def compute_policy_loss(self,
                           states: torch.Tensor,
                           actions: torch.Tensor,
                           advantages: torch.Tensor,
                           old_log_probs: Optional[torch.Tensor] = None,
                           clip_range: Optional[float] = None,
                           attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute policy loss
        
        Args:
            states: State token IDs
            actions: Action token IDs
            advantages: Advantage estimates
            old_log_probs: Old log probabilities for PPO clipping
            clip_range: PPO clipping range
            attention_mask: Attention mask
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute current log probabilities
        current_log_probs = self.compute_action_log_probs(
            states, actions, attention_mask
        )
        
        if old_log_probs is not None and clip_range is not None:
            # PPO clipped loss
            ratio = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            loss = -torch.min(surr1, surr2).mean()
            
            # Compute metrics
            with torch.no_grad():
                clipfrac = ((ratio < (1 - clip_range)) | (ratio > (1 + clip_range))).float().mean()
                kl_div = (old_log_probs - current_log_probs).mean()
        else:
            # Standard policy gradient
            loss = -(current_log_probs * advantages).mean()
            clipfrac = 0.0
            kl_div = 0.0
        
        metrics = {
            'policy_loss': loss.item(),
            'log_probs_mean': current_log_probs.mean().item(),
            'log_probs_std': current_log_probs.std().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'clipfrac': clipfrac if isinstance(clipfrac, float) else clipfrac.item(),
            'kl_divergence': kl_div if isinstance(kl_div, float) else kl_div.item()
        }
        
        return loss, metrics
    
    def compute_entropy(self,
                       input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy of the policy distribution
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Entropy value
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask[:, 1:].float()  # Shift for next token prediction
            entropy = entropy[:, :-1] * mask
            return entropy.sum() / mask.sum()
        else:
            return entropy.mean()
    
    def freeze_model(self):
        """Freeze all model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_model(self):
        """Unfreeze all model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def freeze_layers(self, layer_indices: List[int]):
        """Freeze specific layers
        
        Args:
            layer_indices: List of layer indices to freeze
        """
        if hasattr(self.base_model, 'transformer'):
            # GPT-style models
            layers = self.base_model.transformer.h
        elif hasattr(self.base_model, 'model'):
            # LLaMA-style models
            layers = self.base_model.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        for idx in layer_indices:
            if 0 <= idx < len(layers):
                for param in layers[idx].parameters():
                    param.requires_grad = False
    
    def unfreeze_layers(self, layer_indices: List[int]):
        """Unfreeze specific layers
        
        Args:
            layer_indices: List of layer indices to unfreeze
        """
        if hasattr(self.base_model, 'transformer'):
            # GPT-style models
            layers = self.base_model.transformer.h
        elif hasattr(self.base_model, 'model'):
            # LLaMA-style models
            layers = self.base_model.model.layers
        else:
            raise ValueError("Unknown model architecture")
        
        for idx in layer_indices:
            if 0 <= idx < len(layers):
                for param in layers[idx].parameters():
                    param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get trainable parameters"""
        return [p for p in self.base_model.parameters() if p.requires_grad]
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        self.base_model.save_pretrained(save_directory)
        
        # Save policy config
        import os
        import json
        
        config_dict = {
            'model_name_or_path': self.config.model_name_or_path,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
            'do_sample': self.config.do_sample,
            'max_new_tokens': self.config.max_new_tokens,
            'pad_token_id': self.config.pad_token_id,
            'eos_token_id': self.config.eos_token_id,
            'repetition_penalty': self.config.repetition_penalty,
            'length_penalty': self.config.length_penalty,
            'use_cache': self.config.use_cache
        }
        
        with open(os.path.join(save_directory, 'policy_model_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """Load model from directory"""
        import os
        import json
        
        # Load policy config
        config_path = os.path.join(model_directory, 'policy_model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = PolicyModelConfig(**config_dict)
        else:
            # Fallback to default config
            config = PolicyModelConfig(model_name_or_path=model_directory)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_directory)
        
        # Create policy model
        model = cls(config, base_model=base_model)
        
        return model


def create_policy_model(model_name_or_path: str,
                       config: Optional[PolicyModelConfig] = None,
                       **kwargs) -> Tuple[PolicyModel, PreTrainedTokenizer]:
    """Create policy model and tokenizer
    
    Args:
        model_name_or_path: Model name or path
        config: Policy model configuration
        **kwargs: Additional arguments for PolicyModelConfig
        
    Returns:
        Tuple of (policy_model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create config if not provided
    if config is None:
        config_kwargs = {
            'model_name_or_path': model_name_or_path,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
        config_kwargs.update(kwargs)
        config = PolicyModelConfig(**config_kwargs)
    
    # Create policy model
    policy_model = PolicyModel(config)
    
    return policy_model, tokenizer