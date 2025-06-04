"""Reward model implementation for RLHF training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    AutoModel,
    AutoTokenizer,
    AutoConfig
)


@dataclass
class RewardModelConfig:
    """Configuration for reward model"""
    model_name_or_path: str
    num_labels: int = 1
    dropout_rate: float = 0.1
    pooling_type: str = "last_token"  # last_token, mean, max, cls
    normalize_rewards: bool = True
    reward_scale: float = 1.0
    use_bias: bool = True
    hidden_size: Optional[int] = None
    activation: str = "tanh"  # tanh, relu, gelu
    num_hidden_layers: int = 1
    

class RewardModel(nn.Module):
    """Reward model for scoring text sequences
    
    This model takes text sequences and outputs scalar reward scores.
    It's typically used in RLHF training to provide feedback signals.
    """
    
    def __init__(self, 
                 config: RewardModelConfig,
                 base_model: Optional[PreTrainedModel] = None):
        super().__init__()
        
        self.config = config
        
        # Load base model
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = AutoModel.from_pretrained(config.model_name_or_path)
        
        # Get hidden size
        if config.hidden_size is None:
            self.hidden_size = self.base_model.config.hidden_size
        else:
            self.hidden_size = config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Reward head
        if config.num_hidden_layers == 1:
            self.reward_head = nn.Linear(
                self.hidden_size, 
                config.num_labels, 
                bias=config.use_bias
            )
        else:
            # Multi-layer reward head
            layers = []
            input_size = self.hidden_size
            
            for i in range(config.num_hidden_layers - 1):
                layers.append(nn.Linear(input_size, self.hidden_size, bias=config.use_bias))
                
                # Activation function
                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "gelu":
                    layers.append(nn.GELU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())
                
                layers.append(nn.Dropout(config.dropout_rate))
                input_size = self.hidden_size
            
            # Final layer
            layers.append(nn.Linear(input_size, config.num_labels, bias=config.use_bias))
            
            self.reward_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights"""
        if isinstance(self.reward_head, nn.Linear):
            nn.init.normal_(self.reward_head.weight, std=0.02)
            if self.reward_head.bias is not None:
                nn.init.zeros_(self.reward_head.bias)
        elif isinstance(self.reward_head, nn.Sequential):
            for module in self.reward_head:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of reward model
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            token_type_ids: Token type IDs (optional)
            return_dict: Whether to return a dictionary
            
        Returns:
            Reward scores or dictionary with rewards and hidden states
        """
        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Pool hidden states
        pooled_output = self._pool_hidden_states(hidden_states, attention_mask)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Compute rewards
        rewards = self.reward_head(pooled_output)  # (batch_size, num_labels)
        
        # Apply scaling and normalization
        if self.config.reward_scale != 1.0:
            rewards = rewards * self.config.reward_scale
        
        if self.config.normalize_rewards:
            rewards = torch.tanh(rewards)
        
        if return_dict:
            return {
                'rewards': rewards,
                'hidden_states': hidden_states,
                'pooled_output': pooled_output
            }
        else:
            return rewards
    
    def _pool_hidden_states(self,
                           hidden_states: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool hidden states according to pooling type
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Pooled output of shape (batch_size, hidden_size)
        """
        if self.config.pooling_type == "last_token":
            # Use the last non-padding token
            if attention_mask is not None:
                # Find the last non-padding token for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled_output = hidden_states[batch_indices, seq_lengths]
            else:
                # Use the last token
                pooled_output = hidden_states[:, -1, :]
        
        elif self.config.pooling_type == "mean":
            # Mean pooling over non-padding tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                pooled_output = sum_hidden / sum_mask.clamp(min=1e-8)
            else:
                pooled_output = hidden_states.mean(dim=1)
        
        elif self.config.pooling_type == "max":
            # Max pooling over non-padding tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                hidden_states_masked = hidden_states.masked_fill(~mask_expanded.bool(), float('-inf'))
                pooled_output = hidden_states_masked.max(dim=1)[0]
            else:
                pooled_output = hidden_states.max(dim=1)[0]
        
        elif self.config.pooling_type == "cls":
            # Use [CLS] token (first token)
            pooled_output = hidden_states[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling type: {self.config.pooling_type}")
        
        return pooled_output
    
    def compute_reward(self,
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward scores for input sequences
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Reward scores
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs['rewards'].squeeze(-1)  # Remove last dimension if num_labels=1
    
    def compute_pairwise_rewards(self,
                                chosen_input_ids: torch.Tensor,
                                rejected_input_ids: torch.Tensor,
                                chosen_attention_mask: Optional[torch.Tensor] = None,
                                rejected_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rewards for chosen and rejected sequences
        
        Args:
            chosen_input_ids: Chosen sequence token IDs
            rejected_input_ids: Rejected sequence token IDs
            chosen_attention_mask: Chosen sequence attention mask
            rejected_attention_mask: Rejected sequence attention mask
            
        Returns:
            Tuple of (chosen_rewards, rejected_rewards)
        """
        chosen_rewards = self.compute_reward(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.compute_reward(rejected_input_ids, rejected_attention_mask)
        
        return chosen_rewards, rejected_rewards
    
    def compute_preference_loss(self,
                               chosen_input_ids: torch.Tensor,
                               rejected_input_ids: torch.Tensor,
                               chosen_attention_mask: Optional[torch.Tensor] = None,
                               rejected_attention_mask: Optional[torch.Tensor] = None,
                               margin: float = 0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute preference learning loss
        
        Args:
            chosen_input_ids: Chosen sequence token IDs
            rejected_input_ids: Rejected sequence token IDs
            chosen_attention_mask: Chosen sequence attention mask
            rejected_attention_mask: Rejected sequence attention mask
            margin: Margin for ranking loss
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute rewards
        chosen_rewards, rejected_rewards = self.compute_pairwise_rewards(
            chosen_input_ids, rejected_input_ids,
            chosen_attention_mask, rejected_attention_mask
        )
        
        # Compute ranking loss
        if margin > 0:
            # Margin ranking loss
            loss = F.margin_ranking_loss(
                chosen_rewards, rejected_rewards,
                torch.ones_like(chosen_rewards),
                margin=margin
            )
        else:
            # Log-sigmoid loss (Bradley-Terry model)
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            'reward_loss': loss.item(),
            'reward_accuracy': accuracy.item(),
            'reward_margin': reward_margin.item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'chosen_reward_std': chosen_rewards.std().item(),
            'rejected_reward_std': rejected_rewards.std().item()
        }
        
        return loss, metrics
    
    def freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def freeze_reward_head(self):
        """Freeze reward head parameters"""
        for param in self.reward_head.parameters():
            param.requires_grad = False
    
    def unfreeze_reward_head(self):
        """Unfreeze reward head parameters"""
        for param in self.reward_head.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        head_params = sum(p.numel() for p in self.reward_head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'base_model': base_params,
            'reward_head': head_params,
            'total': total_params,
            'trainable': trainable_params
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save config
        config_dict = {
            'model_name_or_path': self.config.model_name_or_path,
            'num_labels': self.config.num_labels,
            'dropout_rate': self.config.dropout_rate,
            'pooling_type': self.config.pooling_type,
            'normalize_rewards': self.config.normalize_rewards,
            'reward_scale': self.config.reward_scale,
            'use_bias': self.config.use_bias,
            'hidden_size': self.config.hidden_size,
            'activation': self.config.activation,
            'num_hidden_layers': self.config.num_hidden_layers
        }
        
        with open(os.path.join(save_directory, 'reward_model_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_directory: str, base_model: Optional[PreTrainedModel] = None):
        """Load model from directory"""
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_directory, 'reward_model_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = RewardModelConfig(**config_dict)
        
        # Create model
        model = cls(config, base_model=base_model)
        
        # Load state dict
        state_dict_path = os.path.join(model_directory, 'pytorch_model.bin')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model


def create_reward_model(model_name_or_path: str,
                       config: Optional[RewardModelConfig] = None,
                       use_peft: bool = False,
                       peft_config: Optional[Dict[str, Any]] = None,
                       quantization_config: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Tuple[RewardModel, PreTrainedTokenizer]:
    """Create reward model and tokenizer with optional PEFT support
    
    Args:
        model_name_or_path: Model name or path
        config: Reward model configuration
        use_peft: Whether to use PEFT (LoRA)
        peft_config: PEFT configuration dict
        quantization_config: Quantization configuration for QLoRA
        **kwargs: Additional arguments for RewardModelConfig
        
    Returns:
        Tuple of (reward_model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model loading arguments
    model_kwargs = {}
    
    # Add quantization config for QLoRA
    if use_peft and quantization_config:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization_config.get('load_in_4bit', True),
                bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_compute_dtype=getattr(torch, quantization_config.get('bnb_4bit_compute_dtype', 'float16')),
                bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True)
            )
            model_kwargs['quantization_config'] = bnb_config
            print(f"QLoRA enabled for reward model: {quantization_config}")
        except ImportError:
            print("Warning: bitsandbytes not available, falling back to regular LoRA")
    
    # Load base model
    base_model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    
    # Apply PEFT if requested
    if use_peft and peft_config:
        try:
            from peft import LoraConfig, get_peft_model
            
            # Create LoRA config
            lora_config = LoraConfig(
                r=peft_config.get('r', 16),
                lora_alpha=peft_config.get('lora_alpha', 32),
                target_modules=peft_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
                lora_dropout=peft_config.get('lora_dropout', 0.1),
                bias=peft_config.get('bias', 'none'),
                task_type="FEATURE_EXTRACTION"  # For reward model
            )
            
            # Apply LoRA to base model
            base_model = get_peft_model(base_model, lora_config)
            
            # Print LoRA info
            print(f"LoRA applied to reward model:")
            print(f"  - Rank (r): {lora_config.r}")
            print(f"  - Alpha: {lora_config.lora_alpha}")
            print(f"  - Target modules: {lora_config.target_modules}")
            print(f"  - Dropout: {lora_config.lora_dropout}")
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in base_model.parameters())
            print(f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
        except ImportError:
            print("Warning: PEFT library not available, using full fine-tuning for reward model")
    
    # Create config if not provided
    if config is None:
        config_kwargs = {'model_name_or_path': model_name_or_path}
        config_kwargs.update(kwargs)
        config = RewardModelConfig(**config_kwargs)
    
    # Create reward model with the (potentially PEFT-enabled) base model
    reward_model = RewardModel(config, base_model=base_model)
    
    return reward_model, tokenizer