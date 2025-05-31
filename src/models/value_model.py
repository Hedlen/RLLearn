"""Value model implementation for reinforcement learning"""

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
class ValueModelConfig:
    """Configuration for value model"""
    model_name_or_path: str
    dropout_rate: float = 0.1
    pooling_type: str = "last_token"  # last_token, mean, max, cls
    use_bias: bool = True
    hidden_size: Optional[int] = None
    activation: str = "tanh"  # tanh, relu, gelu
    num_hidden_layers: int = 1
    value_head_init_std: float = 0.02
    

class ValueModel(nn.Module):
    """Value model for estimating state values in RL
    
    This model takes text sequences and outputs scalar value estimates.
    It's used in actor-critic methods like PPO to estimate the value function.
    """
    
    def __init__(self, 
                 config: ValueModelConfig,
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
        
        # Value head
        if config.num_hidden_layers == 1:
            self.value_head = nn.Linear(
                self.hidden_size, 
                1,  # Single value output
                bias=config.use_bias
            )
        else:
            # Multi-layer value head
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
            layers.append(nn.Linear(input_size, 1, bias=config.use_bias))
            
            self.value_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize value head weights"""
        if isinstance(self.value_head, nn.Linear):
            nn.init.normal_(self.value_head.weight, std=self.config.value_head_init_std)
            if self.value_head.bias is not None:
                nn.init.zeros_(self.value_head.bias)
        elif isinstance(self.value_head, nn.Sequential):
            for module in self.value_head:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.config.value_head_init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of value model
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            token_type_ids: Token type IDs (optional)
            return_dict: Whether to return a dictionary
            
        Returns:
            Value estimates or dictionary with values and hidden states
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
        
        # Compute values
        values = self.value_head(pooled_output).squeeze(-1)  # (batch_size,)
        
        if return_dict:
            return {
                'values': values,
                'hidden_states': hidden_states,
                'pooled_output': pooled_output
            }
        else:
            return values
    
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
    
    def compute_values(self,
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute value estimates for input sequences
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Value estimates
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs['values']
    
    def compute_value_loss(self,
                          input_ids: torch.Tensor,
                          returns: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None,
                          old_values: Optional[torch.Tensor] = None,
                          clip_range: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute value function loss
        
        Args:
            input_ids: Input token IDs
            returns: Target returns
            attention_mask: Attention mask
            old_values: Old value estimates for clipping
            clip_range: Value clipping range
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute current values
        current_values = self.compute_values(input_ids, attention_mask)
        
        # Compute loss
        if old_values is not None and clip_range is not None:
            # Clipped value loss (PPO-style)
            value_clipped = old_values + torch.clamp(
                current_values - old_values, -clip_range, clip_range
            )
            loss1 = F.mse_loss(current_values, returns)
            loss2 = F.mse_loss(value_clipped, returns)
            loss = torch.max(loss1, loss2)
        else:
            # Standard MSE loss
            loss = F.mse_loss(current_values, returns)
        
        # Compute metrics
        with torch.no_grad():
            explained_variance = self._compute_explained_variance(current_values, returns)
            value_error = torch.abs(current_values - returns).mean()
        
        metrics = {
            'value_loss': loss.item(),
            'explained_variance': explained_variance,
            'value_error': value_error.item(),
            'value_mean': current_values.mean().item(),
            'value_std': current_values.std().item(),
            'returns_mean': returns.mean().item(),
            'returns_std': returns.std().item()
        }
        
        return loss, metrics
    
    def _compute_explained_variance(self,
                                   values: torch.Tensor,
                                   returns: torch.Tensor) -> float:
        """Compute explained variance of value function
        
        Args:
            values: Value function estimates
            returns: Target returns
            
        Returns:
            Explained variance (1.0 is perfect, 0.0 is no better than mean)
        """
        var_returns = torch.var(returns)
        if var_returns == 0:
            return 0.0
        
        explained_var = 1.0 - torch.var(returns - values) / var_returns
        return explained_var.item()
    
    def freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def freeze_value_head(self):
        """Freeze value head parameters"""
        for param in self.value_head.parameters():
            param.requires_grad = False
    
    def unfreeze_value_head(self):
        """Unfreeze value head parameters"""
        for param in self.value_head.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        head_params = sum(p.numel() for p in self.value_head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'base_model': base_params,
            'value_head': head_params,
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
            'dropout_rate': self.config.dropout_rate,
            'pooling_type': self.config.pooling_type,
            'use_bias': self.config.use_bias,
            'hidden_size': self.config.hidden_size,
            'activation': self.config.activation,
            'num_hidden_layers': self.config.num_hidden_layers,
            'value_head_init_std': self.config.value_head_init_std
        }
        
        with open(os.path.join(save_directory, 'value_model_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_directory: str, base_model: Optional[PreTrainedModel] = None):
        """Load model from directory"""
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_directory, 'value_model_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ValueModelConfig(**config_dict)
        
        # Create model
        model = cls(config, base_model=base_model)
        
        # Load state dict
        state_dict_path = os.path.join(model_directory, 'pytorch_model.bin')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model


def create_value_model(model_name_or_path: str,
                      config: Optional[ValueModelConfig] = None,
                      **kwargs) -> Tuple[ValueModel, PreTrainedTokenizer]:
    """Create value model and tokenizer
    
    Args:
        model_name_or_path: Model name or path
        config: Value model configuration
        **kwargs: Additional arguments for ValueModelConfig
        
    Returns:
        Tuple of (value_model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create config if not provided
    if config is None:
        config_kwargs = {'model_name_or_path': model_name_or_path}
        config_kwargs.update(kwargs)
        config = ValueModelConfig(**config_kwargs)
    
    # Create value model
    value_model = ValueModel(config)
    
    return value_model, tokenizer