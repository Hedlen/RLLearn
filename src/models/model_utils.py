"""Model utilities for loading, creating, and managing models"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List, Type
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from .reward_model import RewardModel, RewardModelConfig, create_reward_model
from .value_model import ValueModel, ValueModelConfig, create_value_model
from .policy_model import PolicyModel, PolicyModelConfig, create_policy_model


def load_model_and_tokenizer(model_name_or_path: str,
                             model_type: str = "causal_lm",
                             trust_remote_code: bool = False,
                             torch_dtype: Optional[torch.dtype] = None,
                             device_map: Optional[Union[str, Dict]] = None,
                             **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from HuggingFace or local path
    
    Args:
        model_name_or_path: Model name or path
        model_type: Type of model ('causal_lm', 'base', etc.)
        trust_remote_code: Whether to trust remote code
        torch_dtype: PyTorch data type
        device_map: Device mapping for model
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model based on type
    model_kwargs = {
        'trust_remote_code': trust_remote_code,
        **kwargs
    }
    
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    
    if device_map is not None:
        model_kwargs['device_map'] = device_map
    
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
    elif model_type == "base":
        model = AutoModel.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, tokenizer


def create_model_from_config(config: Dict[str, Any],
                            model_class: Type[nn.Module]) -> nn.Module:
    """Create model from configuration dictionary
    
    Args:
        config: Configuration dictionary
        model_class: Model class to instantiate
        
    Returns:
        Instantiated model
    """
    if model_class == RewardModel:
        model_config = RewardModelConfig(**config)
        model, _ = create_reward_model(
            model_config.model_name_or_path,
            config=model_config
        )
    elif model_class == ValueModel:
        model_config = ValueModelConfig(**config)
        model, _ = create_value_model(
            model_config.model_name_or_path,
            config=model_config
        )
    elif model_class == PolicyModel:
        model_config = PolicyModelConfig(**config)
        model, _ = create_policy_model(
            model_config.model_name_or_path,
            config=model_config
        )
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    return model


def freeze_model_parameters(model: nn.Module,
                           freeze_embeddings: bool = True,
                           freeze_layers: Optional[List[int]] = None,
                           freeze_layer_norm: bool = False,
                           freeze_head: bool = False):
    """Freeze model parameters selectively
    
    Args:
        model: Model to freeze
        freeze_embeddings: Whether to freeze embedding layers
        freeze_layers: List of layer indices to freeze
        freeze_layer_norm: Whether to freeze layer normalization
        freeze_head: Whether to freeze model head
    """
    # Freeze embeddings
    if freeze_embeddings:
        if hasattr(model, 'embeddings'):
            for param in model.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(model, 'embed_tokens'):
            for param in model.embed_tokens.parameters():
                param.requires_grad = False
        elif hasattr(model, 'wte'):
            for param in model.wte.parameters():
                param.requires_grad = False
    
    # Freeze specific layers
    if freeze_layers is not None:
        if hasattr(model, 'transformer'):
            # GPT-style models
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # LLaMA-style models
            layers = model.model.layers
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style models
            layers = model.encoder.layer
        else:
            layers = None
        
        if layers is not None:
            for idx in freeze_layers:
                if 0 <= idx < len(layers):
                    for param in layers[idx].parameters():
                        param.requires_grad = False
    
    # Freeze layer normalization
    if freeze_layer_norm:
        for name, module in model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = False
    
    # Freeze head
    if freeze_head:
        if hasattr(model, 'lm_head'):
            for param in model.lm_head.parameters():
                param.requires_grad = False
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = False


def unfreeze_model_parameters(model: nn.Module):
    """Unfreeze all model parameters
    
    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get trainable parameters from model
    
    Args:
        model: Model to analyze
        
    Returns:
        List of trainable parameters
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
    }


def move_model_to_device(model: nn.Module,
                        device: Union[str, torch.device],
                        dtype: Optional[torch.dtype] = None) -> nn.Module:
    """Move model to specified device and dtype
    
    Args:
        model: Model to move
        device: Target device
        dtype: Target dtype
        
    Returns:
        Model on target device
    """
    model = model.to(device)
    
    if dtype is not None:
        model = model.to(dtype)
    
    return model


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of model parameters
    
    Args:
        model: Model to check
        
    Returns:
        Device of model parameters
    """
    return next(model.parameters()).device


def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Get the dtype of model parameters
    
    Args:
        model: Model to check
        
    Returns:
        Dtype of model parameters
    """
    return next(model.parameters()).dtype


def save_model_checkpoint(model: nn.Module,
                         optimizer: Optional[torch.optim.Optimizer],
                         scheduler: Optional[Any],
                         epoch: int,
                         step: int,
                         loss: float,
                         metrics: Dict[str, float],
                         save_path: str,
                         save_optimizer: bool = True,
                         save_scheduler: bool = True,
                         additional_info: Optional[Dict[str, Any]] = None):
    """Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        metrics: Training metrics
        save_path: Path to save checkpoint
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        additional_info: Additional information to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'metrics': metrics
    }
    
    if optimizer is not None and save_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None and save_scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)


def load_model_checkpoint(model: nn.Module,
                         checkpoint_path: str,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         scheduler: Optional[Any] = None,
                         device: Optional[Union[str, torch.device]] = None,
                         strict: bool = True) -> Dict[str, Any]:
    """Load model checkpoint
    
    Args:
        model: Model to load state into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Checkpoint information
    """
    if device is None:
        device = get_model_device(model)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metrics': checkpoint.get('metrics', {})
    }


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """Get model memory usage in MB
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / (1024 * 1024),
        'buffers_mb': buffer_size / (1024 * 1024),
        'total_mb': total_size / (1024 * 1024)
    }


def print_model_summary(model: nn.Module,
                       input_size: Optional[Tuple[int, ...]] = None,
                       show_weights: bool = False,
                       show_parameters: bool = True):
    """Print model summary
    
    Args:
        model: Model to summarize
        input_size: Input size for forward pass
        show_weights: Whether to show weight details
        show_parameters: Whether to show parameter counts
    """
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)
    
    if show_parameters:
        param_counts = count_parameters(model)
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Frozen parameters: {param_counts['frozen']:,}")
        print(f"Trainable percentage: {param_counts['trainable_percentage']:.2f}%")
        print("-" * 80)
    
    memory_usage = get_model_memory_usage(model)
    print(f"Memory usage: {memory_usage['total_mb']:.2f} MB")
    print(f"  Parameters: {memory_usage['parameters_mb']:.2f} MB")
    print(f"  Buffers: {memory_usage['buffers_mb']:.2f} MB")
    print("-" * 80)
    
    device = get_model_device(model)
    dtype = get_model_dtype(model)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("-" * 80)
    
    if show_weights:
        print("Model architecture:")
        print(model)
        print("-" * 80)
    
    print("=" * 80)


def compare_models(model1: nn.Module,
                  model2: nn.Module,
                  name1: str = "Model 1",
                  name2: str = "Model 2") -> Dict[str, Any]:
    """Compare two models
    
    Args:
        model1: First model
        model2: Second model
        name1: Name for first model
        name2: Name for second model
        
    Returns:
        Comparison results
    """
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    memory1 = get_model_memory_usage(model1)
    memory2 = get_model_memory_usage(model2)
    
    comparison = {
        'models': {
            name1: {
                'parameters': params1,
                'memory_mb': memory1['total_mb'],
                'device': str(get_model_device(model1)),
                'dtype': str(get_model_dtype(model1))
            },
            name2: {
                'parameters': params2,
                'memory_mb': memory2['total_mb'],
                'device': str(get_model_device(model2)),
                'dtype': str(get_model_dtype(model2))
            }
        },
        'differences': {
            'parameter_diff': params2['total'] - params1['total'],
            'memory_diff_mb': memory2['total_mb'] - memory1['total_mb'],
            'trainable_diff': params2['trainable'] - params1['trainable']
        }
    }
    
    return comparison


def validate_model_compatibility(model: nn.Module,
                               tokenizer: PreTrainedTokenizer,
                               sample_input: Optional[torch.Tensor] = None) -> Dict[str, bool]:
    """Validate model and tokenizer compatibility
    
    Args:
        model: Model to validate
        tokenizer: Tokenizer to validate
        sample_input: Sample input for testing
        
    Returns:
        Validation results
    """
    results = {
        'vocab_size_match': False,
        'pad_token_set': False,
        'eos_token_set': False,
        'forward_pass_success': False
    }
    
    # Check vocabulary size
    if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
        results['vocab_size_match'] = model.config.vocab_size == len(tokenizer)
    
    # Check special tokens
    results['pad_token_set'] = tokenizer.pad_token is not None
    results['eos_token_set'] = tokenizer.eos_token is not None
    
    # Test forward pass
    if sample_input is None:
        # Create sample input
        sample_text = "Hello, this is a test."
        sample_input = tokenizer(
            sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )['input_ids']
    
    try:
        with torch.no_grad():
            model.eval()
            device = get_model_device(model)
            sample_input = sample_input.to(device)
            _ = model(sample_input)
        results['forward_pass_success'] = True
    except Exception as e:
        print(f"Forward pass failed: {e}")
        results['forward_pass_success'] = False
    
    return results