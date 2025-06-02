"""Utility functions for trainers"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
import json
from dataclasses import asdict


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration
    
    Args:
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def save_config(config: Any, save_path: str):
    """Save training configuration to file
    
    Args:
        config: Configuration object
        save_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert dataclass to dict if needed
    if hasattr(config, '__dataclass_fields__'):
        config_dict = asdict(config)
    else:
        config_dict = config.__dict__ if hasattr(config, '__dict__') else config
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_advantages(rewards: torch.Tensor,
                      values: torch.Tensor,
                      gamma: float = 0.99,
                      lam: float = 0.95,
                      normalize: bool = True) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: Reward tensor [batch_size, seq_len]
        values: Value tensor [batch_size, seq_len]
        gamma: Discount factor
        lam: GAE lambda parameter
        normalize: Whether to normalize advantages
        
    Returns:
        Advantage tensor
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Compute advantages using GAE
    gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    
    if normalize:
        advantages = whiten(advantages)
    
    return advantages


def whiten(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Whiten (normalize) a tensor
    
    Args:
        tensor: Input tensor
        eps: Small epsilon for numerical stability
        
    Returns:
        Whitened tensor
    """
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + eps)


def compute_ranking_loss(chosen_rewards: torch.Tensor,
                        rejected_rewards: torch.Tensor,
                        margin: float = 0.0,
                        loss_type: str = "hinge") -> torch.Tensor:
    """Compute ranking loss for preference learning
    
    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses
        margin: Margin for ranking loss
        loss_type: Type of ranking loss ("hinge", "log_sigmoid", "cross_entropy")
        
    Returns:
        Ranking loss
    """
    if loss_type == "hinge":
        # Hinge loss: max(0, margin - (chosen_reward - rejected_reward))
        loss = F.relu(margin - (chosen_rewards - rejected_rewards))
    elif loss_type == "log_sigmoid":
        # Log-sigmoid loss: -log(sigmoid(chosen_reward - rejected_reward))
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    elif loss_type == "cross_entropy":
        # Cross-entropy loss with softmax
        logits = torch.stack([rejected_rewards, chosen_rewards], dim=1)
        targets = torch.ones(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, targets)
    else:
        raise ValueError(f"Unknown ranking loss type: {loss_type}")
    
    return loss.mean()


def compute_kl_divergence(log_probs_p: torch.Tensor,
                         log_probs_q: torch.Tensor,
                         reduction: str = "mean") -> torch.Tensor:
    """Compute KL divergence between two distributions
    
    Args:
        log_probs_p: Log probabilities of distribution P
        log_probs_q: Log probabilities of distribution Q
        reduction: Reduction method ("mean", "sum", "none")
        
    Returns:
        KL divergence
    """
    kl_div = log_probs_p - log_probs_q
    
    if reduction == "mean":
        return kl_div.mean()
    elif reduction == "sum":
        return kl_div.sum()
    elif reduction == "none":
        return kl_div
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of a categorical distribution
    
    Args:
        logits: Logits tensor
        dim: Dimension to compute entropy over
        
    Returns:
        Entropy tensor
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with mask
    
    Args:
        tensor: Input tensor
        mask: Boolean mask tensor
        dim: Dimension to compute mean over
        
    Returns:
        Masked mean
    """
    masked_tensor = tensor * mask.float()
    
    if dim is None:
        return masked_tensor.sum() / mask.float().sum().clamp(min=1)
    else:
        return masked_tensor.sum(dim=dim) / mask.float().sum(dim=dim).clamp(min=1)


def masked_var(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with mask
    
    Args:
        tensor: Input tensor
        mask: Boolean mask tensor
        dim: Dimension to compute variance over
        unbiased: Whether to use unbiased estimator
        
    Returns:
        Masked variance
    """
    mean = masked_mean(tensor, mask, dim=dim)
    
    if dim is None:
        diff_squared = ((tensor - mean) ** 2) * mask.float()
        n = mask.float().sum().clamp(min=1)
        if unbiased and n > 1:
            return diff_squared.sum() / (n - 1)
        else:
            return diff_squared.sum() / n
    else:
        if dim < 0:
            dim = tensor.dim() + dim
        
        # Expand mean to match tensor dimensions
        mean_expanded = mean.unsqueeze(dim)
        diff_squared = ((tensor - mean_expanded) ** 2) * mask.float()
        n = mask.float().sum(dim=dim).clamp(min=1)
        
        if unbiased:
            n = n.clamp(min=2)  # Need at least 2 samples for unbiased estimate
            return diff_squared.sum(dim=dim) / (n - 1)
        else:
            return diff_squared.sum(dim=dim) / n


def clip_by_value(tensor: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Clip tensor values to a range
    
    Args:
        tensor: Input tensor
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clipped tensor
    """
    return torch.clamp(tensor, min=min_value, max=max_value)


def exponential_moving_average(current_value: float,
                              new_value: float,
                              decay: float = 0.99) -> float:
    """Compute exponential moving average
    
    Args:
        current_value: Current EMA value
        new_value: New value to incorporate
        decay: Decay factor
        
    Returns:
        Updated EMA value
    """
    return decay * current_value + (1 - decay) * new_value


def get_linear_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                   num_warmup_steps: int,
                                   num_training_steps: int,
                                   last_epoch: int = -1):
    """Create a linear learning rate schedule with warmup
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch number
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                   num_warmup_steps: int,
                                   num_training_steps: int,
                                   num_cycles: float = 0.5,
                                   last_epoch: int = -1):
    """Create a cosine learning rate schedule with warmup
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch number
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_optimizer(model: torch.nn.Module,
                    learning_rate: float,
                    optimizer_type: str = "adamw",
                    weight_decay: float = 0.01,
                    beta1: float = 0.9,
                    beta2: float = 0.999,
                    eps: float = 1e-8) -> torch.optim.Optimizer:
    """Create optimizer for model training
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        weight_decay: Weight decay coefficient
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        eps: Adam epsilon parameter
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps
        )
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=beta1
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(model: torch.nn.Module) -> str:
    """Get human-readable model size
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size string
    """
    total_params, trainable_params = count_parameters(model)
    
    def format_number(num):
        if num >= 1e9:
            return f"{num / 1e9:.1f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.1f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.1f}K"
        else:
            return str(num)
    
    return f"Total: {format_number(total_params)}, Trainable: {format_number(trainable_params)}"


def set_seed(seed: int):
    """Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device
    
    Returns:
        Device (cuda or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch to device
    
    Args:
        batch: Input batch
        device: Target device
        
    Returns:
        Batch moved to device
    """
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create attention mask from input IDs
    
    Args:
        input_ids: Input token IDs
        pad_token_id: Padding token ID
        
    Returns:
        Attention mask
    """
    return (input_ids != pad_token_id).long()


def truncate_sequences(sequences: List[torch.Tensor], max_length: int) -> List[torch.Tensor]:
    """Truncate sequences to maximum length
    
    Args:
        sequences: List of token sequences
        max_length: Maximum sequence length
        
    Returns:
        Truncated sequences
    """
    return [seq[:max_length] if len(seq) > max_length else seq for seq in sequences]


def pad_sequences(sequences: List[torch.Tensor],
                 pad_token_id: int,
                 max_length: Optional[int] = None,
                 padding_side: str = "right") -> torch.Tensor:
    """Pad sequences to the same length
    
    Args:
        sequences: List of token sequences
        pad_token_id: Padding token ID
        max_length: Maximum length (if None, use longest sequence)
        padding_side: Padding side ("left" or "right")
        
    Returns:
        Padded tensor
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), pad_token_id, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        if padding_side == "right":
            padded[i, :seq_len] = seq[:seq_len]
        else:  # left padding
            padded[i, -seq_len:] = seq[:seq_len]
    
    return padded