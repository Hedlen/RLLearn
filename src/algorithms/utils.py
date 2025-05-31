"""Utility functions for reinforcement learning algorithms"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_advantages(rewards: torch.Tensor,
                      values: torch.Tensor,
                      next_values: Optional[torch.Tensor] = None,
                      gamma: float = 0.99,
                      gae_lambda: float = 0.95) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: Reward tensor of shape (batch_size, sequence_length)
        values: Value estimates of shape (batch_size, sequence_length)
        next_values: Next value estimates, if None uses shifted values
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Advantages tensor of same shape as rewards
    """
    if next_values is None:
        # Use shifted values with zero padding
        next_values = torch.cat([
            values[:, 1:],
            torch.zeros_like(values[:, :1])
        ], dim=1)
    
    # Compute TD errors
    td_errors = rewards + gamma * next_values - values
    
    # Compute GAE
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    # Work backwards through time
    for t in reversed(range(rewards.shape[1])):
        gae = td_errors[:, t] + gamma * gae_lambda * gae
        advantages[:, t] = gae
    
    return advantages


def compute_returns(rewards: torch.Tensor,
                   values: torch.Tensor,
                   gamma: float = 0.99,
                   use_gae: bool = True,
                   gae_lambda: float = 0.95) -> torch.Tensor:
    """Compute returns for value function training
    
    Args:
        rewards: Reward tensor of shape (batch_size, sequence_length)
        values: Value estimates of shape (batch_size, sequence_length)
        gamma: Discount factor
        use_gae: Whether to use GAE for return computation
        gae_lambda: GAE lambda parameter
        
    Returns:
        Returns tensor of same shape as rewards
    """
    if use_gae:
        # GAE-based returns
        advantages = compute_advantages(rewards, values, gamma=gamma, gae_lambda=gae_lambda)
        returns = advantages + values
    else:
        # Standard discounted returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        # Work backwards through time
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
    
    return returns


def compute_policy_gradient_loss(log_probs: torch.Tensor,
                                advantages: torch.Tensor,
                                old_log_probs: Optional[torch.Tensor] = None,
                                clip_range: float = 0.2) -> torch.Tensor:
    """Compute policy gradient loss with optional PPO clipping
    
    Args:
        log_probs: Current log probabilities
        advantages: Advantage estimates
        old_log_probs: Old log probabilities for PPO clipping
        clip_range: PPO clipping range
        
    Returns:
        Policy gradient loss
    """
    if old_log_probs is not None:
        # PPO clipped loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        loss = -torch.min(surr1, surr2).mean()
    else:
        # Standard policy gradient
        loss = -(log_probs * advantages).mean()
    
    return loss


def compute_value_loss(values: torch.Tensor,
                      returns: torch.Tensor,
                      old_values: Optional[torch.Tensor] = None,
                      clip_range: float = 0.2) -> torch.Tensor:
    """Compute value function loss with optional clipping
    
    Args:
        values: Current value estimates
        returns: Target returns
        old_values: Old value estimates for clipping
        clip_range: Value clipping range
        
    Returns:
        Value function loss
    """
    if old_values is not None:
        # Clipped value loss (PPO-style)
        value_clipped = old_values + torch.clamp(
            values - old_values, -clip_range, clip_range
        )
        loss1 = F.mse_loss(values, returns)
        loss2 = F.mse_loss(value_clipped, returns)
        loss = torch.max(loss1, loss2)
    else:
        # Standard MSE loss
        loss = F.mse_loss(values, returns)
    
    return loss


def compute_entropy_bonus(logits: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute entropy bonus for exploration
    
    Args:
        logits: Model logits of shape (batch_size, sequence_length, vocab_size)
        attention_mask: Attention mask to exclude padding tokens
        
    Returns:
        Entropy bonus (higher is better for exploration)
    """
    # Compute probabilities and log probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute entropy
    entropy = -(probs * log_probs).sum(dim=-1)
    
    # Apply attention mask if provided
    if attention_mask is not None:
        # Shift mask for next token prediction
        mask = attention_mask[:, 1:].float()
        entropy = entropy[:, :-1] * mask
        return entropy.sum() / mask.sum()
    else:
        return entropy.mean()


def compute_kl_divergence(log_probs: torch.Tensor,
                         old_log_probs: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute KL divergence between old and new policies
    
    Args:
        log_probs: Current log probabilities
        old_log_probs: Old log probabilities
        attention_mask: Attention mask to exclude padding tokens
        
    Returns:
        KL divergence
    """
    kl = old_log_probs - log_probs
    
    if attention_mask is not None:
        # Apply mask
        mask = attention_mask.float()
        kl = kl * mask
        return kl.sum() / mask.sum()
    else:
        return kl.mean()


def normalize_advantages(advantages: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance
    
    Args:
        advantages: Advantage tensor
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized advantages
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


def discount_rewards(rewards: torch.Tensor,
                    gamma: float = 0.99,
                    normalize: bool = True) -> torch.Tensor:
    """Compute discounted cumulative rewards
    
    Args:
        rewards: Reward tensor of shape (batch_size, sequence_length)
        gamma: Discount factor
        normalize: Whether to normalize the returns
        
    Returns:
        Discounted returns
    """
    batch_size, seq_len = rewards.shape
    returns = torch.zeros_like(rewards)
    
    for i in range(batch_size):
        running_return = 0
        for t in reversed(range(seq_len)):
            running_return = rewards[i, t] + gamma * running_return
            returns[i, t] = running_return
    
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def compute_explained_variance(values: torch.Tensor,
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
    
    return 1.0 - torch.var(returns - values) / var_returns


def whiten(tensor: torch.Tensor,
          shift_mean: bool = True,
          eps: float = 1e-8) -> torch.Tensor:
    """Whiten a tensor (zero mean, unit variance)
    
    Args:
        tensor: Input tensor
        shift_mean: Whether to shift mean to zero
        eps: Small epsilon for numerical stability
        
    Returns:
        Whitened tensor
    """
    if shift_mean:
        tensor = tensor - tensor.mean()
    
    std = tensor.std()
    if std > eps:
        tensor = tensor / std
    
    return tensor


def compute_clipfrac(ratio: torch.Tensor,
                    clip_range: float = 0.2) -> float:
    """Compute fraction of ratios that were clipped
    
    Args:
        ratio: Probability ratios
        clip_range: Clipping range
        
    Returns:
        Fraction of clipped ratios
    """
    clipped = torch.logical_or(
        ratio < (1 - clip_range),
        ratio > (1 + clip_range)
    )
    return clipped.float().mean().item()


def compute_approx_kl(log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor) -> float:
    """Compute approximate KL divergence
    
    Args:
        log_probs: Current log probabilities
        old_log_probs: Old log probabilities
        
    Returns:
        Approximate KL divergence
    """
    # Approximate KL: KL ≈ (log_ratio)^2 / 2 for small differences
    log_ratio = log_probs - old_log_probs
    return (log_ratio ** 2 / 2).mean().item()


def masked_mean(tensor: torch.Tensor,
               mask: torch.Tensor,
               dim: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with mask
    
    Args:
        tensor: Input tensor
        mask: Boolean or float mask
        dim: Dimension to compute mean over
        
    Returns:
        Masked mean
    """
    masked_tensor = tensor * mask.float()
    if dim is not None:
        return masked_tensor.sum(dim=dim) / mask.float().sum(dim=dim).clamp(min=1)
    else:
        return masked_tensor.sum() / mask.float().sum().clamp(min=1)


def masked_var(tensor: torch.Tensor,
              mask: torch.Tensor,
              dim: Optional[int] = None,
              unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with mask
    
    Args:
        tensor: Input tensor
        mask: Boolean or float mask
        dim: Dimension to compute variance over
        unbiased: Whether to use unbiased estimator
        
    Returns:
        Masked variance
    """
    mean = masked_mean(tensor, mask, dim=dim)
    if dim is not None:
        mean = mean.unsqueeze(dim)
    
    squared_diff = (tensor - mean) ** 2
    masked_squared_diff = squared_diff * mask.float()
    
    if dim is not None:
        sum_squared_diff = masked_squared_diff.sum(dim=dim)
        count = mask.float().sum(dim=dim)
    else:
        sum_squared_diff = masked_squared_diff.sum()
        count = mask.float().sum()
    
    if unbiased:
        count = count - 1
    
    return sum_squared_diff / count.clamp(min=1)