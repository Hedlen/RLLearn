"""Metrics computation utilities for RL training"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json


def compute_ppo_metrics(log_probs: torch.Tensor, 
                        old_log_probs: torch.Tensor,
                        advantages: torch.Tensor,
                        returns: torch.Tensor,
                        values: torch.Tensor,
                        clip_range: float = 0.2) -> Dict[str, float]:
    """Compute PPO-specific metrics
    
    Args:
        log_probs: Current policy log probabilities
        old_log_probs: Old policy log probabilities
        advantages: Advantage estimates
        returns: Discounted returns
        values: Value function estimates
        clip_range: PPO clipping range
        
    Returns:
        Dictionary of PPO metrics
    """
    with torch.no_grad():
        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Compute clipped ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        
        # Policy loss components
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # Value loss
        value_loss = torch.nn.functional.mse_loss(values, returns)
        
        # KL divergence
        kl_div = (old_log_probs - log_probs).mean()
        
        # Entropy
        entropy = -log_probs.mean()
        
        # Clipping fraction
        clipped = (torch.abs(ratio - 1) > clip_range).float()
        clip_fraction = clipped.mean()
        
        # Explained variance
        y_true = returns.cpu().numpy()
        y_pred = values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_divergence': kl_div.item(),
            'entropy': entropy.item(),
            'clip_fraction': clip_fraction.item(),
            'explained_variance': explained_var,
            'approx_kl': kl_div.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'returns_mean': returns.mean().item(),
            'returns_std': returns.std().item()
        }


def compute_dpo_metrics(policy_logps: torch.Tensor,
                       reference_logps: torch.Tensor,
                       chosen_logps: torch.Tensor,
                       rejected_logps: torch.Tensor,
                       beta: float = 0.1) -> Dict[str, float]:
    """Compute DPO-specific metrics
    
    Args:
        policy_logps: Policy log probabilities
        reference_logps: Reference model log probabilities
        chosen_logps: Log probabilities for chosen responses
        rejected_logps: Log probabilities for rejected responses
        beta: DPO temperature parameter
        
    Returns:
        Dictionary of DPO metrics
    """
    with torch.no_grad():
        # Compute log ratios
        chosen_logratios = chosen_logps - reference_logps
        rejected_logratios = rejected_logps - reference_logps
        
        # DPO loss
        logits = beta * (chosen_logratios - rejected_logratios)
        loss = -torch.nn.functional.logsigmoid(logits).mean()
        
        # Accuracy (how often chosen > rejected)
        accuracy = (chosen_logratios > rejected_logratios).float().mean()
        
        # Margin (difference between chosen and rejected)
        margin = (chosen_logratios - rejected_logratios).mean()
        
        return {
            'dpo_loss': loss.item(),
            'accuracy': accuracy.item(),
            'margin': margin.item(),
            'chosen_logratios_mean': chosen_logratios.mean().item(),
            'rejected_logratios_mean': rejected_logratios.mean().item(),
            'chosen_logratios_std': chosen_logratios.std().item(),
            'rejected_logratios_std': rejected_logratios.std().item()
        }


def compute_reward_metrics(rewards: torch.Tensor,
                          baseline_rewards: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Compute reward-related metrics
    
    Args:
        rewards: Reward values
        baseline_rewards: Optional baseline rewards for comparison
        
    Returns:
        Dictionary of reward metrics
    """
    with torch.no_grad():
        metrics = {
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'reward_min': rewards.min().item(),
            'reward_max': rewards.max().item(),
            'reward_median': rewards.median().item()
        }
        
        if baseline_rewards is not None:
            improvement = rewards - baseline_rewards
            metrics.update({
                'reward_improvement_mean': improvement.mean().item(),
                'reward_improvement_std': improvement.std().item(),
                'improvement_rate': (improvement > 0).float().mean().item()
            })
        
        return metrics


def compute_generation_metrics(generated_texts: List[str],
                              reference_texts: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute text generation metrics
    
    Args:
        generated_texts: List of generated text strings
        reference_texts: Optional reference texts for comparison
        
    Returns:
        Dictionary of generation metrics
    """
    metrics = {}
    
    # Length statistics
    lengths = [len(text.split()) for text in generated_texts]
    metrics.update({
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'length_min': np.min(lengths),
        'length_max': np.max(lengths)
    })
    
    # Diversity metrics
    unique_texts = len(set(generated_texts))
    metrics['diversity'] = unique_texts / len(generated_texts)
    
    # Repetition metrics
    repetition_scores = []
    for text in generated_texts:
        words = text.split()
        if len(words) > 1:
            unique_words = len(set(words))
            repetition_score = 1 - (unique_words / len(words))
            repetition_scores.append(repetition_score)
    
    if repetition_scores:
        metrics['repetition_mean'] = np.mean(repetition_scores)
        metrics['repetition_std'] = np.std(repetition_scores)
    
    return metrics


def compute_training_metrics(loss_history: List[float],
                           learning_rates: List[float],
                           grad_norms: List[float]) -> Dict[str, float]:
    """Compute training process metrics
    
    Args:
        loss_history: List of loss values
        learning_rates: List of learning rate values
        grad_norms: List of gradient norm values
        
    Returns:
        Dictionary of training metrics
    """
    metrics = {}
    
    if loss_history:
        metrics.update({
            'loss_mean': np.mean(loss_history),
            'loss_std': np.std(loss_history),
            'loss_trend': np.polyfit(range(len(loss_history)), loss_history, 1)[0]
        })
    
    if learning_rates:
        metrics.update({
            'lr_mean': np.mean(learning_rates),
            'lr_current': learning_rates[-1] if learning_rates else 0
        })
    
    if grad_norms:
        metrics.update({
            'grad_norm_mean': np.mean(grad_norms),
            'grad_norm_std': np.std(grad_norms),
            'grad_norm_max': np.max(grad_norms)
        })
    
    return metrics


def compute_metrics(predictions: Any, 
                   labels: Any, 
                   metric_type: str = "classification") -> Dict[str, float]:
    """Compute evaluation metrics based on type
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        metric_type: Type of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    if metric_type == "classification":
        return compute_classification_metrics(predictions, labels)
    elif metric_type == "regression":
        return compute_regression_metrics(predictions, labels)
    elif metric_type == "generation":
        return compute_generation_metrics(predictions, labels)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def compute_classification_metrics(predictions: np.ndarray, 
                                 labels: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_regression_metrics(predictions: np.ndarray, 
                             labels: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }


class MetricsTracker:
    """Track and aggregate metrics during training"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step = 0
    
    def update(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Update metrics"""
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest metrics"""
        return {key: values[-1] for key, values in self.metrics.items() if values}
    
    def get_average(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get average metrics over window"""
        averages = {}
        for key, values in self.metrics.items():
            if values:
                if window is not None:
                    values = values[-window:]
                averages[key] = np.mean(values)
        return averages
    
    def get_average_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get average metrics over window (alias for get_average)"""
        return self.get_average(window)
    
    def save(self, filepath: str):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from file"""
        with open(filepath, 'r') as f:
            loaded_metrics = json.load(f)
            self.metrics = defaultdict(list, loaded_metrics)