"""Base class for RL algorithms"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..utils.logger import get_logger
from ..utils.metrics import MetricsTracker


class BaseRLAlgorithm(ABC):
    """Base class for all RL algorithms"""
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.algorithm_config = config.get('algorithms', {})
        self.logger = get_logger()
        self.metrics_tracker = MetricsTracker()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.step = 0
        self.epoch = 0
        
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for the algorithm
        
        Args:
            batch: Batch of training data
            
        Returns:
            Tuple of (loss, metrics)
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def generate_response(self,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         max_new_tokens: int = 1024,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> torch.Tensor:
        """Generate response using the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated token IDs
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        self.model.train()
        return outputs
    
    def compute_log_probs(self,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log probabilities for given sequences
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (if None, use input_ids)
            
        Returns:
            Log probabilities
        """
        if labels is None:
            labels = input_ids
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather log probabilities for the target tokens
        gathered_log_probs = torch.gather(
            log_probs[:, :-1], 
            dim=-1, 
            index=labels[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding tokens
        mask = (labels[:, 1:] != -100) & (labels[:, 1:] != self.tokenizer.pad_token_id)
        gathered_log_probs = gathered_log_probs * mask
        
        return gathered_log_probs.sum(dim=-1)
    
    def compute_values(self, 
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute value estimates (for algorithms that need them)
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Value estimates
        """
        # This is a placeholder - implement value function if needed
        # For now, return zeros
        return torch.zeros(input_ids.shape[0], device=self.device)
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save algorithm checkpoint
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'config': self.config,
            'metrics': dict(self.metrics_tracker.metrics)
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load algorithm checkpoint
        
        Args:
            checkpoint_path: Path to load checkpoint from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def get_model_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable model parameters
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count statistics
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    def set_training_mode(self, training: bool = True) -> None:
        """Set model training mode
        
        Args:
            training: Whether to set training mode
        """
        self.model.train(training)
    
    def update_step(self) -> None:
        """Update training step counter"""
        self.step += 1
    
    def update_epoch(self) -> None:
        """Update training epoch counter"""
        self.epoch += 1
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log training metrics
        
        Args:
            metrics: Metrics to log
            step: Training step (if None, use current step)
        """
        if step is None:
            step = self.step
        
        self.metrics_tracker.update(metrics, step)
        
        # Log to logger
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}")
        
        # Log to TensorBoard if available
        if hasattr(self.logger, 'tb_logger') and self.logger.tb_logger:
            for key, value in metrics.items():
                self.logger.tb_logger.log_scalar(f"train/{key}", value, step)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest training metrics
        
        Returns:
            Latest metrics
        """
        return self.metrics_tracker.get_latest()
    
    def get_average_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get average training metrics
        
        Args:
            window: Window size for averaging (if None, use all)
            
        Returns:
            Average metrics
        """
        return self.metrics_tracker.get_average(window)
    
    def reset_metrics(self) -> None:
        """Reset metrics tracker"""
        self.metrics_tracker = MetricsTracker()
    
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training (move to device, etc.)
        
        Args:
            batch: Input batch
            
        Returns:
            Prepared batch
        """
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
        return prepared_batch
    
    def clip_gradients(self, max_norm: float = 1.0) -> float:
        """Clip gradients by norm
        
        Args:
            max_norm: Maximum gradient norm
            
        Returns:
            Gradient norm before clipping
        """
        parameters = self.get_model_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return grad_norm.item()
    
    def __str__(self) -> str:
        """String representation of the algorithm"""
        param_info = self.get_parameter_count()
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"trainable_params={param_info['trainable_parameters']:,}, "
            f"step={self.step}, "
            f"epoch={self.epoch}"
            f")"
        )