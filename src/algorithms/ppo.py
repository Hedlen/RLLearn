"""Proximal Policy Optimization (PPO) algorithm implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseRLAlgorithm
from .utils import compute_advantages, compute_returns
from ..utils.metrics import compute_ppo_metrics


class PPOAlgorithm(BaseRLAlgorithm):
    """Proximal Policy Optimization algorithm for language models"""
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: Dict[str, Any],
                 value_model: Optional[PreTrainedModel] = None):
        super().__init__(model, tokenizer, config)
        
        # PPO-specific configuration
        self.ppo_config = self.algorithm_config.get('ppo', {})
        self.clip_range = self.ppo_config.get('clip_range', 0.2)
        self.vf_coef = self.ppo_config.get('vf_coef', 0.1)
        self.ent_coef = self.ppo_config.get('ent_coef', 0.0)
        self.target_kl = self.ppo_config.get('target_kl', 0.1)
        self.gamma = self.ppo_config.get('gamma', 1.0)
        self.gae_lambda = self.ppo_config.get('gae_lambda', 0.95)
        self.ppo_epochs = self.ppo_config.get('ppo_epochs', 4)
        self.mini_batch_size = self.ppo_config.get('mini_batch_size', 1)
        
        # Value model (can be the same as policy model or separate)
        self.value_model = value_model if value_model is not None else model
        self.separate_value_model = value_model is not None
        
        # Add value head if using the same model for both policy and value
        if not self.separate_value_model:
            self.value_head = nn.Linear(model.config.hidden_size, 1)
            self.value_head.to(self.device)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss
        
        Args:
            batch: Batch containing queries, responses, rewards, old_log_probs, etc.
            
        Returns:
            Tuple of (total_loss, metrics)
        """
        # Extract batch components
        query_input_ids = batch['query_input_ids']
        response_input_ids = batch['response_input_ids']
        rewards = batch['rewards']
        old_log_probs = batch.get('old_log_probs')
        old_values = batch.get('old_values')
        
        # Combine query and response
        full_input_ids = torch.cat([query_input_ids, response_input_ids], dim=1)
        attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()
        
        # Compute current log probabilities
        current_log_probs = self.compute_log_probs(
            full_input_ids, 
            attention_mask, 
            response_input_ids
        )
        
        # Compute current values
        current_values = self.compute_values(full_input_ids, attention_mask)
        
        # If old values/log_probs not provided, use current ones (first iteration)
        if old_log_probs is None:
            old_log_probs = current_log_probs.detach()
        if old_values is None:
            old_values = current_values.detach()
        
        # Compute returns and advantages
        returns = compute_returns(rewards, current_values, self.gamma)
        advantages = compute_advantages(
            rewards, current_values, old_values, self.gamma, self.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(current_values, returns)
        
        # Compute entropy loss
        entropy_loss = self._compute_entropy_loss(full_input_ids, attention_mask)
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.vf_coef * value_loss - 
            self.ent_coef * entropy_loss
        )
        
        # Compute metrics
        metrics = compute_ppo_metrics(
            current_log_probs, old_log_probs, advantages, returns, 
            current_values, self.clip_range
        )
        
        metrics.update({
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        })
        
        return total_loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one PPO training step
        
        Args:
            batch: Training batch
            
        Returns:
            Training metrics
        """
        batch = self.prepare_batch(batch)
        
        # Store old log probs and values for multiple epochs
        with torch.no_grad():
            query_input_ids = batch['query_input_ids']
            response_input_ids = batch['response_input_ids']
            full_input_ids = torch.cat([query_input_ids, response_input_ids], dim=1)
            attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()
            
            old_log_probs = self.compute_log_probs(
                full_input_ids, attention_mask, response_input_ids
            )
            old_values = self.compute_values(full_input_ids, attention_mask)
        
        batch['old_log_probs'] = old_log_probs
        batch['old_values'] = old_values
        
        total_metrics = {}
        
        # Multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            # Mini-batch training
            batch_size = batch['query_input_ids'].shape[0]
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size)
                mini_batch_indices = indices[start_idx:end_idx]
                
                # Create mini-batch
                mini_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        mini_batch[key] = value[mini_batch_indices]
                    else:
                        mini_batch[key] = value
                
                # Compute loss and metrics
                loss, metrics = self.compute_loss(mini_batch)
                
                # Backward pass
                loss.backward()
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = []
                    total_metrics[key].append(value)
                
                # Early stopping based on KL divergence
                if metrics.get('kl_divergence', 0) > self.target_kl:
                    self.logger.info(f"Early stopping due to KL divergence: {metrics['kl_divergence']:.4f}")
                    break
        
        # Average metrics
        averaged_metrics = {}
        for key, values in total_metrics.items():
            averaged_metrics[key] = sum(values) / len(values)
        
        self.update_step()
        return averaged_metrics
    
    def compute_values(self, 
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute value estimates
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Value estimates
        """
        if self.separate_value_model:
            # Use separate value model
            outputs = self.value_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Assume value model has a value head
            values = outputs.last_hidden_state[:, -1, :]  # Use last token
            values = self.value_head(values).squeeze(-1)
        else:
            # Use same model with value head
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]  # Use last token
            values = self.value_head(last_hidden_state).squeeze(-1)
        
        return values
    
    def _compute_entropy_loss(self, 
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss for regularization
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Entropy loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Mask out padding tokens
        mask = attention_mask[:, 1:].float()  # Shift by 1 for next token prediction
        entropy = entropy[:, :-1] * mask  # Remove last position
        
        return entropy.sum() / mask.sum()
    
    def generate_and_evaluate(self,
                             queries: torch.Tensor,
                             query_masks: torch.Tensor,
                             reward_fn: callable,
                             max_new_tokens: int = 1024) -> Dict[str, torch.Tensor]:
        """Generate responses and evaluate with reward function
        
        Args:
            queries: Query token IDs
            query_masks: Query attention masks
            reward_fn: Function to compute rewards
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Dictionary with generated responses, rewards, etc.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate responses
            generated = self.generate_response(
                queries, query_masks, max_new_tokens=max_new_tokens
            )
            
            # Extract only the generated part (remove query)
            query_length = queries.shape[1]
            responses = generated[:, query_length:]
            
            # Compute rewards
            rewards = reward_fn(queries, responses)
            
            # Compute log probabilities and values
            full_input_ids = generated
            attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()
            
            log_probs = self.compute_log_probs(
                full_input_ids, attention_mask, responses
            )
            values = self.compute_values(full_input_ids, attention_mask)
        
        self.model.train()
        
        return {
            'query_input_ids': queries,
            'response_input_ids': responses,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values
        }
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save PPO checkpoint
        
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
        
        # Save value head if using shared model
        if not self.separate_value_model:
            checkpoint['value_head_state_dict'] = self.value_head.state_dict()
        
        # Save separate value model if used
        if self.separate_value_model:
            checkpoint['value_model_state_dict'] = self.value_model.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved PPO checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load PPO checkpoint
        
        Args:
            checkpoint_path: Path to load checkpoint from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        # Load value head if using shared model
        if not self.separate_value_model and 'value_head_state_dict' in checkpoint:
            self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        
        # Load separate value model if used
        if self.separate_value_model and 'value_model_state_dict' in checkpoint:
            self.value_model.load_state_dict(checkpoint['value_model_state_dict'])
        
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
        
        self.logger.info(f"Loaded PPO checkpoint from {checkpoint_path}")