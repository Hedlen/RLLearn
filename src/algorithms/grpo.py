"""Group Relative Policy Optimization (GRPO) algorithm implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseRLAlgorithm
from .utils import compute_advantages, compute_returns
from ..utils.metrics import compute_ppo_metrics


class GRPOAlgorithm(BaseRLAlgorithm):
    """Group Relative Policy Optimization algorithm for language models
    
    GRPO is a variant of PPO that uses group-relative advantages to reduce
    variance and improve training stability.
    """
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: Dict[str, Any],
                 value_model: Optional[PreTrainedModel] = None):
        super().__init__(model, tokenizer, config)
        
        # GRPO-specific configuration
        self.grpo_config = self.algorithm_config.get('grpo', {})
        self.clip_range = self.grpo_config.get('clip_range', 0.2)
        self.vf_coef = self.grpo_config.get('vf_coef', 0.1)
        self.ent_coef = self.grpo_config.get('ent_coef', 0.0)
        self.target_kl = self.grpo_config.get('target_kl', 0.1)
        self.gamma = self.grpo_config.get('gamma', 1.0)
        self.gae_lambda = self.grpo_config.get('gae_lambda', 0.95)
        self.grpo_epochs = self.grpo_config.get('grpo_epochs', 4)
        self.mini_batch_size = self.grpo_config.get('mini_batch_size', 1)
        
        # Group-relative specific parameters
        self.group_size = self.grpo_config.get('group_size', 4)
        self.baseline_type = self.grpo_config.get('baseline_type', 'group_mean')  # group_mean, group_median, learned
        self.advantage_normalization = self.grpo_config.get('advantage_normalization', 'group')  # group, global, none
        self.use_group_critic = self.grpo_config.get('use_group_critic', False)
        
        # Value model (can be the same as policy model or separate)
        self.value_model = value_model if value_model is not None else model
        self.separate_value_model = value_model is not None
        
        # Add value head if using the same model for both policy and value
        if not self.separate_value_model:
            self.value_head = nn.Linear(model.config.hidden_size, 1)
            self.value_head.to(self.device)
        
        # Group critic for group-relative baseline
        if self.use_group_critic:
            self.group_critic = nn.Sequential(
                nn.Linear(model.config.hidden_size * self.group_size, model.config.hidden_size),
                nn.ReLU(),
                nn.Linear(model.config.hidden_size, 1)
            )
            self.group_critic.to(self.device)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss
        
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
        
        batch_size = query_input_ids.shape[0]
        
        # Ensure batch size is divisible by group size
        if batch_size % self.group_size != 0:
            # Pad or truncate to make it divisible
            target_size = (batch_size // self.group_size) * self.group_size
            if target_size == 0:
                target_size = self.group_size
            
            query_input_ids = query_input_ids[:target_size]
            response_input_ids = response_input_ids[:target_size]
            rewards = rewards[:target_size]
            if old_log_probs is not None:
                old_log_probs = old_log_probs[:target_size]
            if old_values is not None:
                old_values = old_values[:target_size]
            
            batch_size = target_size
        
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
        
        # Compute group-relative advantages
        advantages = self._compute_group_relative_advantages(
            rewards, current_values, old_values, full_input_ids, attention_mask
        )
        
        # Compute returns
        returns = compute_returns(rewards, current_values, self.gamma)
        
        # Compute policy loss
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(current_values, returns)
        
        # Compute group critic loss if used
        group_critic_loss = 0.0
        if self.use_group_critic:
            group_critic_loss = self._compute_group_critic_loss(
                full_input_ids, attention_mask, rewards
            )
        
        # Compute entropy loss
        entropy_loss = self._compute_entropy_loss(full_input_ids, attention_mask)
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.vf_coef * value_loss + 
            0.1 * group_critic_loss -  # Weight for group critic loss
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
            'entropy_loss': entropy_loss.item(),
            'group_critic_loss': group_critic_loss if isinstance(group_critic_loss, float) else group_critic_loss.item()
        })
        
        return total_loss, metrics
    
    def _compute_group_relative_advantages(self,
                                          rewards: torch.Tensor,
                                          current_values: torch.Tensor,
                                          old_values: torch.Tensor,
                                          full_input_ids: torch.Tensor,
                                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages
        
        Args:
            rewards: Reward tensor
            current_values: Current value estimates
            old_values: Old value estimates
            full_input_ids: Full input token IDs
            attention_mask: Attention mask
            
        Returns:
            Group-relative advantages
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // self.group_size
        
        # Reshape into groups
        group_rewards = rewards.view(num_groups, self.group_size)
        group_values = current_values.view(num_groups, self.group_size)
        
        # Compute group baselines
        if self.baseline_type == 'group_mean':
            group_baselines = group_rewards.mean(dim=1, keepdim=True)
        elif self.baseline_type == 'group_median':
            group_baselines = group_rewards.median(dim=1, keepdim=True)[0]
        elif self.baseline_type == 'learned' and self.use_group_critic:
            group_baselines = self._compute_learned_group_baseline(
                full_input_ids, attention_mask
            )
        else:
            # Fallback to group mean
            group_baselines = group_rewards.mean(dim=1, keepdim=True)
        
        # Compute group-relative advantages
        group_advantages = group_rewards - group_baselines
        
        # Flatten back to original shape
        advantages = group_advantages.view(-1)
        
        # Normalize advantages
        if self.advantage_normalization == 'group':
            # Normalize within each group
            for i in range(num_groups):
                start_idx = i * self.group_size
                end_idx = (i + 1) * self.group_size
                group_adv = advantages[start_idx:end_idx]
                advantages[start_idx:end_idx] = (
                    group_adv - group_adv.mean()
                ) / (group_adv.std() + 1e-8)
        elif self.advantage_normalization == 'global':
            # Global normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # else: no normalization
        
        return advantages
    
    def _compute_learned_group_baseline(self,
                                       full_input_ids: torch.Tensor,
                                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute learned group baseline using group critic
        
        Args:
            full_input_ids: Full input token IDs
            attention_mask: Attention mask
            
        Returns:
            Group baseline values
        """
        batch_size = full_input_ids.shape[0]
        num_groups = batch_size // self.group_size
        
        # Get hidden states for all samples
        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of the last token
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_size)
        
        # Reshape into groups and concatenate
        group_hidden = hidden_states.view(
            num_groups, self.group_size * self.model.config.hidden_size
        )
        
        # Compute group baselines
        group_baselines = self.group_critic(group_hidden)  # (num_groups, 1)
        
        # Expand to match group size
        group_baselines = group_baselines.expand(-1, self.group_size)  # (num_groups, group_size)
        
        return group_baselines
    
    def _compute_group_critic_loss(self,
                                  full_input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  rewards: torch.Tensor) -> torch.Tensor:
        """Compute group critic loss
        
        Args:
            full_input_ids: Full input token IDs
            attention_mask: Attention mask
            rewards: Reward tensor
            
        Returns:
            Group critic loss
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // self.group_size
        
        # Compute predicted group baselines
        predicted_baselines = self._compute_learned_group_baseline(
            full_input_ids, attention_mask
        ).view(-1)  # Flatten
        
        # Compute target group baselines (group means)
        group_rewards = rewards.view(num_groups, self.group_size)
        target_baselines = group_rewards.mean(dim=1).repeat_interleave(self.group_size)
        
        # MSE loss
        loss = F.mse_loss(predicted_baselines, target_baselines)
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one GRPO training step
        
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
        
        # Multiple GRPO epochs
        for epoch in range(self.grpo_epochs):
            # Mini-batch training
            batch_size = batch['query_input_ids'].shape[0]
            
            # Ensure batch size is divisible by group size for mini-batching
            effective_batch_size = (batch_size // self.group_size) * self.group_size
            if effective_batch_size == 0:
                effective_batch_size = min(batch_size, self.group_size)
            
            indices = torch.randperm(effective_batch_size)
            
            for start_idx in range(0, effective_batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, effective_batch_size)
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
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save GRPO checkpoint
        
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
        
        # Save group critic if used
        if self.use_group_critic:
            checkpoint['group_critic_state_dict'] = self.group_critic.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved GRPO checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load GRPO checkpoint
        
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
        
        # Load group critic if used
        if self.use_group_critic and 'group_critic_state_dict' in checkpoint:
            self.group_critic.load_state_dict(checkpoint['group_critic_state_dict'])
        
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
        
        self.logger.info(f"Loaded GRPO checkpoint from {checkpoint_path}")