"""PPO trainer implementation"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer, TrainingConfig
from ..algorithms import PPOAlgorithm
from ..models import PolicyModel, ValueModel
from ..utils import compute_advantages, compute_returns


@dataclass
class PPOTrainingConfig(TrainingConfig):
    """PPO-specific training configuration"""
    # PPO hyperparameters
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_kl_divergence: float = 0.01
    target_kl: Optional[float] = None
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    
    # Training parameters
    mini_batch_size: Optional[int] = None
    normalize_advantages: bool = True
    
    # Reward model
    reward_model_path: Optional[str] = None
    

class PPOTrainer(BaseTrainer):
    """PPO trainer for reinforcement learning from human feedback
    
    This trainer implements the Proximal Policy Optimization algorithm
    for training language models with human feedback.
    """
    
    def __init__(self,
                 config: PPOTrainingConfig,
                 policy_model: PolicyModel,
                 value_model: ValueModel,
                 tokenizer,
                 train_dataloader: Optional[DataLoader] = None,
                 eval_dataloader: Optional[DataLoader] = None,
                 reward_model: Optional[Any] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 callbacks: Optional[List] = None):
        
        # Initialize PPO algorithm
        self.ppo_algorithm = PPOAlgorithm(
            policy_model=policy_model,
            value_model=value_model,
            config={
                'clip_range': config.clip_range,
                'clip_range_vf': config.clip_range_vf,
                'value_loss_coef': config.value_loss_coef,
                'entropy_coef': config.entropy_coef,
                'max_kl_divergence': config.max_kl_divergence,
                'target_kl': config.target_kl,
                'ppo_epochs': config.ppo_epochs,
                'mini_batch_size': config.mini_batch_size or config.per_device_train_batch_size,
                'gamma': config.gamma,
                'gae_lambda': config.gae_lambda,
                'normalize_advantages': config.normalize_advantages
            }
        )
        
        self.reward_model = reward_model
        
        # Initialize base trainer with policy model
        super().__init__(
            config=config,
            model=policy_model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks
        )
        
        self.policy_model = policy_model
        self.value_model = value_model
        
        # Move models to device
        self.value_model = self.value_model.to(self.device)
        if self.reward_model is not None:
            self.reward_model = self.reward_model.to(self.device)
    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for both policy and value models"""
        if self.optimizer is not None:
            return self.optimizer
        
        # Get trainable parameters from both models
        policy_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        value_params = [p for p in self.value_model.parameters() if p.requires_grad]
        
        all_params = policy_params + value_params
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        self.optimizer = optimizer
        return optimizer
    
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss for a batch
        
        Args:
            batch: Input batch containing:
                - input_ids: Prompt token IDs
                - attention_mask: Attention mask for prompts
                - response_ids: Response token IDs (optional)
                - rewards: Rewards for responses (optional)
                
        Returns:
            Tuple of (loss, metrics)
        """
        # Extract batch data
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        
        # Generate responses if not provided
        if 'response_ids' not in batch:
            with torch.no_grad():
                response_ids, response_log_probs = self.policy_model.sample_actions(
                    states=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                )
        else:
            response_ids = batch['response_ids']
            response_log_probs = self.policy_model.compute_action_log_probs(
                states=input_ids,
                actions=response_ids,
                attention_mask=attention_mask
            )
        
        # Compute rewards if not provided
        if 'rewards' not in batch:
            if self.reward_model is not None:
                rewards = self._compute_rewards(input_ids, response_ids, attention_mask)
            else:
                # Use dummy rewards for testing
                rewards = torch.zeros(response_ids.shape[0], device=self.device)
        else:
            rewards = batch['rewards']
        
        # Compute values
        full_input_ids = torch.cat([input_ids, response_ids], dim=1)
        values = self.value_model.compute_values(full_input_ids)
        
        # Extract values for responses only
        prompt_length = input_ids.shape[1]
        response_values = values[:, prompt_length:]
        
        # Compute advantages and returns
        advantages = compute_advantages(
            rewards=rewards,
            values=response_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        returns = compute_returns(
            rewards=rewards,
            values=response_values,
            gamma=self.config.gamma
        )
        
        # Store old log probs for PPO clipping
        old_log_probs = response_log_probs.detach()
        
        # PPO training loop
        total_loss = 0.0
        total_metrics = {}
        
        for ppo_epoch in range(self.config.ppo_epochs):
            # Compute current log probs
            current_log_probs = self.policy_model.compute_action_log_probs(
                states=input_ids,
                actions=response_ids,
                attention_mask=attention_mask
            )
            
            # Compute current values
            current_values = self.value_model.compute_values(full_input_ids)
            current_response_values = current_values[:, prompt_length:]
            
            # Compute PPO loss
            loss, metrics = self.ppo_algorithm.compute_loss(
                states=input_ids,
                actions=response_ids,
                advantages=advantages,
                returns=returns,
                old_log_probs=old_log_probs,
                current_log_probs=current_log_probs,
                values=current_response_values,
                attention_mask=attention_mask
            )
            
            total_loss += loss
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
        
        # Average metrics over PPO epochs
        avg_loss = total_loss / self.config.ppo_epochs
        avg_metrics = {k: v / self.config.ppo_epochs for k, v in total_metrics.items()}
        
        # Add additional metrics
        avg_metrics.update({
            'rewards_mean': rewards.mean().item(),
            'rewards_std': rewards.std().item(),
            'response_length': response_ids.shape[1],
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'returns_mean': returns.mean().item(),
            'returns_std': returns.std().item()
        })
        
        return avg_loss, avg_metrics
    
    def _compute_rewards(self,
                        input_ids: torch.Tensor,
                        response_ids: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute rewards using reward model
        
        Args:
            input_ids: Prompt token IDs
            response_ids: Response token IDs
            attention_mask: Attention mask
            
        Returns:
            Rewards for each response
        """
        if self.reward_model is None:
            return torch.zeros(response_ids.shape[0], device=self.device)
        
        # Combine prompt and response
        full_input_ids = torch.cat([input_ids, response_ids], dim=1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        response_mask = (response_ids != self.tokenizer.pad_token_id).long()
        full_attention_mask = torch.cat([attention_mask, response_mask], dim=1)
        
        # Compute rewards
        with torch.no_grad():
            rewards = self.reward_model.compute_rewards(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask
            )
        
        return rewards
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model
        
        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.policy_model.eval()
        self.value_model.eval()
        
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute loss and metrics
                loss, metrics = self.compute_loss(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
                total_metrics['eval_loss'] = total_metrics.get('eval_loss', 0.0) + loss.item()
                num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        # Set models back to training mode
        self.policy_model.train()
        self.value_model.train()
        
        return avg_metrics
    
    def generate_responses(self,
                          prompts: List[str],
                          max_new_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          top_k: Optional[int] = None,
                          top_p: Optional[float] = None,
                          do_sample: bool = True) -> List[str]:
        """Generate responses for given prompts
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            do_sample: Whether to use sampling
            
        Returns:
            List of generated responses
        """
        self.policy_model.eval()
        
        # Set generation parameters
        gen_kwargs = {
            'max_new_tokens': max_new_tokens or self.config.max_new_tokens,
            'temperature': temperature or self.config.temperature,
            'top_k': top_k or self.config.top_k,
            'top_p': top_p or self.config.top_p,
            'do_sample': do_sample
        }
        
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Generate response
                generated = self.policy_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **gen_kwargs
                )
                
                # Extract response (remove prompt)
                prompt_length = inputs['input_ids'].shape[1]
                response_ids = generated[:, prompt_length:]
                
                # Decode response
                response = self.tokenizer.decode(
                    response_ids[0],
                    skip_special_tokens=True
                )
                
                responses.append(response)
        
        self.policy_model.train()
        return responses
    
    def compute_policy_metrics(self,
                              prompts: List[str],
                              responses: List[str]) -> Dict[str, float]:
        """Compute policy-specific metrics
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            Policy metrics
        """
        metrics = {}
        
        # Tokenize prompts and responses
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        response_inputs = self.tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # Compute log probabilities
            log_probs = self.policy_model.compute_action_log_probs(
                states=prompt_inputs['input_ids'],
                actions=response_inputs['input_ids'],
                attention_mask=prompt_inputs['attention_mask']
            )
            
            # Compute entropy
            full_input_ids = torch.cat([
                prompt_inputs['input_ids'],
                response_inputs['input_ids']
            ], dim=1)
            
            entropy = self.policy_model.compute_entropy(full_input_ids)
            
            # Compute values
            values = self.value_model.compute_values(full_input_ids)
            
            metrics.update({
                'policy_log_probs_mean': log_probs.mean().item(),
                'policy_log_probs_std': log_probs.std().item(),
                'policy_entropy': entropy.item(),
                'value_estimates_mean': values.mean().item(),
                'value_estimates_std': values.std().item(),
                'response_lengths_mean': response_inputs['input_ids'].shape[1],
            })
        
        return metrics
    
    def save_model(self, save_directory: str):
        """Save both policy and value models
        
        Args:
            save_directory: Directory to save models
        """
        import os
        
        # Save policy model
        policy_dir = os.path.join(save_directory, "policy_model")
        self.policy_model.save_pretrained(policy_dir)
        
        # Save value model
        value_dir = os.path.join(save_directory, "value_model")
        self.value_model.save_pretrained(value_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save PPO config
        import json
        config_dict = {
            'ppo_epochs': self.config.ppo_epochs,
            'clip_range': self.config.clip_range,
            'clip_range_vf': self.config.clip_range_vf,
            'value_loss_coef': self.config.value_loss_coef,
            'entropy_coef': self.config.entropy_coef,
            'max_kl_divergence': self.config.max_kl_divergence,
            'target_kl': self.config.target_kl,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
            'normalize_advantages': self.config.normalize_advantages
        }
        
        with open(os.path.join(save_directory, 'ppo_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"PPO models saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls,
                       model_directory: str,
                       config: Optional[PPOTrainingConfig] = None,
                       **kwargs):
        """Load PPO trainer from pretrained models
        
        Args:
            model_directory: Directory containing saved models
            config: Training configuration
            **kwargs: Additional arguments
            
        Returns:
            PPOTrainer instance
        """
        import os
        import json
        from transformers import AutoTokenizer
        from ..models import PolicyModel, ValueModel
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        
        # Load policy model
        policy_dir = os.path.join(model_directory, "policy_model")
        policy_model = PolicyModel.from_pretrained(policy_dir)
        
        # Load value model
        value_dir = os.path.join(model_directory, "value_model")
        value_model = ValueModel.from_pretrained(value_dir)
        
        # Load PPO config if available
        ppo_config_path = os.path.join(model_directory, 'ppo_config.json')
        if os.path.exists(ppo_config_path) and config is None:
            with open(ppo_config_path, 'r') as f:
                ppo_config_dict = json.load(f)
            
            # Create config with loaded parameters
            config = PPOTrainingConfig(
                model_name_or_path=model_directory,
                output_dir="./output",
                **ppo_config_dict,
                **kwargs
            )
        elif config is None:
            config = PPOTrainingConfig(
                model_name_or_path=model_directory,
                output_dir="./output",
                **kwargs
            )
        
        return cls(
            config=config,
            policy_model=policy_model,
            value_model=value_model,
            tokenizer=tokenizer,
            **kwargs
        )