"""Group Relative Policy Optimization (GRPO) trainer implementation"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np

from .base_trainer import BaseTrainer, TrainingConfig
from ..models import PolicyModel, RewardModel
from ..algorithms.utils import compute_advantages
from .trainer_utils import whiten


@dataclass
class GRPOConfig(TrainingConfig):
    """GRPO training configuration"""
    # GRPO specific hyperparameters
    group_size: int = 4  # Number of responses per prompt
    beta: float = 0.1  # KL penalty coefficient
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    
    # Policy optimization
    clip_range: float = 0.2  # PPO clipping range
    clip_range_vf: Optional[float] = None  # Value function clipping range
    vf_coef: float = 0.5  # Value function loss coefficient
    ent_coef: float = 0.01  # Entropy loss coefficient
    
    # Training parameters
    ppo_epochs: int = 4  # Number of PPO epochs per batch
    mini_batch_size: int = 1  # Mini-batch size for PPO updates
    max_grad_norm: float = 1.0  # Gradient clipping norm
    
    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    
    # Reward and value estimation
    normalize_rewards: bool = True
    normalize_advantages: bool = True
    use_baseline: bool = True  # Use value function as baseline
    
    # Logging and evaluation
    log_with_wandb: bool = False
    eval_steps: int = 500
    save_steps: int = 1000


class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization trainer
    
    GRPO is a variant of PPO that optimizes policies using group-wise comparisons.
    It generates multiple responses per prompt and uses relative rankings to compute
    advantages, making it more stable than standard RLHF approaches.
    """
    
    def __init__(self,
                 config: GRPOConfig,
                 policy_model: PolicyModel,
                 reward_model: Optional[RewardModel] = None,
                 reward_function: Optional[callable] = None,
                 value_model: Optional[PolicyModel] = None,
                 tokenizer=None,
                 train_dataloader: Optional[DataLoader] = None,
                 eval_dataloader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 callbacks: Optional[List] = None):
        
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
        self.reward_model = reward_model
        self.reward_function = reward_function
        self.value_model = value_model or policy_model  # Use policy model as value model if not provided
        
        # Validate reward configuration
        if self.reward_model is None and self.reward_function is None:
            self.logger.warning("No reward model or reward function provided. Using dummy rewards (all zeros).")
        elif self.reward_model is not None and self.reward_function is not None:
            self.logger.warning("Both reward model and reward function provided. Using reward function.")
            self.reward_model = None  # Prioritize reward function
        
        # Initialize reference model (frozen copy of policy model)
        self.ref_model = self._create_reference_model()
        
        # Training statistics
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'kl_divergence': [],
            'rewards': [],
            'advantages': []
        }
    
    def _create_reference_model(self) -> PolicyModel:
        """Create a frozen reference model for KL penalty computation"""
        ref_model = type(self.policy_model)(self.policy_model.config)
        ref_model.load_state_dict(self.policy_model.state_dict())
        
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        
        ref_model.eval()
        return ref_model
    
    def generate_responses(self,
                          prompts: List[str],
                          num_responses_per_prompt: Optional[int] = None) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """Generate multiple responses for each prompt
        
        Args:
            prompts: List of prompt strings
            num_responses_per_prompt: Number of responses to generate per prompt
            
        Returns:
            Tuple of (responses, response_tokens, log_probs)
        """
        if num_responses_per_prompt is None:
            num_responses_per_prompt = self.config.group_size
        
        self.policy_model.eval()
        
        all_responses = []
        all_response_tokens = []
        all_log_probs = []
        
        with torch.no_grad():
            for prompt in prompts:
                prompt_responses = []
                prompt_tokens = []
                prompt_log_probs = []
                
                for _ in range(num_responses_per_prompt):
                    # Tokenize prompt
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.policy_model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_k=self.config.top_k,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                    
                    # Extract generated tokens (excluding prompt)
                    prompt_length = inputs['input_ids'].shape[1]
                    response_tokens = outputs.sequences[0][prompt_length:]
                    
                    # Decode response
                    response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    
                    # Compute log probabilities
                    log_probs = self._compute_log_probs(
                        outputs.sequences[0].unsqueeze(0),
                        outputs.scores,
                        prompt_length
                    )
                    
                    prompt_responses.append(response)
                    prompt_tokens.append(response_tokens)
                    prompt_log_probs.append(log_probs)
                
                all_responses.extend(prompt_responses)
                all_response_tokens.extend(prompt_tokens)
                all_log_probs.extend(prompt_log_probs)
        
        return all_responses, all_response_tokens, all_log_probs
    
    def _compute_log_probs(self,
                          sequences: torch.Tensor,
                          scores: List[torch.Tensor],
                          prompt_length: int) -> torch.Tensor:
        """Compute log probabilities for generated tokens"""
        log_probs = []
        
        for i, score in enumerate(scores):
            token_id = sequences[0, prompt_length + i]
            log_prob = F.log_softmax(score[0], dim=-1)[token_id]
            log_probs.append(log_prob)
        
        return torch.stack(log_probs)
    
    def compute_rewards(self,
                       prompts: List[str],
                       responses: List[str]) -> List[float]:
        """Compute reward scores for prompt-response pairs
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            List of reward scores
        """
        # Priority: reward_function > reward_model > dummy rewards
        if self.reward_function is not None:
            # Use custom reward function
            return self._compute_rewards_with_function(prompts, responses)
        elif self.reward_model is not None:
            # Use reward model
            return self._compute_rewards_with_model(prompts, responses)
        else:
            # Use dummy rewards if no reward source is provided
            return [0.0] * len(responses)
    
    def _compute_rewards_with_function(self,
                                      prompts: List[str],
                                      responses: List[str]) -> List[float]:
        """Compute rewards using custom reward function
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            List of reward scores
        """
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            try:
                # Call custom reward function
                reward = self.reward_function(prompt, response)
                
                # Ensure reward is a float
                if isinstance(reward, (int, float)):
                    rewards.append(float(reward))
                elif isinstance(reward, torch.Tensor):
                    rewards.append(reward.item())
                else:
                    self.logger.warning(f"Invalid reward type {type(reward)}, using 0.0")
                    rewards.append(0.0)
                    
            except Exception as e:
                self.logger.error(f"Error in reward function: {e}, using 0.0")
                rewards.append(0.0)
        
        return rewards
    
    def _compute_rewards_with_model(self,
                                   prompts: List[str],
                                   responses: List[str]) -> List[float]:
        """Compute rewards using reward model
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            List of reward scores
        """
        self.reward_model.eval()
        rewards = []
        
        with torch.no_grad():
            for prompt, response in zip(prompts, responses):
                # Combine prompt and response
                full_text = f"{prompt}{response}"
                
                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # Get reward
                outputs = self.reward_model(**inputs, return_dict=True)
                reward = outputs['rewards'].squeeze().item()
                rewards.append(reward)
        
        return rewards
    
    def compute_group_advantages(self,
                                rewards: List[float],
                                group_size: int) -> List[float]:
        """Compute group-relative advantages
        
        Args:
            rewards: List of reward scores
            group_size: Number of responses per group
            
        Returns:
            List of advantages
        """
        advantages = []
        
        for i in range(0, len(rewards), group_size):
            group_rewards = rewards[i:i + group_size]
            
            if self.config.normalize_rewards:
                # Normalize within group
                group_rewards = np.array(group_rewards)
                if len(group_rewards) > 1 and group_rewards.std() > 0:
                    group_rewards = (group_rewards - group_rewards.mean()) / group_rewards.std()
                else:
                    group_rewards = group_rewards - group_rewards.mean()
            
            advantages.extend(group_rewards.tolist())
        
        return advantages
    
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss for a batch
        
        Args:
            batch: Input batch containing prompts and responses
            
        Returns:
            Tuple of (loss, metrics)
        """
        prompts = batch['prompts']
        
        # Generate responses
        responses, response_tokens, old_log_probs = self.generate_responses(prompts)
        
        # Compute rewards
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.config.group_size)
        
        rewards = self.compute_rewards(expanded_prompts, responses)
        
        # Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards, self.config.group_size)
        
        # Convert to tensors
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        
        if self.config.normalize_advantages:
            advantages = whiten(advantages)
        
        # Compute new log probabilities
        new_log_probs = self._compute_new_log_probs(expanded_prompts, responses)
        
        # Compute policy loss (PPO-style)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute KL divergence penalty
        kl_div = (old_log_probs - new_log_probs).mean()
        kl_penalty = self.config.beta * kl_div
        
        # Compute entropy bonus
        entropy = -new_log_probs.mean()
        entropy_bonus = self.config.ent_coef * entropy
        
        # Total loss
        total_loss = policy_loss + kl_penalty - entropy_bonus
        
        # Compute metrics
        metrics = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_div.item(),
            'entropy': entropy.item(),
            'mean_reward': np.mean(rewards),
            'mean_advantage': advantages.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item()
        }
        
        # Update statistics
        self.stats['policy_loss'].append(policy_loss.item())
        self.stats['kl_divergence'].append(kl_div.item())
        self.stats['entropy_loss'].append(entropy.item())
        self.stats['rewards'].extend(rewards)
        self.stats['advantages'].extend(advantages.cpu().tolist())
        
        return total_loss, metrics
    
    def _compute_new_log_probs(self,
                              prompts: List[str],
                              responses: List[str]) -> torch.Tensor:
        """Compute log probabilities for responses using current policy"""
        self.policy_model.train()
        
        all_log_probs = []
        
        for prompt, response in zip(prompts, responses):
            # Tokenize prompt and response
            prompt_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            full_inputs = self.tokenizer(
                prompt + response,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Forward pass
            outputs = self.policy_model(**full_inputs, return_dict=True)
            logits = outputs.logits
            
            # Extract response logits
            prompt_length = prompt_inputs['input_ids'].shape[1]
            response_logits = logits[0, prompt_length-1:-1]  # Shift by 1 for next token prediction
            response_tokens = full_inputs['input_ids'][0, prompt_length:]
            
            # Compute log probabilities
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
            
            all_log_probs.append(token_log_probs.sum())
        
        return torch.stack(all_log_probs)
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step
        
        Args:
            batch: Training batch
            
        Returns:
            Training metrics
        """
        self.policy_model.train()
        
        # Compute loss
        loss, metrics = self.compute_loss(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model
        
        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.policy_model.eval()
        
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
                
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            for key in total_metrics:
                total_metrics[key] /= num_batches
        
        # Add eval prefix
        eval_metrics = {f'eval_{key}': value for key, value in total_metrics.items()}
        
        return eval_metrics
    
    def save_model(self, save_directory: str):
        """Save the policy model
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save policy model
        self.policy_model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        self.logger.info(f"GRPO model saved to {save_directory}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics
        
        Returns:
            Dictionary of training statistics
        """
        stats = {}
        
        for key, values in self.stats.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
        
        return stats