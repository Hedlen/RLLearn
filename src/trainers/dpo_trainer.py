"""DPO trainer implementation"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer, TrainingConfig
from ..algorithms import DPOAlgorithm
from ..models import PolicyModel


@dataclass
class DPOTrainingConfig(TrainingConfig):
    """DPO-specific training configuration"""
    # DPO hyperparameters
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # "sigmoid", "hinge", "ipo"
    
    # Reference model
    reference_model_path: Optional[str] = None
    reference_free: bool = False
    
    # Generation parameters (for evaluation)
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    
    # Training parameters
    max_length: int = 2048
    max_prompt_length: int = 1024
    

class DPOTrainer(BaseTrainer):
    """DPO trainer for direct preference optimization
    
    This trainer implements the Direct Preference Optimization algorithm
    for training language models directly on preference data without
    requiring a separate reward model.
    """
    
    def __init__(self,
                 config: DPOTrainingConfig,
                 model: PolicyModel,
                 tokenizer,
                 train_dataloader: Optional[DataLoader] = None,
                 eval_dataloader: Optional[DataLoader] = None,
                 reference_model: Optional[PolicyModel] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 callbacks: Optional[List] = None):
        
        # Initialize DPO algorithm
        self.dpo_algorithm = DPOAlgorithm(
            model=model,
            reference_model=reference_model,
            config={
                'beta': config.beta,
                'label_smoothing': config.label_smoothing,
                'loss_type': config.loss_type,
                'reference_free': config.reference_free
            }
        )
        
        # Initialize base trainer
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks
        )
        
        self.reference_model = reference_model
        
        # Move reference model to device
        if self.reference_model is not None:
            self.reference_model = self.reference_model.to(self.device)
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss for a batch
        
        Args:
            batch: Input batch containing:
                - prompt_ids: Prompt token IDs
                - chosen_ids: Chosen response token IDs
                - rejected_ids: Rejected response token IDs
                - attention_mask: Attention mask (optional)
                
        Returns:
            Tuple of (loss, metrics)
        """
        # Extract batch data
        prompt_ids = batch['prompt_ids']
        chosen_ids = batch['chosen_ids']
        rejected_ids = batch['rejected_ids']
        attention_mask = batch.get('attention_mask')
        
        # Combine prompts with responses
        chosen_input_ids = torch.cat([prompt_ids, chosen_ids], dim=1)
        rejected_input_ids = torch.cat([prompt_ids, rejected_ids], dim=1)
        
        # Create attention masks if not provided
        if attention_mask is None:
            prompt_mask = (prompt_ids != self.tokenizer.pad_token_id).long()
        else:
            prompt_mask = attention_mask
        
        chosen_mask = (chosen_ids != self.tokenizer.pad_token_id).long()
        rejected_mask = (rejected_ids != self.tokenizer.pad_token_id).long()
        
        chosen_attention_mask = torch.cat([prompt_mask, chosen_mask], dim=1)
        rejected_attention_mask = torch.cat([prompt_mask, rejected_mask], dim=1)
        
        # Compute DPO loss
        loss, metrics = self.dpo_algorithm.compute_loss(
            prompt_ids=prompt_ids,
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids,
            chosen_attention_mask=chosen_attention_mask,
            rejected_attention_mask=rejected_attention_mask
        )
        
        # Add additional metrics
        metrics.update({
            'chosen_length': chosen_ids.shape[1],
            'rejected_length': rejected_ids.shape[1],
            'prompt_length': prompt_ids.shape[1]
        })
        
        return loss, metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model
        
        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        if self.reference_model is not None:
            self.reference_model.eval()
        
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
        
        # Compute preference accuracy
        if 'chosen_rewards' in avg_metrics and 'rejected_rewards' in avg_metrics:
            avg_metrics['preference_accuracy'] = (
                avg_metrics['chosen_rewards'] > avg_metrics['rejected_rewards']
            ).float().mean().item()
        
        # Set model back to training mode
        self.model.train()
        
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
        self.model.eval()
        
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
                    truncation=True,
                    max_length=self.config.max_prompt_length
                ).to(self.device)
                
                # Generate response
                generated = self.model.generate(
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
        
        self.model.train()
        return responses
    
    def compute_preference_metrics(self,
                                  prompts: List[str],
                                  chosen_responses: List[str],
                                  rejected_responses: List[str]) -> Dict[str, float]:
        """Compute preference-specific metrics
        
        Args:
            prompts: List of prompts
            chosen_responses: List of chosen responses
            rejected_responses: List of rejected responses
            
        Returns:
            Preference metrics
        """
        metrics = {}
        
        # Tokenize inputs
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length
        ).to(self.device)
        
        chosen_inputs = self.tokenizer(
            chosen_responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length - self.config.max_prompt_length
        ).to(self.device)
        
        rejected_inputs = self.tokenizer(
            rejected_responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length - self.config.max_prompt_length
        ).to(self.device)
        
        with torch.no_grad():
            # Compute log probabilities
            chosen_log_probs = self.model.compute_log_probs(
                torch.cat([prompt_inputs['input_ids'], chosen_inputs['input_ids']], dim=1),
                target_ids=chosen_inputs['input_ids']
            )
            
            rejected_log_probs = self.model.compute_log_probs(
                torch.cat([prompt_inputs['input_ids'], rejected_inputs['input_ids']], dim=1),
                target_ids=rejected_inputs['input_ids']
            )
            
            # Compute reference log probabilities if reference model exists
            if self.reference_model is not None:
                chosen_ref_log_probs = self.reference_model.compute_log_probs(
                    torch.cat([prompt_inputs['input_ids'], chosen_inputs['input_ids']], dim=1),
                    target_ids=chosen_inputs['input_ids']
                )
                
                rejected_ref_log_probs = self.reference_model.compute_log_probs(
                    torch.cat([prompt_inputs['input_ids'], rejected_inputs['input_ids']], dim=1),
                    target_ids=rejected_inputs['input_ids']
                )
                
                # Compute KL divergences
                chosen_kl = chosen_ref_log_probs - chosen_log_probs
                rejected_kl = rejected_ref_log_probs - rejected_log_probs
                
                metrics.update({
                    'chosen_kl_divergence': chosen_kl.mean().item(),
                    'rejected_kl_divergence': rejected_kl.mean().item(),
                    'kl_divergence_diff': (chosen_kl - rejected_kl).mean().item()
                })
            
            # Compute implicit rewards
            chosen_rewards = self.config.beta * chosen_log_probs
            rejected_rewards = self.config.beta * rejected_log_probs
            
            if self.reference_model is not None:
                chosen_rewards += chosen_ref_log_probs
                rejected_rewards += rejected_ref_log_probs
            
            # Compute preference accuracy
            preference_accuracy = (chosen_rewards > rejected_rewards).float().mean()
            
            metrics.update({
                'chosen_log_probs': chosen_log_probs.mean().item(),
                'rejected_log_probs': rejected_log_probs.mean().item(),
                'log_probs_diff': (chosen_log_probs - rejected_log_probs).mean().item(),
                'chosen_rewards': chosen_rewards.mean().item(),
                'rejected_rewards': rejected_rewards.mean().item(),
                'reward_diff': (chosen_rewards - rejected_rewards).mean().item(),
                'preference_accuracy': preference_accuracy.item(),
                'chosen_length_mean': chosen_inputs['input_ids'].shape[1],
                'rejected_length_mean': rejected_inputs['input_ids'].shape[1]
            })
        
        return metrics
    
    def compute_win_rate(self,
                        prompts: List[str],
                        responses_a: List[str],
                        responses_b: List[str]) -> float:
        """Compute win rate of responses A vs responses B
        
        Args:
            prompts: List of prompts
            responses_a: List of responses A
            responses_b: List of responses B
            
        Returns:
            Win rate of A vs B (0.0 to 1.0)
        """
        metrics = self.compute_preference_metrics(prompts, responses_a, responses_b)
        return metrics.get('preference_accuracy', 0.5)
    
    def save_model(self, save_directory: str):
        """Save model and configuration
        
        Args:
            save_directory: Directory to save model
        """
        import os
        import json
        
        # Save model
        self.model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save DPO config
        config_dict = {
            'beta': self.config.beta,
            'label_smoothing': self.config.label_smoothing,
            'loss_type': self.config.loss_type,
            'reference_model_path': self.config.reference_model_path,
            'reference_free': self.config.reference_free,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'top_k': self.config.top_k,
            'top_p': self.config.top_p,
            'max_length': self.config.max_length,
            'max_prompt_length': self.config.max_prompt_length
        }
        
        with open(os.path.join(save_directory, 'dpo_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"DPO model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls,
                       model_directory: str,
                       config: Optional[DPOTrainingConfig] = None,
                       reference_model_path: Optional[str] = None,
                       **kwargs):
        """Load DPO trainer from pretrained model
        
        Args:
            model_directory: Directory containing saved model
            config: Training configuration
            reference_model_path: Path to reference model
            **kwargs: Additional arguments
            
        Returns:
            DPOTrainer instance
        """
        import os
        import json
        from transformers import AutoTokenizer
        from ..models import PolicyModel
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        
        # Load model
        model = PolicyModel.from_pretrained(model_directory)
        
        # Load reference model if specified
        reference_model = None
        if reference_model_path:
            reference_model = PolicyModel.from_pretrained(reference_model_path)
        
        # Load DPO config if available
        dpo_config_path = os.path.join(model_directory, 'dpo_config.json')
        if os.path.exists(dpo_config_path) and config is None:
            with open(dpo_config_path, 'r') as f:
                dpo_config_dict = json.load(f)
            
            # Create config with loaded parameters
            config = DPOTrainingConfig(
                model_name_or_path=model_directory,
                output_dir="./output",
                **dpo_config_dict,
                **kwargs
            )
        elif config is None:
            config = DPOTrainingConfig(
                model_name_or_path=model_directory,
                output_dir="./output",
                **kwargs
            )
        
        return cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            reference_model=reference_model,
            **kwargs
        )
    
    def create_preference_dataset(self,
                                 prompts: List[str],
                                 responses: List[List[str]],
                                 preferences: List[int]) -> List[Dict[str, str]]:
        """Create preference dataset from prompts, responses, and preferences
        
        Args:
            prompts: List of prompts
            responses: List of response pairs for each prompt
            preferences: List of preferred response indices (0 or 1)
            
        Returns:
            List of preference examples
        """
        dataset = []
        
        for prompt, response_pair, preference in zip(prompts, responses, preferences):
            if len(response_pair) != 2:
                raise ValueError("Each response pair must contain exactly 2 responses")
            
            chosen = response_pair[preference]
            rejected = response_pair[1 - preference]
            
            dataset.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })
        
        return dataset