"""Direct Preference Optimization (DPO) algorithm implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaseRLAlgorithm
from ..utils.metrics import compute_dpo_metrics


class DPOAlgorithm(BaseRLAlgorithm):
    """Direct Preference Optimization algorithm for language models"""
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: Dict[str, Any],
                 reference_model: Optional[PreTrainedModel] = None):
        super().__init__(model, tokenizer, config)
        
        # DPO-specific configuration
        self.dpo_config = self.algorithm_config.get('dpo', {})
        self.beta = self.dpo_config.get('beta', 0.1)  # Temperature parameter
        self.label_smoothing = self.dpo_config.get('label_smoothing', 0.0)
        self.loss_type = self.dpo_config.get('loss_type', 'sigmoid')  # sigmoid, hinge, ipo
        self.reference_free = self.dpo_config.get('reference_free', False)
        
        # Reference model (frozen copy of the initial model)
        if reference_model is not None:
            self.reference_model = reference_model
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
        else:
            self.reference_model = None
            
        self.use_reference_model = not self.reference_free and self.reference_model is not None
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss
        
        Args:
            batch: Batch containing chosen and rejected responses with their prompts
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Extract batch components
        prompt_input_ids = batch['prompt_input_ids']
        chosen_input_ids = batch['chosen_input_ids']
        rejected_input_ids = batch['rejected_input_ids']
        
        # Compute log probabilities for chosen and rejected responses
        chosen_logps = self._compute_sequence_logps(
            prompt_input_ids, chosen_input_ids
        )
        rejected_logps = self._compute_sequence_logps(
            prompt_input_ids, rejected_input_ids
        )
        
        # Compute reference log probabilities if using reference model
        if self.use_reference_model:
            with torch.no_grad():
                ref_chosen_logps = self._compute_sequence_logps(
                    prompt_input_ids, chosen_input_ids, use_reference=True
                )
                ref_rejected_logps = self._compute_sequence_logps(
                    prompt_input_ids, rejected_input_ids, use_reference=True
                )
        else:
            ref_chosen_logps = 0.0
            ref_rejected_logps = 0.0
        
        # Compute DPO loss
        if self.loss_type == 'sigmoid':
            loss = self._compute_sigmoid_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
        elif self.loss_type == 'hinge':
            loss = self._compute_hinge_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
        elif self.loss_type == 'ipo':
            loss = self._compute_ipo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute metrics
        metrics = compute_dpo_metrics(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, self.beta
        )
        
        metrics.update({
            'loss': loss.item(),
            'chosen_logps': chosen_logps.mean().item(),
            'rejected_logps': rejected_logps.mean().item()
        })
        
        if self.use_reference_model:
            metrics.update({
                'ref_chosen_logps': ref_chosen_logps.mean().item(),
                'ref_rejected_logps': ref_rejected_logps.mean().item()
            })
        
        return loss, metrics
    
    def _compute_sequence_logps(self,
                               prompt_input_ids: torch.Tensor,
                               response_input_ids: torch.Tensor,
                               use_reference: bool = False) -> torch.Tensor:
        """Compute log probabilities for a sequence
        
        Args:
            prompt_input_ids: Prompt token IDs
            response_input_ids: Response token IDs
            use_reference: Whether to use reference model
            
        Returns:
            Log probabilities for the response sequence
        """
        # Combine prompt and response
        full_input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
        attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()
        
        # Choose model
        model = self.reference_model if use_reference else self.model
        
        # Forward pass
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        
        # Compute log probabilities for response tokens only
        prompt_length = prompt_input_ids.shape[1]
        response_logits = logits[:, prompt_length-1:-1, :]  # Shift for next token prediction
        response_labels = response_input_ids
        
        # Compute log probabilities
        log_probs = F.log_softmax(response_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=response_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding tokens
        response_mask = (response_labels != self.tokenizer.pad_token_id).float()
        masked_log_probs = gathered_log_probs * response_mask
        
        # Sum log probabilities over sequence length
        sequence_log_probs = masked_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def _compute_sigmoid_loss(self,
                             chosen_logps: torch.Tensor,
                             rejected_logps: torch.Tensor,
                             ref_chosen_logps: torch.Tensor,
                             ref_rejected_logps: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid DPO loss
        
        Args:
            chosen_logps: Log probabilities for chosen responses
            rejected_logps: Log probabilities for rejected responses
            ref_chosen_logps: Reference log probabilities for chosen responses
            ref_rejected_logps: Reference log probabilities for rejected responses
            
        Returns:
            DPO loss
        """
        # Compute preference logits
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        
        # DPO loss with sigmoid
        logits = chosen_rewards - rejected_rewards
        
        if self.label_smoothing > 0:
            # Label smoothing
            loss = -F.logsigmoid(logits) * (1 - self.label_smoothing) - \
                   F.logsigmoid(-logits) * self.label_smoothing
        else:
            loss = -F.logsigmoid(logits)
        
        return loss.mean()
    
    def _compute_hinge_loss(self,
                           chosen_logps: torch.Tensor,
                           rejected_logps: torch.Tensor,
                           ref_chosen_logps: torch.Tensor,
                           ref_rejected_logps: torch.Tensor) -> torch.Tensor:
        """Compute hinge DPO loss
        
        Args:
            chosen_logps: Log probabilities for chosen responses
            rejected_logps: Log probabilities for rejected responses
            ref_chosen_logps: Reference log probabilities for chosen responses
            ref_rejected_logps: Reference log probabilities for rejected responses
            
        Returns:
            Hinge DPO loss
        """
        # Compute preference logits
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        
        # Hinge loss
        margin = 1.0
        loss = torch.clamp(margin - (chosen_rewards - rejected_rewards), min=0)
        
        return loss.mean()
    
    def _compute_ipo_loss(self,
                         chosen_logps: torch.Tensor,
                         rejected_logps: torch.Tensor,
                         ref_chosen_logps: torch.Tensor,
                         ref_rejected_logps: torch.Tensor) -> torch.Tensor:
        """Compute IPO (Identity Preference Optimization) loss
        
        Args:
            chosen_logps: Log probabilities for chosen responses
            rejected_logps: Log probabilities for rejected responses
            ref_chosen_logps: Reference log probabilities for chosen responses
            ref_rejected_logps: Reference log probabilities for rejected responses
            
        Returns:
            IPO loss
        """
        # Compute preference logits
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        
        # IPO loss (quadratic)
        diff = chosen_rewards - rejected_rewards
        loss = (diff - 1) ** 2
        
        return loss.mean()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one DPO training step
        
        Args:
            batch: Training batch
            
        Returns:
            Training metrics
        """
        batch = self.prepare_batch(batch)
        
        # Compute loss and metrics
        loss, metrics = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        self.update_step()
        return metrics
    
    def evaluate_preferences(self,
                           prompts: torch.Tensor,
                           chosen_responses: torch.Tensor,
                           rejected_responses: torch.Tensor) -> Dict[str, float]:
        """Evaluate preference accuracy
        
        Args:
            prompts: Prompt token IDs
            chosen_responses: Chosen response token IDs
            rejected_responses: Rejected response token IDs
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Compute log probabilities
            chosen_logps = self._compute_sequence_logps(prompts, chosen_responses)
            rejected_logps = self._compute_sequence_logps(prompts, rejected_responses)
            
            if self.use_reference_model:
                ref_chosen_logps = self._compute_sequence_logps(
                    prompts, chosen_responses, use_reference=True
                )
                ref_rejected_logps = self._compute_sequence_logps(
                    prompts, rejected_responses, use_reference=True
                )
            else:
                ref_chosen_logps = 0.0
                ref_rejected_logps = 0.0
            
            # Compute rewards
            chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
            
            # Compute accuracy
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            
            # Compute margin
            margin = (chosen_rewards - rejected_rewards).mean()
        
        self.model.train()
        
        return {
            'preference_accuracy': accuracy.item(),
            'reward_margin': margin.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item()
        }
    
    def compute_rewards(self,
                       prompts: torch.Tensor,
                       responses: torch.Tensor) -> torch.Tensor:
        """Compute DPO rewards for responses
        
        Args:
            prompts: Prompt token IDs
            responses: Response token IDs
            
        Returns:
            Reward scores
        """
        with torch.no_grad():
            # Compute log probabilities
            logps = self._compute_sequence_logps(prompts, responses)
            
            if self.use_reference_model:
                ref_logps = self._compute_sequence_logps(
                    prompts, responses, use_reference=True
                )
            else:
                ref_logps = 0.0
            
            # Compute rewards
            rewards = self.beta * (logps - ref_logps)
        
        return rewards
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save DPO checkpoint
        
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
        self.logger.info(f"Saved DPO checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load DPO checkpoint
        
        Args:
            checkpoint_path: Path to load checkpoint from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
        
        self.logger.info(f"Loaded DPO checkpoint from {checkpoint_path}")