"""Reward model trainer implementation"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer, TrainingConfig
from ..models import RewardModel
from .trainer_utils import compute_ranking_loss


@dataclass
class RewardTrainingConfig(TrainingConfig):
    """Reward model training configuration"""
    # Reward model specific hyperparameters
    margin: float = 0.0
    loss_type: str = "ranking"  # "ranking", "regression", "classification"
    ranking_loss_type: str = "hinge"  # "hinge", "log_sigmoid", "cross_entropy"
    label_smoothing: float = 0.0
    
    # Data processing
    max_length: int = 2048
    truncation_side: str = "right"  # "left", "right"
    
    # Training specific
    gradient_checkpointing: bool = False
    dataloader_drop_last: bool = True
    
    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Model specific
    freeze_base_model: bool = False
    freeze_layers: Optional[List[int]] = None


class RewardModelTrainer(BaseTrainer):
    """Reward model trainer for RLHF training
    
    This trainer implements reward model training using preference data.
    The reward model learns to score text sequences based on human preferences.
    """
    
    def __init__(self,
                 config: RewardTrainingConfig,
                 model: RewardModel,
                 tokenizer,
                 train_dataloader: Optional[DataLoader] = None,
                 eval_dataloader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 callbacks: Optional[List] = None):
        
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
        
        self.reward_model = model
        
        # Freeze base model if specified
        if config.freeze_base_model:
            self._freeze_base_model()
        
        # Freeze specific layers if specified
        if config.freeze_layers:
            self._freeze_layers(config.freeze_layers)
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.reward_model.base_model.parameters():
            param.requires_grad = False
        
        self.logger.info("Frozen base model parameters")
    
    def _freeze_layers(self, layer_indices: List[int]):
        """Freeze specific transformer layers"""
        if hasattr(self.reward_model.base_model, 'encoder'):
            layers = self.reward_model.base_model.encoder.layer
        elif hasattr(self.reward_model.base_model, 'layers'):
            layers = self.reward_model.base_model.layers
        elif hasattr(self.reward_model.base_model, 'h'):
            layers = self.reward_model.base_model.h
        else:
            self.logger.warning("Could not find transformer layers to freeze")
            return
        
        for i in layer_indices:
            if i < len(layers):
                for param in layers[i].parameters():
                    param.requires_grad = False
        
        self.logger.info(f"Frozen layers: {layer_indices}")
    
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute reward model loss for a batch
        
        Args:
            batch: Input batch containing:
                - chosen_input_ids: Preferred response token IDs
                - chosen_attention_mask: Preferred response attention mask
                - rejected_input_ids: Rejected response token IDs
                - rejected_attention_mask: Rejected response attention mask
                - (optional) labels: Ground truth reward scores
                
        Returns:
            Tuple of (loss, metrics)
        """
        if self.config.loss_type == "ranking":
            return self._compute_ranking_loss(batch)
        elif self.config.loss_type == "regression":
            return self._compute_regression_loss(batch)
        elif self.config.loss_type == "classification":
            return self._compute_classification_loss(batch)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def _compute_ranking_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute ranking loss for preference data"""
        # Get chosen and rejected sequences
        chosen_input_ids = batch['chosen_input_ids']
        chosen_attention_mask = batch['chosen_attention_mask']
        rejected_input_ids = batch['rejected_input_ids']
        rejected_attention_mask = batch['rejected_attention_mask']
        
        # Forward pass for chosen sequences
        chosen_outputs = self.reward_model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            return_dict=True
        )
        chosen_rewards = chosen_outputs['rewards'].squeeze(-1)
        
        # Forward pass for rejected sequences
        rejected_outputs = self.reward_model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            return_dict=True
        )
        rejected_rewards = rejected_outputs['rewards'].squeeze(-1)
        
        # Compute ranking loss
        if self.config.ranking_loss_type == "hinge":
            # Hinge loss: max(0, margin - (chosen_reward - rejected_reward))
            loss = F.relu(self.config.margin - (chosen_rewards - rejected_rewards))
        elif self.config.ranking_loss_type == "log_sigmoid":
            # Log-sigmoid loss: -log(sigmoid(chosen_reward - rejected_reward))
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
        elif self.config.ranking_loss_type == "cross_entropy":
            # Cross-entropy loss with softmax
            logits = torch.stack([rejected_rewards, chosen_rewards], dim=1)
            targets = torch.ones(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, targets, label_smoothing=self.config.label_smoothing)
        else:
            raise ValueError(f"Unknown ranking loss type: {self.config.ranking_loss_type}")
        
        loss = loss.mean()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_diff = (chosen_rewards - rejected_rewards).mean()
            chosen_reward_mean = chosen_rewards.mean()
            rejected_reward_mean = rejected_rewards.mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'reward_diff': reward_diff.item(),
            'chosen_reward_mean': chosen_reward_mean.item(),
            'rejected_reward_mean': rejected_reward_mean.item()
        }
        
        return loss, metrics
    
    def _compute_regression_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute regression loss for labeled reward data"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].float()
        
        # Forward pass
        outputs = self.reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        rewards = outputs['rewards'].squeeze(-1)
        
        # Compute MSE loss
        loss = F.mse_loss(rewards, labels)
        
        # Compute metrics
        with torch.no_grad():
            mae = F.l1_loss(rewards, labels)
            correlation = torch.corrcoef(torch.stack([rewards, labels]))[0, 1]
        
        metrics = {
            'loss': loss.item(),
            'mse': loss.item(),
            'mae': mae.item(),
            'correlation': correlation.item() if not torch.isnan(correlation) else 0.0,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item()
        }
        
        return loss, metrics
    
    def _compute_classification_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute classification loss for categorical reward data"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].long()
        
        # Forward pass
        outputs = self.reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs['rewards']  # Should be (batch_size, num_classes)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels, label_smoothing=self.config.label_smoothing)
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
        
        return loss, metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the reward model
        
        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.reward_model.eval()
        
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
    
    def predict_rewards(self,
                       texts: List[str],
                       batch_size: int = 8) -> List[float]:
        """Predict reward scores for given texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for prediction
            
        Returns:
            List of reward scores
        """
        self.reward_model.eval()
        rewards = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.reward_model(**inputs, return_dict=True)
                batch_rewards = outputs['rewards'].squeeze(-1)
                
                rewards.extend(batch_rewards.cpu().tolist())
        
        return rewards
    
    def save_model(self, save_directory: str):
        """Save reward model
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        self.reward_model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        self.logger.info(f"Reward model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls,
                       model_directory: str,
                       config: Optional[RewardTrainingConfig] = None,
                       **kwargs):
        """Load reward model trainer from directory
        
        Args:
            model_directory: Directory containing the saved model
            config: Training configuration
            **kwargs: Additional arguments
            
        Returns:
            RewardModelTrainer instance
        """
        from ..models import RewardModel
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        reward_model = RewardModel.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        
        # Create default config if not provided
        if config is None:
            config = RewardTrainingConfig(
                model_name_or_path=model_directory,
                output_dir="./outputs"
            )
        
        return cls(
            config=config,
            model=reward_model,
            tokenizer=tokenizer,
            **kwargs
        )