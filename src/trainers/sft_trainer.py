"""Supervised Fine-Tuning (SFT) trainer implementation"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np

from .base_trainer import BaseTrainer, TrainingConfig
from ..models import PolicyModel


@dataclass
class SFTConfig(TrainingConfig):
    """SFT training configuration"""
    # Data processing
    max_length: int = 2048
    truncation_side: str = "right"  # "left", "right"
    padding_side: str = "right"  # "left", "right"
    
    # Training specific
    gradient_checkpointing: bool = False
    dataloader_drop_last: bool = True
    label_smoothing: float = 0.0
    
    # Loss computation
    ignore_index: int = -100
    loss_type: str = "cross_entropy"  # "cross_entropy", "focal"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Model specific
    freeze_base_model: bool = False
    freeze_layers: Optional[List[int]] = None
    
    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Generation for evaluation
    eval_generation: bool = True
    eval_max_new_tokens: int = 128
    eval_temperature: float = 1.0
    eval_top_k: int = 50
    eval_top_p: float = 0.95
    eval_do_sample: bool = True


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning trainer for language models
    
    This trainer implements standard supervised fine-tuning for language models
    using instruction-response pairs or conversational data.
    """
    
    def __init__(self,
                 config: SFTConfig,
                 model: PolicyModel,
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
        
        self.policy_model = model
        
        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()
        
        # Freeze base model if specified
        if config.freeze_base_model:
            self._freeze_base_model()
        
        # Freeze specific layers if specified
        if config.freeze_layers:
            self._freeze_layers(config.freeze_layers)
        
        # Training statistics
        self.stats = {
            'train_loss': [],
            'train_perplexity': [],
            'eval_loss': [],
            'eval_perplexity': [],
            'learning_rate': []
        }
    
    def _freeze_base_model(self):
        """Freeze base model parameters except the head"""
        # Freeze all parameters first
        for param in self.policy_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the language modeling head
        if hasattr(self.policy_model, 'lm_head'):
            for param in self.policy_model.lm_head.parameters():
                param.requires_grad = True
        elif hasattr(self.policy_model, 'head'):
            for param in self.policy_model.head.parameters():
                param.requires_grad = True
        
        self.logger.info("Frozen base model parameters (except head)")
    
    def _freeze_layers(self, layer_indices: List[int]):
        """Freeze specific transformer layers"""
        # Get transformer layers based on model architecture
        if hasattr(self.policy_model, 'transformer'):
            if hasattr(self.policy_model.transformer, 'h'):
                layers = self.policy_model.transformer.h  # GPT-style
            elif hasattr(self.policy_model.transformer, 'layers'):
                layers = self.policy_model.transformer.layers  # Some other architectures
            else:
                self.logger.warning("Could not find transformer layers to freeze")
                return
        elif hasattr(self.policy_model, 'model'):
            if hasattr(self.policy_model.model, 'layers'):
                layers = self.policy_model.model.layers  # LLaMA-style
            elif hasattr(self.policy_model.model, 'encoder'):
                layers = self.policy_model.model.encoder.layer  # BERT-style
            else:
                self.logger.warning("Could not find model layers to freeze")
                return
        else:
            self.logger.warning("Could not find transformer layers to freeze")
            return
        
        for i in layer_indices:
            if i < len(layers):
                for param in layers[i].parameters():
                    param.requires_grad = False
        
        self.logger.info(f"Frozen layers: {layer_indices}")
    
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SFT loss for a batch
        
        Args:
            batch: Input batch containing:
                - input_ids: Input token IDs
                - attention_mask: Attention mask
                - labels: Target token IDs (with ignore_index for non-target tokens)
                
        Returns:
            Tuple of (loss, metrics)
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        # Get logits and compute custom loss if needed
        if self.config.loss_type == "cross_entropy":
            loss = outputs.loss
        elif self.config.loss_type == "focal":
            logits = outputs.logits
            loss = self._compute_focal_loss(logits, labels)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Compute metrics
        with torch.no_grad():
            # Compute perplexity
            perplexity = torch.exp(loss) if loss < 10 else torch.tensor(float('inf'))
            
            # Compute accuracy (only for non-ignored tokens)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Mask for non-ignored tokens
            mask = (labels != self.config.ignore_index)
            
            if mask.sum() > 0:
                accuracy = (predictions == labels)[mask].float().mean()
            else:
                accuracy = torch.tensor(0.0)
        
        metrics = {
            'loss': loss.item(),
            'perplexity': perplexity.item(),
            'accuracy': accuracy.item()
        }
        
        return loss, metrics
    
    def _compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for handling class imbalance
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            
        Returns:
            Focal loss
        """
        # Flatten logits and labels
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits, 
            labels, 
            ignore_index=self.config.ignore_index,
            reduction='none'
        )
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.config.focal_alpha * (1 - pt) ** self.config.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
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
        if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        # Update statistics
        self.stats['train_loss'].append(metrics['loss'])
        self.stats['train_perplexity'].append(metrics['perplexity'])
        if 'learning_rate' in metrics:
            self.stats['learning_rate'].append(metrics['learning_rate'])
        
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
        
        # Update statistics
        if 'loss' in total_metrics:
            self.stats['eval_loss'].append(total_metrics['loss'])
        if 'perplexity' in total_metrics:
            self.stats['eval_perplexity'].append(total_metrics['perplexity'])
        
        # Add eval prefix
        eval_metrics = {f'eval_{key}': value for key, value in total_metrics.items()}
        
        # Generate samples for evaluation if enabled
        if self.config.eval_generation:
            generation_metrics = self._evaluate_generation()
            eval_metrics.update(generation_metrics)
        
        return eval_metrics
    
    def _evaluate_generation(self) -> Dict[str, Any]:
        """Evaluate model generation quality
        
        Returns:
            Generation evaluation metrics
        """
        if not hasattr(self, '_eval_prompts'):
            # Use some sample prompts for generation evaluation
            self._eval_prompts = [
                "What is the capital of France?",
                "Explain the concept of machine learning.",
                "Write a short story about a robot."
            ]
        
        self.policy_model.eval()
        generated_texts = []
        
        with torch.no_grad():
            for prompt in self._eval_prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Generate response
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.config.eval_max_new_tokens,
                    temperature=self.config.eval_temperature,
                    top_k=self.config.eval_top_k,
                    top_p=self.config.eval_top_p,
                    do_sample=self.config.eval_do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode generated text
                prompt_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][prompt_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        # Compute generation metrics
        metrics = {
            'eval_avg_generation_length': np.mean([len(text.split()) for text in generated_texts]),
            'eval_generation_samples': generated_texts[:3]  # Store first 3 samples
        }
        
        return metrics
    
    def generate_text(self,
                     prompts: List[str],
                     max_new_tokens: Optional[int] = None,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     do_sample: Optional[bool] = None) -> List[str]:
        """Generate text for given prompts
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            List of generated texts
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.eval_max_new_tokens
        temperature = temperature or self.config.eval_temperature
        top_k = top_k or self.config.eval_top_k
        top_p = top_p or self.config.eval_top_p
        do_sample = do_sample if do_sample is not None else self.config.eval_do_sample
        
        self.policy_model.eval()
        generated_texts = []
        
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
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode generated text
                prompt_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][prompt_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        return generated_texts
    
    def save_model(self, save_directory: str):
        """Save the model
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        self.policy_model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        self.logger.info(f"SFT model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls,
                       model_directory: str,
                       config: Optional[SFTConfig] = None,
                       **kwargs):
        """Load SFT trainer from directory
        
        Args:
            model_directory: Directory containing the saved model
            config: Training configuration
            **kwargs: Additional arguments
            
        Returns:
            SFTTrainer instance
        """
        from ..models import PolicyModel
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        policy_model = PolicyModel.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        
        # Create default config if not provided
        if config is None:
            config = SFTConfig(
                model_name_or_path=model_directory,
                output_dir="./outputs"
            )
        
        return cls(
            config=config,
            model=policy_model,
            tokenizer=tokenizer,
            **kwargs
        )
    
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
                stats[f'{key}_latest'] = values[-1]
        
        return stats