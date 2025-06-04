"""Base trainer class for reinforcement learning training"""

import os
import time
import json
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler
)

from ..utils import (
    setup_logger,
    TensorBoardLogger,
    MetricsTracker,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint
)
from ..models import get_model_device, count_parameters


@dataclass
class TrainingConfig:
    """Base training configuration"""
    # Model and data
    model_name_or_path: str
    output_dir: str
    experiment_name: Optional[str] = None
    dataset_path: Optional[str] = None
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Scheduler
    lr_scheduler_type: str = "linear"
    
    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"  # "steps", "epoch", "no"
    save_strategy: str = "steps"  # "steps", "epoch", "no"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Logging
    logging_dir: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: Optional[str] = None
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Custom
    remove_unused_columns: bool = True
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    def __post_init__(self):
        # 如果设置了实验名称，则在output_dir下创建实验子目录
        if self.experiment_name:
            self.output_dir = os.path.join(self.output_dir, self.experiment_name)
        
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")


class BaseTrainer(ABC):
    """Base trainer class for reinforcement learning
    
    This class provides the common functionality for all RL trainers,
    including training loop, evaluation, checkpointing, and logging.
    """
    
    def __init__(self,
                 config: TrainingConfig,
                 model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: Optional[DataLoader] = None,
                 eval_dataloader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 callbacks: Optional[List[Callable]] = None):
        
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.early_stopping_counter = 0
        self.is_training = False
        
        # Setup device
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup logging - 保存到实验目录下
        experiment_name = getattr(config, 'experiment_name', None)
        if experiment_name:
            log_file_path = os.path.join(config.output_dir, experiment_name, "logs", "training.log")
        else:
            log_file_path = os.path.join(config.output_dir, "training.log")
        
        self.logger = setup_logger(
            name=self.__class__.__name__,
            log_file=log_file_path
        )
        
        # Setup TensorBoard logging
        if "tensorboard" in config.report_to:
            self.tb_logger = TensorBoardLogger(config.logging_dir)
        else:
            self.tb_logger = None
        
        # Setup metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Create output directory and logging directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)
        
        # 如果有实验名称，确保创建相应的子目录
        if experiment_name:
            eval_results_dir = os.path.join(config.output_dir, "eval_results")
            checkpoints_dir = os.path.join(config.output_dir, "checkpoints")
            os.makedirs(eval_results_dir, exist_ok=True)
            os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Set random seed
        self._set_seed(config.seed)
        
        # Log model info
        self._log_model_info()
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    def _log_model_info(self):
        """Log model information"""
        param_counts = count_parameters(self.model)
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Total parameters: {param_counts['total']:,}")
        self.logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
        self.logger.info(f"Device: {self.device}")
    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        if self.optimizer is not None:
            return self.optimizer
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        self.optimizer = optimizer
        return optimizer
    
    def setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler"""
        if self.scheduler is not None:
            return self.scheduler
        
        if self.optimizer is None:
            self.setup_optimizer()
        
        # Calculate total training steps
        if self.train_dataloader is not None:
            num_training_steps = (
                len(self.train_dataloader) * self.config.num_epochs
            ) // self.config.gradient_accumulation_steps
        else:
            num_training_steps = 1000  # Default fallback
        
        # Calculate warmup steps
        if self.config.warmup_steps > 0:
            num_warmup_steps = self.config.warmup_steps
        else:
            num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        # Create scheduler
        scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.scheduler = scheduler
        return scheduler
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model
        
        Returns:
            Evaluation metrics
        """
        pass
    
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step
        
        Args:
            batch: Input batch
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Compute loss
        loss, metrics = self.compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.fp16:
            with torch.cuda.amp.autocast():
                loss.backward()
        else:
            loss.backward()
        
        # Update metrics
        metrics['loss'] = loss.item() * self.config.gradient_accumulation_steps
        metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop
        
        Returns:
            Training history
        """
        if self.train_dataloader is None:
            raise ValueError("Training dataloader is required for training")
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        self.setup_scheduler()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._resume_from_checkpoint()
        
        self.logger.info("Starting training...")
        self.logger.info(f"Number of epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size per device: {self.config.per_device_train_batch_size}")
        self.logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        self.is_training = True
        training_history = []
        
        try:
            for epoch in range(self.epoch, self.config.num_epochs):
                self.epoch = epoch
                epoch_metrics = self._train_epoch()
                training_history.append(epoch_metrics)
                
                # Early stopping check
                if self._should_stop_early():
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        finally:
            self.is_training = False
            self.logger.info("Training completed")
            
            # Save final checkpoint
            self._save_checkpoint("final")
        
        return {
            'training_history': training_history,
            'final_metrics': training_history[-1] if training_history else {},
            'global_step': self.global_step,
            'epochs_completed': self.epoch + 1
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch
        
        Returns:
            Epoch metrics
        """
        epoch_start_time = time.time()
        epoch_metrics = {}
        
        self.model.train()
        
        for step, batch in enumerate(self.train_dataloader):
            # Training step
            step_metrics = self.training_step(batch)
            
            # Update metrics tracker
            self.metrics_tracker.update(step_metrics)
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(step_metrics)
                
                # Evaluation
                if (self.config.evaluation_strategy == "steps" and 
                    self.global_step % self.config.eval_steps == 0):
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, prefix="eval")
                    epoch_metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                
                # Saving
                if (self.config.save_strategy == "steps" and 
                    self.global_step % self.config.save_steps == 0):
                    self._save_checkpoint(f"step-{self.global_step}")
        
        # End of epoch evaluation
        if self.config.evaluation_strategy == "epoch":
            eval_metrics = self.evaluate()
            self._log_metrics(eval_metrics, prefix="eval")
            epoch_metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        
        # End of epoch saving
        if self.config.save_strategy == "epoch":
            self._save_checkpoint(f"epoch-{self.epoch}")
        
        # Compute epoch metrics
        train_metrics = self.metrics_tracker.get_average_metrics()
        epoch_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
        
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time
        epoch_metrics['epoch'] = self.epoch
        epoch_metrics['global_step'] = self.global_step
        
        self.logger.info(
            f"Epoch {self.epoch} completed in {epoch_time:.2f}s - "
            f"Train loss: {train_metrics.get('loss', 0):.4f}"
        )
        
        # Reset metrics tracker
        self.metrics_tracker.reset()
        
        return epoch_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics
        
        Args:
            metrics: Metrics to log
            prefix: Prefix for metric names
        """
        # Console logging
        metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {self.global_step} - {metric_str}")
        
        # TensorBoard logging
        if self.tb_logger is not None:
            for key, value in metrics.items():
                self.tb_logger.log_scalar(f"{prefix}/{key}", value, self.global_step)
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint
        
        Args:
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        
        # Get current metrics
        current_metrics = self.metrics_tracker.get_average_metrics()
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=current_metrics.get('loss', 0.0),
            metrics=current_metrics,
            checkpoint_dir=checkpoint_dir,
            filename=checkpoint_name
        )
        
        self.logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint"""
        if os.path.isdir(self.config.resume_from_checkpoint):
            # Find latest checkpoint in directory
            checkpoint_path = find_latest_checkpoint(self.config.resume_from_checkpoint)
        else:
            # Direct path to checkpoint file
            checkpoint_path = self.config.resume_from_checkpoint
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_info = load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                checkpoint_path=checkpoint_path
            )
            
            self.epoch = checkpoint_info['epoch']
            self.global_step = checkpoint_info['step']
            
            self.logger.info(
                f"Resumed from checkpoint: {checkpoint_path} "
                f"(epoch {self.epoch}, step {self.global_step})"
            )
        else:
            self.logger.warning(f"Checkpoint not found: {self.config.resume_from_checkpoint}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early
        
        Returns:
            True if training should stop
        """
        if self.config.early_stopping_patience <= 0:
            return False
        
        # Get current evaluation metric (assuming 'eval_loss' for now)
        current_metrics = self.metrics_tracker.get_average_metrics()
        current_metric = current_metrics.get('eval_loss')
        
        if current_metric is None:
            return False
        
        # Check if this is the best metric so far
        if (self.best_metric is None or 
            current_metric < self.best_metric - self.config.early_stopping_threshold):
            self.best_metric = current_metric
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def save_model(self, save_directory: str):
        """Save model and tokenizer
        
        Args:
            save_directory: Directory to save model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_directory)
        else:
            torch.save(self.model.state_dict(), 
                      os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save training config
        config_dict = {
            'model_name_or_path': self.config.model_name_or_path,
            'learning_rate': self.config.learning_rate,
            'num_epochs': self.config.num_epochs,
            'per_device_train_batch_size': self.config.per_device_train_batch_size,
            'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
            'max_grad_norm': self.config.max_grad_norm,
            'weight_decay': self.config.weight_decay,
            'warmup_steps': self.config.warmup_steps,
            'lr_scheduler_type': self.config.lr_scheduler_type
        }
        
        with open(os.path.join(save_directory, 'training_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Model saved to {save_directory}")
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state
        
        Returns:
            Training state dictionary
        """
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'early_stopping_counter': self.early_stopping_counter,
            'is_training': self.is_training,
            'metrics': self.metrics_tracker.get_average_metrics()
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'tb_logger') and self.tb_logger is not None:
            self.tb_logger.close()