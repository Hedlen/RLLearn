"""DeepSpeed utilities for distributed training"""

import os
import json
import torch
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

from .distributed import get_rank, get_world_size, is_main_process


def is_deepspeed_available() -> bool:
    """Check if DeepSpeed is available"""
    return DEEPSPEED_AVAILABLE


def create_deepspeed_config(
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    cpu_offload: bool = False,
    nvme_offload: bool = False,
    nvme_path: str = "/tmp",
    pin_memory: bool = True,
    train_batch_size: Optional[int] = None,
    train_micro_batch_size_per_gpu: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
    fp16_enabled: bool = False,
    bf16_enabled: bool = False,
    gradient_clipping: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """Create DeepSpeed configuration dictionary
    
    Args:
        zero_stage: ZeRO optimization stage (0, 1, 2, 3)
        offload_optimizer: Whether to offload optimizer states
        offload_param: Whether to offload parameters
        cpu_offload: Whether to use CPU offload
        nvme_offload: Whether to use NVMe offload
        nvme_path: Path for NVMe offload
        pin_memory: Whether to pin memory
        train_batch_size: Total training batch size
        train_micro_batch_size_per_gpu: Micro batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        fp16_enabled: Whether to enable FP16
        bf16_enabled: Whether to enable BF16
        gradient_clipping: Gradient clipping value
        **kwargs: Additional configuration options
    
    Returns:
        DeepSpeed configuration dictionary
    """
    config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    
    # ZeRO optimization configuration
    if zero_stage > 0:
        zero_config = {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        }
        
        # Stage 2 and 3 specific configurations
        if zero_stage >= 2:
            zero_config.update({
                "cpu_offload": cpu_offload,
                "pin_memory": pin_memory,
            })
            
            # Optimizer offload
            if offload_optimizer:
                zero_config["offload_optimizer"] = {
                    "device": "cpu" if cpu_offload else "nvme",
                    "pin_memory": pin_memory,
                }
                if nvme_offload:
                    zero_config["offload_optimizer"]["nvme_path"] = nvme_path
        
        # Stage 3 specific configurations
        if zero_stage == 3:
            # Parameter offload
            if offload_param:
                zero_config["offload_param"] = {
                    "device": "cpu" if cpu_offload else "nvme",
                    "pin_memory": pin_memory,
                }
                if nvme_offload:
                    zero_config["offload_param"]["nvme_path"] = nvme_path
            
            # Additional stage 3 optimizations
            zero_config.update({
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            })
        
        config["zero_optimization"] = zero_config
    
    # Mixed precision configuration
    if fp16_enabled:
        config["fp16"] = {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    
    if bf16_enabled:
        config["bf16"] = {
            "enabled": True,
        }
    
    # Optimizer configuration
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        }
    }
    
    # Scheduler configuration
    config["scheduler"] = {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        }
    }
    
    # Add any additional configuration
    config.update(kwargs)
    
    return config


def save_deepspeed_config(config: Dict[str, Any], config_path: str) -> str:
    """Save DeepSpeed configuration to file
    
    Args:
        config: DeepSpeed configuration dictionary
        config_path: Path to save the configuration file
    
    Returns:
        Path to the saved configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def load_deepspeed_config(config_path: str) -> Dict[str, Any]:
    """Load DeepSpeed configuration from file
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        DeepSpeed configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def auto_select_strategy(
    model_size_gb: float,
    available_gpu_memory_gb: float,
    num_gpus: int = 1
) -> str:
    """Automatically select the best distributed strategy
    
    Args:
        model_size_gb: Model size in GB
        available_gpu_memory_gb: Available GPU memory in GB
        num_gpus: Number of available GPUs
    
    Returns:
        Recommended strategy: 'ddp', 'deepspeed_stage2', or 'deepspeed_stage3'
    """
    # Estimate memory requirements (rough approximation)
    # Model + gradients + optimizer states ≈ 4x model size for full precision
    # With mixed precision ≈ 2.5x model size
    estimated_memory_gb = model_size_gb * 2.5
    
    # If model fits comfortably in single GPU memory, use DDP
    if estimated_memory_gb < available_gpu_memory_gb * 0.8:
        return 'ddp'
    
    # If model is too large for single GPU but manageable with ZeRO-2
    elif model_size_gb < 7.0:  # Models smaller than 7B
        return 'deepspeed_stage2'
    
    # For very large models, use ZeRO-3
    else:
        return 'deepspeed_stage3'


def get_model_size_gb(model: torch.nn.Module) -> float:
    """Estimate model size in GB
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in GB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Convert bytes to GB
    size_gb = param_size / (1024 ** 3)
    return size_gb


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB
    
    Returns:
        Available GPU memory in GB
    """
    if torch.cuda.is_available():
        # Get memory of the current device
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        return total_memory / (1024 ** 3)
    else:
        return 0.0


def initialize_deepspeed(
    model: torch.nn.Module,
    config: Dict[str, Any],
    model_parameters: Optional[list] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    dist_init_required: bool = True
):
    """Initialize DeepSpeed engine
    
    Args:
        model: PyTorch model
        config: DeepSpeed configuration
        model_parameters: Model parameters (optional)
        optimizer: Optimizer (optional)
        lr_scheduler: Learning rate scheduler (optional)
        dist_init_required: Whether distributed initialization is required
    
    Returns:
        Tuple of (model_engine, optimizer, lr_scheduler)
    """
    if not is_deepspeed_available():
        raise ImportError("DeepSpeed is not available. Please install it with: pip install deepspeed")
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=config,
        model_parameters=model_parameters,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=dist_init_required
    )
    
    return model_engine, optimizer, lr_scheduler


def is_deepspeed_zero3_enabled(config: Dict[str, Any]) -> bool:
    """Check if DeepSpeed ZeRO-3 is enabled
    
    Args:
        config: DeepSpeed configuration
    
    Returns:
        True if ZeRO-3 is enabled
    """
    zero_config = config.get("zero_optimization", {})
    return zero_config.get("stage", 0) == 3


def gather_across_processes(tensor: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Gather tensor across processes for DeepSpeed ZeRO-3
    
    Args:
        tensor: Tensor to gather
        config: DeepSpeed configuration
    
    Returns:
        Gathered tensor
    """
    if is_deepspeed_zero3_enabled(config) and is_deepspeed_available():
        # For ZeRO-3, we need to gather parameters
        with deepspeed.zero.GatheredParameters(tensor, modifier_rank=0):
            return tensor.clone()
    return tensor