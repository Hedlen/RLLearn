"""Configuration management utilities"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['model', 'training', 'algorithms', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    # Create directory if it doesn't exist
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model configuration
    model_config = config.get('model', {})
    if not model_config.get('name'):
        raise ValueError("Model name must be specified")
    
    # Validate training configuration
    training_config = config.get('training', {})
    if training_config.get('learning_rate', 0) <= 0:
        raise ValueError("Learning rate must be positive")
    
    if training_config.get('per_device_train_batch_size', 0) <= 0:
        raise ValueError("Batch size must be positive")
    
    # Validate algorithm-specific configurations
    algorithms_config = config.get('algorithms', {})
    
    # PPO validation
    if 'ppo' in algorithms_config:
        ppo_config = algorithms_config['ppo']
        if ppo_config.get('clip_range', 0) <= 0:
            raise ValueError("PPO clip_range must be positive")
        if not 0 <= ppo_config.get('gae_lambda', 0.95) <= 1:
            raise ValueError("PPO gae_lambda must be between 0 and 1")
    
    # DPO validation
    if 'dpo' in algorithms_config:
        dpo_config = algorithms_config['dpo']
        if dpo_config.get('beta', 0) <= 0:
            raise ValueError("DPO beta must be positive")
    
    # Hardware validation
    hardware_config = config.get('hardware', {})
    if hardware_config.get('torch_dtype') not in [None, 'float16', 'float32', 'bfloat16']:
        raise ValueError("Invalid torch_dtype specified")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'training.learning_rate')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'training.learning_rate')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value