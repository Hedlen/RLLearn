#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path

from src.utils import load_config, setup_logger
from src.data import DataProcessor
from src.data.merger import merge_datasets_for_algorithm, merge_validation_datasets_for_algorithm
from src.evaluators import ModelEvaluator

def _get_algorithm_data_paths(config, algorithm_type):
    """Get optimized data paths for algorithm with fallback logic"""
    # Try new datasets configuration first
    train_file = merge_datasets_for_algorithm(config, algorithm_type)
    eval_file = merge_validation_datasets_for_algorithm(config, algorithm_type)
    
    # Fallback to legacy config
    if not train_file:
        train_file = config.get('data', {}).get('train_file')
    if not eval_file:
        eval_file = config.get('data', {}).get('validation_file')
    
    # Fallback to common file names
    if not train_file:
        common_files = {
            'sft': ["./data/alpaca_chinese_sft.json", "./data/sft_train.json", "./data/train.json"],
            'reward': ["./data/preference_train.json", "./data/reward_train.json", "./data/train.json"],
            'rlhf': ["./data/rlhf_train.json", "./data/ppo_train.json", "./data/train.json"]
        }
        for file_path in common_files.get(algorithm_type, []):
            if os.path.exists(file_path):
                train_file = file_path
                break
    
    if not eval_file:
        common_eval_files = {
            'sft': ["./data/alpaca_chinese_eval.json", "./data/sft_eval.json", "./data/eval.json"],
            'reward': ["./data/preference_eval.json", "./data/reward_eval.json", "./data/eval.json"],
            'rlhf': ["./data/rlhf_eval.json", "./data/ppo_eval.json", "./data/eval.json"]
        }
        for file_path in common_eval_files.get(algorithm_type, []):
            if os.path.exists(file_path):
                eval_file = file_path
                break
    
    return train_file, eval_file


def _create_dataloaders(train_dataset, eval_file, processor, data_collator, training_config, dataset_type):
    """Create optimized train and eval dataloaders"""
    from torch.utils.data import DataLoader
    
    # Create train dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=training_config.dataloader_num_workers,
        pin_memory=training_config.dataloader_pin_memory,
        drop_last=training_config.dataloader_drop_last
    )
    
    # Create eval dataloader if eval file exists
    eval_dataloader = None
    if eval_file and os.path.exists(eval_file):
        logger = logging.getLogger(__name__)
        logger.info(f"Loading evaluation data from: {eval_file}")
        eval_dataset = processor.load_dataset(eval_file, dataset_type=dataset_type)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=training_config.dataloader_num_workers,
            pin_memory=training_config.dataloader_pin_memory
        )
    else:
        logger = logging.getLogger(__name__)
        logger.info("No evaluation data file found, training without validation.")
    
    return train_dataloader, eval_dataloader


def main():
    parser = argparse.ArgumentParser(description="RL Learning Framework")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "eval", "data_process"],
        required=True,
        help="Mode to run: train, eval, or data_process"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        choices=["sft", "reward", "ppo", "dpo", "grpo"],
        help="Algorithm to use for training"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        help="Experiment name (creates subdirectory under output_dir)"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.experiment_name:
        config['training']['experiment_name'] = args.experiment_name
    
    # Setup logging - 保存到实验目录下
    base_output_dir = config['training']['output_dir']
    experiment_name = config.get('training', {}).get('experiment_name')
    
    if experiment_name:
        log_dir = os.path.join(base_output_dir, experiment_name, "logs")
    else:
        log_dir = os.path.join(base_output_dir, "logs")
    
    logger = setup_logger(
        name="rl_learning",
        level=config['logging']['log_level'],
        log_dir=log_dir
    )
    
    logger.info(f"Starting RL Learning Framework in {args.mode} mode")
    logger.info(f"Configuration loaded from {args.config}")
    
    try:
        if args.mode == "data_process":
            # Data processing mode
            processor = DataProcessor(config)
            processor.process_data()
            
        elif args.mode == "train":
            # Training mode
            if not args.algorithm:
                raise ValueError("Algorithm must be specified for training mode")
            
            # Update config with command line arguments
            if args.resume_from_checkpoint:
                config['training']['resume_from_checkpoint'] = args.resume_from_checkpoint
            
            # Import model creation functions
            from src.models import create_policy_model, create_reward_model, create_value_model
            
            # Create shared data processor for efficiency
            processor = DataProcessor(config=config)
            
            # Create models and tokenizer based on algorithm
            if args.algorithm == "sft":
                from src.trainers.sft_trainer import SFTConfig
                from src.data.collator import DataCollatorForSFT
                from torch.utils.data import DataLoader
                
                training_config = SFTConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                # Create policy model with PEFT support
                model_config = config['model']
                model, tokenizer = create_policy_model(
                    model_name_or_path=model_config['model_name_or_path'],
                    use_peft=model_config.get('use_peft', False),
                    peft_config=model_config.get('peft_config'),
                    quantization_config=model_config.get('quantization_config')
                )
                
                # Get optimized data paths
                train_file, eval_file = _get_algorithm_data_paths(config, 'sft')
                
                if not train_file:
                    raise FileNotFoundError(f"No training data file found for SFT. Please configure datasets in config.yaml")
                
                logger.info(f"Loading SFT training data from: {train_file}")
                train_dataset = processor.load_dataset(train_file, dataset_type="sft")
                
                # Create data collator
                data_collator = DataCollatorForSFT(
                    tokenizer=tokenizer,
                    max_length=training_config.max_length,
                    padding=True
                )
                
                # Create dataloaders
                train_dataloader, eval_dataloader = _create_dataloaders(
                    train_dataset, eval_file, processor, data_collator, training_config, "sft"
                )
                
                from src.trainers.sft_trainer import SFTTrainer
                trainer = SFTTrainer(training_config, model, tokenizer, train_dataloader, eval_dataloader)
            elif args.algorithm == "reward":
                from src.trainers.reward_trainer import RewardTrainingConfig, RewardModelTrainer
                from src.data.collator import DataCollatorForRewardModel
                
                training_config = RewardTrainingConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                # Create reward model with PEFT support
                model_config = config['model']
                model, tokenizer = create_reward_model(
                    model_name_or_path=model_config['model_name_or_path'],
                    use_peft=model_config.get('use_peft', False),
                    peft_config=model_config.get('peft_config'),
                    quantization_config=model_config.get('quantization_config')
                )
                
                # Get optimized data paths
                train_file, eval_file = _get_algorithm_data_paths(config, 'reward')
                
                if not train_file:
                    raise FileNotFoundError(f"No training data file found for reward training. Please configure datasets in config.yaml")
                
                logger.info(f"Loading reward training data from: {train_file}")
                train_dataset = processor.load_dataset(train_file, dataset_type="preference")
                
                # Create data collator
                data_collator = DataCollatorForRewardModel(
                    tokenizer=tokenizer,
                    max_length=training_config.max_length,
                    padding=True
                )
                
                # Create dataloaders
                train_dataloader, eval_dataloader = _create_dataloaders(
                    train_dataset, eval_file, processor, data_collator, training_config, "preference"
                )
                
                trainer = RewardModelTrainer(training_config, model, tokenizer, train_dataloader, eval_dataloader)
            elif args.algorithm == "ppo":
                from src.trainers.ppo_trainer import PPOTrainingConfig, PPOTrainer
                from src.data.collator import DataCollatorForRL
                
                training_config = PPOTrainingConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                # Create policy model with PEFT support
                model_config = config['model']
                policy_model, tokenizer = create_policy_model(
                    model_name_or_path=model_config['model_name_or_path'],
                    use_peft=model_config.get('use_peft', False),
                    peft_config=model_config.get('peft_config'),
                    quantization_config=model_config.get('quantization_config')
                )
                value_model, _ = create_value_model(config['model']['model_name_or_path'])
                
                # Get optimized data paths (PPO typically uses SFT data)
                train_file, eval_file = _get_algorithm_data_paths(config, 'sft')
                
                if not train_file:
                    raise FileNotFoundError(f"No training data file found for PPO training. Please configure datasets in config.yaml")
                
                logger.info(f"Loading PPO training data from: {train_file}")
                train_dataset = processor.load_dataset(train_file, dataset_type="sft")
                
                # Create data collator
                data_collator = DataCollatorForRL(
                    tokenizer=tokenizer,
                    max_length=training_config.max_length,
                    padding=True
                )
                
                # Create dataloaders
                train_dataloader, eval_dataloader = _create_dataloaders(
                    train_dataset, eval_file, processor, data_collator, training_config, "sft"
                )
                
                trainer = PPOTrainer(training_config, policy_model, value_model, tokenizer, train_dataloader, eval_dataloader)
            elif args.algorithm == "dpo":
                from src.trainers.dpo_trainer import DPOTrainingConfig, DPOTrainer
                from src.data.collator import DataCollatorForPreference
                
                training_config = DPOTrainingConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                # Create policy model with PEFT support
                model_config = config['model']
                model, tokenizer = create_policy_model(
                    model_name_or_path=model_config['model_name_or_path'],
                    use_peft=model_config.get('use_peft', False),
                    peft_config=model_config.get('peft_config'),
                    quantization_config=model_config.get('quantization_config')
                )
                
                # Get optimized data paths (DPO uses preference data like reward model)
                train_file, eval_file = _get_algorithm_data_paths(config, 'reward')
                
                if not train_file:
                    raise FileNotFoundError(f"No training data file found for DPO training. Please configure datasets in config.yaml")
                
                logger.info(f"Loading DPO training data from: {train_file}")
                train_dataset = processor.load_dataset(train_file, dataset_type="preference")
                
                # Create data collator
                data_collator = DataCollatorForPreference(
                    tokenizer=tokenizer,
                    max_length=training_config.max_length,
                    padding=True
                )
                
                # Create dataloaders
                train_dataloader, eval_dataloader = _create_dataloaders(
                    train_dataset, eval_file, processor, data_collator, training_config, "preference"
                )
                
                trainer = DPOTrainer(training_config, model, tokenizer, train_dataloader, eval_dataloader)
            elif args.algorithm == "grpo":
                from src.trainers.grpo_trainer import GRPOConfig, GRPOTrainer
                from src.data.collator import DataCollatorForRL
                
                training_config = GRPOConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                # Create policy model with PEFT support
                model_config = config['model']
                policy_model, tokenizer = create_policy_model(
                    model_name_or_path=model_config['model_name_or_path'],
                    use_peft=model_config.get('use_peft', False),
                    peft_config=model_config.get('peft_config'),
                    quantization_config=model_config.get('quantization_config')
                )
                # Create reward model or reward function based on configuration
                reward_model = None
                reward_function = None
                
                # Get GRPO reward configuration
                grpo_config = config.get('algorithms', {}).get('grpo', {})
                reward_config = grpo_config.get('reward_config', {})
                reward_type = reward_config.get('reward_type', 'model')  # Default to 'model'
                
                if reward_type == 'function':
                    # Use custom reward function
                    reward_function_config = reward_config.get('reward_function', {})
                    module_path = reward_function_config.get('module_path')
                    function_name = reward_function_config.get('function_name')
                    function_kwargs = reward_function_config.get('function_kwargs', {})
                    
                    if module_path and function_name:
                        try:
                            import importlib
                            module = importlib.import_module(module_path)
                            reward_function_class = getattr(module, function_name)
                            
                            # Check if it's a class or function
                            if hasattr(reward_function_class, '__call__') and hasattr(reward_function_class, '__init__'):
                                # It's a class, instantiate it with kwargs
                                reward_function = reward_function_class(**function_kwargs)
                                logger.info(f"Using custom reward function class: {module_path}.{function_name} with kwargs: {function_kwargs}")
                            else:
                                # It's a function, use it directly (kwargs will be passed during call)
                                reward_function = reward_function_class
                                logger.info(f"Using custom reward function: {module_path}.{function_name}")
                        except Exception as e:
                            logger.error(f"Failed to load custom reward function {module_path}.{function_name}: {e}")
                            logger.warning("Falling back to reward model")
                            reward_type = 'model'
                    else:
                        logger.warning("Invalid reward function configuration (missing module_path or function_name), falling back to reward model")
                        reward_type = 'model'
                
                if reward_type == 'model':
                    # Use reward model
                    rlhf_config = config.get('rlhf', {})
                    reward_model_path = rlhf_config.get('reward_model_path')
                    
                    if reward_model_path and os.path.exists(reward_model_path):
                        # Load pre-trained reward model
                        logger.info(f"Loading pre-trained reward model from: {reward_model_path}")
                        reward_model, _ = create_reward_model(
                            model_name_or_path=reward_model_path,
                            use_peft=False,  # Pre-trained model doesn't need PEFT
                            peft_config=None,
                            quantization_config=None
                        )
                    else:
                        # Create new reward model (fallback)
                        logger.warning(f"Reward model path not found: {reward_model_path}, creating new reward model")
                        model_config = config['model']
                        reward_model, _ = create_reward_model(
                            model_name_or_path=model_config['model_name_or_path'],
                            use_peft=model_config.get('use_peft', False),
                            peft_config=model_config.get('peft_config'),
                            quantization_config=model_config.get('quantization_config')
                        )
                
                # Get optimized data paths (GRPO typically uses SFT data)
                train_file, eval_file = _get_algorithm_data_paths(config, 'sft')
                
                if not train_file:
                    raise FileNotFoundError(f"No training data file found for GRPO training. Please configure datasets in config.yaml")
                
                logger.info(f"Loading GRPO training data from: {train_file}")
                train_dataset = processor.load_dataset(train_file, dataset_type="sft")
                
                # Create data collator
                data_collator = DataCollatorForRL(
                    tokenizer=tokenizer,
                    max_length=training_config.max_length,
                    padding=True
                )
                
                # Create dataloaders
                train_dataloader, eval_dataloader = _create_dataloaders(
                    train_dataset, eval_file, processor, data_collator, training_config, "sft"
                )
                
                trainer = GRPOTrainer(
                    training_config, 
                    policy_model, 
                    reward_model=reward_model,
                    reward_function=reward_function,
                    tokenizer=tokenizer, 
                    train_dataloader=train_dataloader, 
                    eval_dataloader=eval_dataloader
                )
            elif args.algorithm == "rlhf":
                # RLHF training - unified interface for PPO/DPO/GRPO
                rlhf_config = config.get('algorithms', {}).get('rlhf', {})
                algorithm_type = rlhf_config.get('algorithm_type', 'ppo')  # Default to PPO
                
                logger.info(f"Starting RLHF training with {algorithm_type.upper()} algorithm")
                
                # Create policy model
                model_config = config['model']
                policy_model, tokenizer = create_policy_model(
                    model_name_or_path=model_config['model_name_or_path'],
                    use_peft=model_config.get('use_peft', False),
                    peft_config=model_config.get('peft_config'),
                    quantization_config=model_config.get('quantization_config')
                )
                
                # Handle reward model/function based on RLHF configuration
                reward_model = None
                reward_function = None
                
                reward_config = rlhf_config.get('reward_config', {})
                reward_type = reward_config.get('reward_type', 'model')
                
                if reward_type == 'function':
                    # Use custom reward function
                    reward_function_config = reward_config.get('reward_function', {})
                    module_path = reward_function_config.get('module_path')
                    function_name = reward_function_config.get('function_name')
                    function_kwargs = reward_function_config.get('function_kwargs', {})
                    
                    if module_path and function_name:
                        try:
                            import importlib
                            module = importlib.import_module(module_path)
                            reward_function_class = getattr(module, function_name)
                            
                            if hasattr(reward_function_class, '__call__') and hasattr(reward_function_class, '__init__'):
                                reward_function = reward_function_class(**function_kwargs)
                                logger.info(f"Using custom reward function class: {module_path}.{function_name}")
                            else:
                                reward_function = reward_function_class
                                logger.info(f"Using custom reward function: {module_path}.{function_name}")
                        except Exception as e:
                            logger.error(f"Failed to load custom reward function: {e}")
                            reward_type = 'model'
                    else:
                        logger.warning("Invalid reward function configuration, falling back to reward model")
                        reward_type = 'model'
                
                if reward_type == 'model':
                    # Use reward model
                    global_rlhf_config = config.get('rlhf', {})
                    reward_model_path = global_rlhf_config.get('reward_model_path')
                    
                    if reward_model_path and os.path.exists(reward_model_path):
                        logger.info(f"Loading pre-trained reward model from: {reward_model_path}")
                        reward_model, _ = create_reward_model(
                            model_name_or_path=reward_model_path,
                            use_peft=False,
                            peft_config=None,
                            quantization_config=None
                        )
                    else:
                        logger.warning(f"Reward model path not found: {reward_model_path}, creating new reward model")
                        reward_model, _ = create_reward_model(
                            model_name_or_path=model_config['model_name_or_path'],
                            use_peft=model_config.get('use_peft', False),
                            peft_config=model_config.get('peft_config'),
                            quantization_config=model_config.get('quantization_config')
                        )
                
                # Get training data (RLHF typically uses SFT data for generation)
                train_file, eval_file = _get_algorithm_data_paths(config, 'rlhf')
                
                if not train_file:
                    # Fallback to SFT data if no RLHF data is configured
                    logger.warning("No RLHF training data found, falling back to SFT data")
                    train_file, eval_file = _get_algorithm_data_paths(config, 'sft')
                
                if not train_file:
                    raise FileNotFoundError(f"No training data file found for RLHF training. Please configure datasets in config.yaml")
                
                logger.info(f"Loading RLHF training data from: {train_file}")
                train_dataset = processor.load_dataset(train_file, dataset_type="sft")
                
                # Route to specific algorithm trainer based on algorithm_type
                if algorithm_type == 'ppo':
                    from src.trainers.ppo_trainer import PPOTrainingConfig, PPOTrainer
                    from src.data.collator import DataCollatorForRL
                    
                    # Merge RLHF config with PPO-specific config
                    ppo_config = {**config['training'], **rlhf_config}
                    training_config = PPOTrainingConfig(
                        model_name_or_path=config['model']['model_name_or_path'],
                        **ppo_config
                    )
                    
                    # Create value model for PPO
                    value_model, _ = create_value_model(config['model']['model_name_or_path'])
                    
                    data_collator = DataCollatorForRL(
                        tokenizer=tokenizer,
                        max_length=training_config.max_length,
                        padding=True
                    )
                    
                    train_dataloader, eval_dataloader = _create_dataloaders(
                        train_dataset, eval_file, processor, data_collator, training_config, "sft"
                    )
                    
                    trainer = PPOTrainer(
                        training_config, 
                        policy_model, 
                        value_model, 
                        tokenizer, 
                        train_dataloader, 
                        eval_dataloader
                    )
                    
                elif algorithm_type == 'dpo':
                    from src.trainers.dpo_trainer import DPOTrainingConfig, DPOTrainer
                    from src.data.collator import DataCollatorForPreference
                    
                    # Merge RLHF config with DPO-specific config
                    dpo_config = {**config['training'], **rlhf_config}
                    training_config = DPOTrainingConfig(
                        model_name_or_path=config['model']['model_name_or_path'],
                        **dpo_config
                    )
                    
                    # DPO needs preference data
                    train_file, eval_file = _get_algorithm_data_paths(config, 'reward')
                    if not train_file:
                        raise FileNotFoundError("DPO requires preference data. Please configure reward datasets.")
                    
                    train_dataset = processor.load_dataset(train_file, dataset_type="preference")
                    
                    data_collator = DataCollatorForPreference(
                        tokenizer=tokenizer,
                        max_length=training_config.max_length,
                        padding=True
                    )
                    
                    train_dataloader, eval_dataloader = _create_dataloaders(
                        train_dataset, eval_file, processor, data_collator, training_config, "preference"
                    )
                    
                    trainer = DPOTrainer(
                        training_config, 
                        policy_model, 
                        tokenizer, 
                        train_dataloader, 
                        eval_dataloader
                    )
                    
                elif algorithm_type == 'grpo':
                    from src.trainers.grpo_trainer import GRPOConfig, GRPOTrainer
                    from src.data.collator import DataCollatorForRL
                    
                    # Merge RLHF config with GRPO-specific config
                    grpo_config = {**config['training'], **rlhf_config}
                    training_config = GRPOConfig(
                        model_name_or_path=config['model']['model_name_or_path'],
                        **grpo_config
                    )
                    
                    data_collator = DataCollatorForRL(
                        tokenizer=tokenizer,
                        max_length=training_config.max_length,
                        padding=True
                    )
                    
                    train_dataloader, eval_dataloader = _create_dataloaders(
                        train_dataset, eval_file, processor, data_collator, training_config, "sft"
                    )
                    
                    trainer = GRPOTrainer(
                        training_config, 
                        policy_model, 
                        reward_model=reward_model,
                        reward_function=reward_function,
                        tokenizer=tokenizer, 
                        train_dataloader=train_dataloader, 
                        eval_dataloader=eval_dataloader
                    )
                    
                else:
                    raise ValueError(f"Unknown RLHF algorithm type: {algorithm_type}. Supported: ppo, dpo, grpo")
                    
            else:
                raise ValueError(f"Unknown algorithm: {args.algorithm}")
                
            trainer.train()
            
        elif args.mode == "eval":
            # Evaluation mode
            evaluator = ModelEvaluator(config)
            evaluator.evaluate()
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        sys.exit(1)
    
    logger.info("RL Learning Framework completed successfully")


if __name__ == "__main__":
    main()