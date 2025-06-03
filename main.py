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
    
    # Setup logging
    logger = setup_logger(
        name="rl_learning",
        level=config['logging']['log_level'],
        log_dir=os.path.join(config['training']['output_dir'], "logs")
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
                model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
                
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
                model, tokenizer = create_reward_model(config['model']['model_name_or_path'])
                
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
                policy_model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
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
                model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
                
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
                policy_model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
                reward_model, _ = create_reward_model(config['model']['model_name_or_path'])
                
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
                
                trainer = GRPOTrainer(training_config, policy_model, reward_model, tokenizer, train_dataloader, eval_dataloader)
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