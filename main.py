#!/usr/bin/env python3
"""
RL Learning Framework - Main Entry Point

A comprehensive framework for reinforcement learning with large language models,
supporting PPO, DPO, GRPO algorithms and RLHF training.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from src.utils.logger import setup_logger
from src.trainers import PPOTrainer, DPOTrainer, GRPOTrainer, SFTTrainer, RewardModelTrainer
from src.evaluators import ModelEvaluator
from src.data import DataProcessor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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
            
            # Import model creation functions
            from src.models import create_policy_model, create_reward_model, create_value_model
            
            # Create models and tokenizer based on algorithm
            if args.algorithm == "sft":
                from src.trainers.sft_trainer import SFTConfig
                training_config = SFTConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
                trainer = SFTTrainer(training_config, model, tokenizer)
            elif args.algorithm == "reward":
                from src.trainers.reward_trainer import RewardTrainingConfig
                training_config = RewardTrainingConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                model, tokenizer = create_reward_model(config['model']['model_name_or_path'])
                trainer = RewardModelTrainer(training_config, model, tokenizer)
            elif args.algorithm == "ppo":
                from src.trainers.ppo_trainer import PPOTrainingConfig
                training_config = PPOTrainingConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                policy_model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
                value_model, _ = create_value_model(config['model']['model_name_or_path'])
                trainer = PPOTrainer(training_config, policy_model, value_model, tokenizer)
            elif args.algorithm == "dpo":
                from src.trainers.dpo_trainer import DPOTrainingConfig
                training_config = DPOTrainingConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
                trainer = DPOTrainer(training_config, model, tokenizer)
            elif args.algorithm == "grpo":
                from src.trainers.grpo_trainer import GRPOConfig
                training_config = GRPOConfig(
                    model_name_or_path=config['model']['model_name_or_path'],
                    **config['training']
                )
                policy_model, tokenizer = create_policy_model(config['model']['model_name_or_path'])
                reward_model, _ = create_reward_model(config['model']['model_name_or_path'])
                trainer = GRPOTrainer(training_config, policy_model, reward_model, tokenizer)
            else:
                raise ValueError(f"Unknown algorithm: {args.algorithm}")
                
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            
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