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
from src.trainers import PPOTrainer, DPOTrainer, GRPOTrainer
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
        choices=["ppo", "dpo", "grpo"],
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
        use_tensorboard=config['logging']['use_tensorboard'],
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
                
            if args.algorithm == "ppo":
                trainer = PPOTrainer(config)
            elif args.algorithm == "dpo":
                trainer = DPOTrainer(config)
            elif args.algorithm == "grpo":
                trainer = GRPOTrainer(config)
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