#!/usr/bin/env python3
"""Complete training and evaluation example for RL Learning Framework"""

import os
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime

import torch
from transformers import AutoTokenizer

from src.data import DataProcessor
from src.trainers import (
    SFTTrainer, SFTConfig,
    RewardModelTrainer, RewardTrainingConfig,
    PPOTrainer, PPOTrainingConfig,
    DPOTrainer, DPOTrainingConfig,
    GRPOTrainer, GRPOConfig
)
from src.models import create_policy_model, create_value_model, create_reward_model
from src.evaluators import ModelEvaluator, AutomaticEvaluator
from src.utils import setup_logger, load_config


def setup_environment():
    """Setup training environment"""
    # Create necessary directories
    dirs = [
        "./data", "./output", "./logs", "./eval_results",
        "./output/sft", "./output/reward", "./output/ppo", "./output/dpo", "./output/grpo"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logger("./logs/training.log")
    
    print("Environment setup complete!")


def prepare_sample_data():
    """Prepare sample training data"""
    
    # SFT training data
    sft_data = [
        {
            "prompt": "什么是人工智能？",
            "response": "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
        },
        {
            "prompt": "解释机器学习的基本概念",
            "response": "机器学习是人工智能的一个子集，它使计算机能够从数据中学习并做出预测或决策，而无需明确编程。"
        },
        {
            "prompt": "深度学习与传统机器学习的区别",
            "response": "深度学习使用多层神经网络来学习数据的复杂模式，而传统机器学习通常使用较简单的算法和手工特征工程。"
        }
    ]
    
    # Preference data for reward model and DPO
    preference_data = [
        {
            "prompt": "如何学习编程？",
            "chosen": "学习编程需要循序渐进：首先选择一门编程语言，然后学习基础语法，多做练习项目，阅读优秀代码，参与开源项目。",
            "rejected": "直接开始写复杂程序就行了。"
        },
        {
            "prompt": "什么是数据结构？",
            "chosen": "数据结构是计算机存储、组织数据的方式，包括数组、链表、栈、队列、树、图等，选择合适的数据结构能提高程序效率。",
            "rejected": "数据结构就是存储数据的东西。"
        }
    ]
    
    # Save data
    with open("./data/sft_train.json", "w", encoding="utf-8") as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)
    
    with open("./data/preference_train.json", "w", encoding="utf-8") as f:
        json.dump(preference_data, f, indent=2, ensure_ascii=False)
    
    print("Sample data prepared!")


def train_sft_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_steps: int = 100):
    """Train SFT model"""
    print("\n=== Training SFT Model ===")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model, _ = create_policy_model(model_name)
    
    # Prepare data
    processor = DataProcessor(tokenizer=tokenizer, max_length=2048)
    train_dataset = processor.load_dataset("./data/sft_train.json", dataset_type="sft")
    
    # Training configuration
    config = SFTConfig(
        model_name_or_path=model_name,
        output_dir="./output/sft",
        max_steps=max_steps,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=50,
        warmup_steps=10,
        max_length=2048
    )
    
    # Create trainer
    trainer = SFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("./output/sft/final_model")
    print("SFT training completed!")
    
    return "./output/sft/final_model"


def train_reward_model(sft_model_path: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_steps: int = 50):
    """Train reward model"""
    print("\n=== Training Reward Model ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create reward model
    model, _ = create_reward_model(sft_model_path)
    
    # Prepare data
    processor = DataProcessor(tokenizer=tokenizer, max_length=2048)
    train_dataset = processor.load_dataset("./data/preference_train.json", dataset_type="preference")
    
    # Training configuration
    config = RewardTrainingConfig(
        model_name_or_path=sft_model_path,
        output_dir="./output/reward",
        max_steps=max_steps,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_steps=25,
        warmup_steps=5,
        max_length=2048,
        loss_type="hinge"
    )
    
    # Create trainer
    trainer = RewardModelTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("./output/reward/final_model")
    print("Reward model training completed!")
    
    return "./output/reward/final_model"


def train_grpo_model(sft_model_path: str, reward_model_path: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_steps: int = 50):
    """Train GRPO model"""
    print("\n=== Training GRPO Model ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    policy_model, _ = create_policy_model(sft_model_path)
    reward_model, _ = create_reward_model(reward_model_path)
    
    # Prepare data
    processor = DataProcessor(tokenizer=tokenizer, max_length=2048)
    train_dataset = processor.load_dataset("./data/sft_train.json", dataset_type="sft")
    
    # Training configuration
    config = GRPOConfig(
        model_name_or_path=sft_model_path,
        output_dir="./output/grpo",
        max_steps=max_steps,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_steps=25,
        warmup_steps=5,
        group_size=4,
        beta=0.1,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        config=config,
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        train_dataloader=None  # Will be created from dataset
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("./output/grpo/final_model")
    print("GRPO training completed!")
    
    return "./output/grpo/final_model"


def train_ppo_model(sft_model_path: str, reward_model_path: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_steps: int = 50):
    """Train PPO model"""
    print("\n=== Training PPO Model ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    policy_model, _ = create_policy_model(sft_model_path)
    value_model, _ = create_value_model(sft_model_path)
    reward_model, _ = create_reward_model(reward_model_path)
    
    # Prepare data (use prompts from SFT data)
    with open("./data/sft_train.json", "r", encoding="utf-8") as f:
        sft_data = json.load(f)
    
    prompts = [item["prompt"] for item in sft_data]
    
    # Training configuration
    config = PPOTrainingConfig(
        model_name_or_path=sft_model_path,
        output_dir="./output/ppo",
        max_steps=max_steps,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_steps=25,
        warmup_steps=5,
        ppo_epochs=2,
        clip_range=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        max_new_tokens=128
    )
    
    # Create trainer
    trainer = PPOTrainer(
        config=config,
        policy_model=policy_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        prompts=prompts
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("./output/ppo/final_model")
    print("PPO training completed!")
    
    return "./output/ppo/final_model"


def train_dpo_model(sft_model_path: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_steps: int = 50):
    """Train DPO model"""
    print("\n=== Training DPO Model ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    model, _ = create_policy_model(sft_model_path)
    reference_model, _ = create_policy_model(sft_model_path)
    
    # Prepare data
    processor = DataProcessor(tokenizer=tokenizer, max_length=2048)
    train_dataset = processor.load_dataset("./data/preference_train.json", dataset_type="preference")
    
    # Training configuration
    config = DPOTrainingConfig(
        model_name_or_path=sft_model_path,
        output_dir="./output/dpo",
        max_steps=max_steps,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_steps=25,
        warmup_steps=5,
        beta=0.1,
        loss_type="sigmoid",
        max_length=2048
    )
    
    # Create trainer
    trainer = DPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reference_model=reference_model,
        train_dataset=train_dataset
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("./output/dpo/final_model")
    print("DPO training completed!")
    
    return "./output/dpo/final_model"


def evaluate_models(model_paths: dict, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Evaluate trained models"""
    print("\n=== Evaluating Models ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create evaluator
    evaluator = AutomaticEvaluator(
        tokenizer=tokenizer,
        output_dir="./eval_results"
    )
    
    # Test prompts
    test_prompts = [
        "什么是深度学习？",
        "如何提高编程技能？",
        "解释云计算的概念"
    ]
    
    results = {}
    
    for model_type, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Model {model_type} not found at {model_path}, skipping...")
            continue
            
        print(f"\nEvaluating {model_type} model...")
        
        try:
            # Load model
            if model_type == "reward":
                model, _ = create_reward_model(model_path)
                # For reward model, evaluate preference accuracy
                chosen_texts = ["这是一个详细且准确的回答"]
                rejected_texts = ["这个回答不够好"]
                
                model_results = evaluator.evaluate_reward_model(
                    reward_model=model,
                    chosen_texts=chosen_texts,
                    rejected_texts=rejected_texts
                )
            else:
                model, _ = create_policy_model(model_path)
                
                # Generate responses
                model_results = evaluator.evaluate_generation_quality(
                    predictions=evaluator._generate_responses(
                        model=model,
                        prompts=test_prompts,
                        generation_kwargs={
                            'max_new_tokens': 64,
                            'temperature': 0.7,
                            'do_sample': True,
                            'pad_token_id': tokenizer.eos_token_id
                        },
                        batch_size=1
                    )
                )
                
                # Add perplexity
                perplexity_results = evaluator.evaluate_model_perplexity(
                    model=model,
                    texts=test_prompts,
                    batch_size=1
                )
                model_results.update(perplexity_results)
            
            results[model_type] = model_results
            
            # Save individual results
            result_file = evaluator.save_results(
                model_results, 
                f"{model_type}_evaluation.json"
            )
            print(f"Results saved to: {result_file}")
            
        except Exception as e:
            print(f"Error evaluating {model_type}: {e}")
            continue
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()} Model:")
        if model_type == "reward":
            print(f"  Preference Accuracy: {model_results.get('accuracy', 0):.4f}")
            print(f"  Reward Margin: {model_results.get('margin', 0):.4f}")
        else:
            print(f"  Average Length: {model_results.get('avg_length', 0):.2f}")
            print(f"  Diversity (Distinct-2): {model_results.get('distinct_2', 0):.4f}")
            print(f"  Perplexity: {model_results.get('perplexity', 0):.2f}")
    
    return results


def generate_samples(model_paths: dict, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Generate sample responses from trained models"""
    print("\n=== Generating Sample Responses ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts
    test_prompts = [
        "什么是人工智能？",
        "如何学习机器学习？"
    ]
    
    # Create evaluator for generation
    evaluator = AutomaticEvaluator(tokenizer=tokenizer)
    
    all_responses = {}
    
    for model_type, model_path in model_paths.items():
        if model_type == "reward" or not os.path.exists(model_path):
            continue
            
        print(f"\nGenerating responses with {model_type} model...")
        
        try:
            model, _ = create_policy_model(model_path)
            
            responses = evaluator._generate_responses(
                model=model,
                prompts=test_prompts,
                generation_kwargs={
                    'max_new_tokens': 100,
                    'temperature': 0.7,
                    'do_sample': True,
                    'pad_token_id': tokenizer.eos_token_id
                },
                batch_size=1
            )
            
            all_responses[model_type] = responses
            
            for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"{model_type.upper()} Response: {response}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error generating with {model_type}: {e}")
    
    # Save all responses
    output_file = "./eval_results/sample_responses.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompts": test_prompts,
            "responses": all_responses,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nSample responses saved to: {output_file}")


def main():
    """Main training and evaluation pipeline"""
    parser = argparse.ArgumentParser(description="RL Learning Framework Training Example")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--only_sft", action="store_true", help="Only train SFT model")
    
    args = parser.parse_args()
    
    print("=== RL Learning Framework Training Example ===")
    print(f"Base model: {args.model_name}")
    print(f"Max steps: {args.max_steps}")
    
    # Setup
    setup_environment()
    
    if not args.skip_training:
        prepare_sample_data()
        
        # Train models step by step - RLHF Pipeline
        print("\nStarting RLHF training pipeline...")
        print("Training order: SFT → Reward Model → RL Algorithms (PPO/DPO/GRPO)")
        print("⚠️  Note: PPO and GRPO require reward model, DPO can work independently")
        
        # Step 1: Train SFT model (Foundation for all subsequent training)
        print("\n📚 Step 1/4: Training SFT model...")
        sft_model_path = train_sft_model(args.model_name, args.max_steps)
        
        if args.only_sft:
            print("\n✅ SFT-only training completed!")
            model_paths = {"sft": sft_model_path}
        else:
            # Step 2: Train reward model (Required for PPO and GRPO)
            print("\n🎯 Step 2/4: Training Reward model...")
            print("   Using SFT model as base for reward model training")
            reward_model_path = train_reward_model(sft_model_path, args.model_name, args.max_steps // 2)
            
            # Step 3: Train RL algorithms
            print("\n🚀 Step 3/4: Training RL algorithms...")
            
            # PPO training (requires reward model)
            print("   Training PPO model (requires reward model)...")
            ppo_model_path = train_ppo_model(sft_model_path, reward_model_path, args.model_name, args.max_steps // 2)
            
            # DPO training (independent of reward model)
            print("   Training DPO model (independent training)...")
            dpo_model_path = train_dpo_model(sft_model_path, args.model_name, args.max_steps // 2)
            
            # GRPO training (requires reward model)
            print("   Training GRPO model (requires reward model)...")
            grpo_model_path = train_grpo_model(sft_model_path, reward_model_path, args.model_name, args.max_steps // 2)
            
            model_paths = {
                "sft": sft_model_path,
                "reward": reward_model_path,
                "ppo": ppo_model_path,
                "dpo": dpo_model_path,
                "grpo": grpo_model_path
            }
            
            print("\n✅ All RL algorithms training completed!")
    else:
        print("\n📂 Using existing trained models...")
        # Use existing models
        model_paths = {
            "sft": "./output/sft/final_model",
            "reward": "./output/reward/final_model",
            "ppo": "./output/ppo/final_model",
            "dpo": "./output/dpo/final_model",
            "grpo": "./output/grpo/final_model"
        }
    
    # Step 4: Evaluate and compare models
    print("\n📊 Step 4/4: Evaluating trained models...")
    evaluation_results = evaluate_models(model_paths, args.model_name)
    
    # Generate sample responses for comparison
    print("\n💬 Generating sample responses for comparison...")
    generate_samples(model_paths, args.model_name)
    
    print("\n=== Training and Evaluation Complete! ===")
    print("Check the following directories for results:")
    print("  - ./output/: Trained models")
    print("  - ./eval_results/: Evaluation results")
    print("  - ./logs/: Training logs")


if __name__ == "__main__":
    main()