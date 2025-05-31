#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5 3B模型RLHF训练示例

本脚本展示了如何使用Qwen2.5 3B模型进行完整的RLHF训练流程：
1. 数据准备
2. 监督微调 (SFT)
3. 偏好优化 (PPO/DPO)
"""

import os
import json
import argparse
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """设置训练环境"""
    # 创建必要的目录
    directories = [
        "./data",
        "./output",
        "./output/sft",
        "./output/ppo",
        "./output/dpo",
        "./logs",
        "./cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建目录: {directory}")

def prepare_data(max_samples=5000):
    """准备训练数据"""
    logger.info("=== 步骤1: 准备训练数据 ===")
    
    # 检查数据准备脚本是否存在
    if not os.path.exists("prepare_datasets.py"):
        logger.error("未找到数据准备脚本 prepare_datasets.py")
        return False
    
    # 运行数据准备脚本
    import subprocess
    
    try:
        # 下载中文数据集用于SFT
        logger.info("下载中文数据集...")
        result = subprocess.run([
            "python", "prepare_datasets.py",
            "--datasets", "belle", "alpaca-chinese",
            "--max_samples", str(max_samples),
            "--output_dir", "./data"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"数据下载失败: {result.stderr}")
            return False
        
        # 下载偏好数据集
        logger.info("下载偏好数据集...")
        result = subprocess.run([
            "python", "prepare_datasets.py",
            "--datasets", "hh-rlhf",
            "--max_samples", str(max_samples // 2),
            "--output_dir", "./data"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"偏好数据下载失败，将使用合成数据: {result.stderr}")
        
        logger.info("数据准备完成")
        return True
        
    except Exception as e:
        logger.error(f"数据准备过程中出错: {e}")
        return False

def create_sft_config():
    """创建SFT训练配置"""
    config = {
        "model": {
            "name_or_path": "Qwen/Qwen2.5-3B",
            "cache_dir": "./cache"
        },
        "training": {
            "algorithm": "sft",
            "output_dir": "./output/sft",
            "num_epochs": 2,
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False
        },
        "data": {
            "train_file": "./data/belle_sft_train.json",
            "eval_file": "./data/alpaca_chinese_sft_train.json",
            "max_length": 512,
            "max_prompt_length": 256
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "do_sample": True
        },
        "logging": {
            "level": "INFO",
            "log_dir": "./logs/sft",
            "tensorboard": True
        }
    }
    
    config_file = "./config_sft.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"SFT配置已保存到: {config_file}")
    return config_file

def create_ppo_config():
    """创建PPO训练配置"""
    config = {
        "model": {
            "name_or_path": "./output/sft",  # 使用SFT训练后的模型
            "cache_dir": "./cache"
        },
        "training": {
            "algorithm": "ppo",
            "output_dir": "./output/ppo",
            "num_epochs": 1,
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_grad_norm": 1.0,
            "warmup_steps": 50,
            "logging_steps": 5,
            "save_steps": 200,
            "eval_steps": 200
        },
        "ppo": {
            "ppo_epochs": 4,
            "clip_range": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "max_grad_norm": 1.0
        },
        "data": {
            "train_file": "./data/synthetic_preference_train.json",
            "max_length": 512,
            "max_prompt_length": 256
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "do_sample": True
        },
        "logging": {
            "level": "INFO",
            "log_dir": "./logs/ppo",
            "tensorboard": True
        }
    }
    
    config_file = "./config_ppo.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"PPO配置已保存到: {config_file}")
    return config_file

def create_dpo_config():
    """创建DPO训练配置"""
    config = {
        "model": {
            "name_or_path": "./output/sft",  # 使用SFT训练后的模型
            "cache_dir": "./cache"
        },
        "training": {
            "algorithm": "dpo",
            "output_dir": "./output/dpo",
            "num_epochs": 1,
            "learning_rate": 5e-6,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_grad_norm": 1.0,
            "warmup_steps": 50,
            "logging_steps": 5,
            "save_steps": 200,
            "eval_steps": 200
        },
        "dpo": {
            "beta": 0.1,
            "loss_type": "sigmoid",
            "label_smoothing": 0.0
        },
        "data": {
            "train_file": "./data/hh_rlhf_helpful-base_train.json",
            "max_length": 512,
            "max_prompt_length": 256
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "do_sample": True
        },
        "logging": {
            "level": "INFO",
            "log_dir": "./logs/dpo",
            "tensorboard": True
        }
    }
    
    config_file = "./config_dpo.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"DPO配置已保存到: {config_file}")
    return config_file

def run_training(config_file, stage_name):
    """运行训练"""
    logger.info(f"=== 开始{stage_name}训练 ===")
    
    # 检查主训练脚本是否存在
    if not os.path.exists("main.py"):
        logger.error("未找到主训练脚本 main.py")
        return False
    
    import subprocess
    
    try:
        # 运行训练
        result = subprocess.run([
            "python", "main.py",
            "--config", config_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"{stage_name}训练失败: {result.stderr}")
            return False
        
        logger.info(f"{stage_name}训练完成")
        return True
        
    except Exception as e:
        logger.error(f"{stage_name}训练过程中出错: {e}")
        return False

def test_model(model_path, test_prompts=None):
    """测试训练后的模型"""
    logger.info(f"=== 测试模型: {model_path} ===")
    
    if test_prompts is None:
        test_prompts = [
            "请介绍一下人工智能的发展历史。",
            "如何学习机器学习？",
            "请解释什么是深度学习。",
            "Python和Java有什么区别？",
            "请推荐一些好的编程学习资源。"
        ]
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info("模型加载成功，开始生成测试...")
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n=== 测试 {i+1}/{len(test_prompts)} ===")
            logger.info(f"问题: {prompt}")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"回答: {response}")
            logger.info("-" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 3B模型RLHF训练示例")
    parser.add_argument("--stage", choices=["all", "data", "sft", "ppo", "dpo", "test"],
                       default="all", help="要执行的训练阶段")
    parser.add_argument("--max_samples", type=int, default=5000,
                       help="每个数据集的最大样本数")
    parser.add_argument("--test_model_path", type=str, default="./output/ppo",
                       help="要测试的模型路径")
    
    args = parser.parse_args()
    
    logger.info("=== Qwen2.5 3B模型RLHF训练开始 ===")
    
    # 设置环境
    setup_environment()
    
    success = True
    
    # 数据准备阶段
    if args.stage in ["all", "data"]:
        if not prepare_data(args.max_samples):
            logger.error("数据准备失败")
            success = False
    
    # SFT训练阶段
    if success and args.stage in ["all", "sft"]:
        logger.info("=== 步骤2: 监督微调 (SFT) ===")
        sft_config = create_sft_config()
        if not run_training(sft_config, "SFT"):
            logger.error("SFT训练失败")
            success = False
    
    # PPO训练阶段
    if success and args.stage in ["all", "ppo"]:
        logger.info("=== 步骤3: PPO强化学习 ===")
        ppo_config = create_ppo_config()
        if not run_training(ppo_config, "PPO"):
            logger.error("PPO训练失败")
            success = False
    
    # DPO训练阶段（可选）
    if success and args.stage in ["dpo"]:
        logger.info("=== 步骤4: DPO偏好优化 ===")
        dpo_config = create_dpo_config()
        if not run_training(dpo_config, "DPO"):
            logger.error("DPO训练失败")
            success = False
    
    # 模型测试
    if args.stage in ["all", "test"] and os.path.exists(args.test_model_path):
        logger.info("=== 步骤5: 模型测试 ===")
        test_model(args.test_model_path)
    
    if success:
        logger.info("\n=== 训练流程完成 ===")
        logger.info("训练后的模型保存在以下位置:")
        logger.info(f"- SFT模型: ./output/sft")
        logger.info(f"- PPO模型: ./output/ppo")
        logger.info(f"- DPO模型: ./output/dpo")
        logger.info("\n可以使用以下命令测试模型:")
        logger.info(f"python {__file__} --stage test --test_model_path ./output/ppo")
    else:
        logger.error("训练过程中出现错误，请检查日志")

if __name__ == "__main__":
    main()