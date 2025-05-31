#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备脚本 - 为Qwen2.5 3B模型RLHF训练准备公开数据集

支持的数据集类型：
1. SFT (Supervised Fine-tuning) 数据
2. 偏好学习数据 (Preference Learning)
3. 对话数据 (Conversation Data)

支持的公开数据集：
- Anthropic HH-RLHF (英文偏好数据)
- BELLE (中文指令数据)
- Alpaca Chinese (中文指令数据)
- MOSS (中文对话数据)
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """数据集准备器"""
    
    def __init__(self, output_dir: str = "./data", tokenizer_name: str = "Qwen/Qwen2.5-3B"):
        self.output_dir = output_dir
        self.tokenizer_name = tokenizer_name
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"无法加载分词器 {tokenizer_name}: {e}")
            self.tokenizer = None
    
    def download_hh_rlhf(self, subset: str = "helpful-base", max_samples: Optional[int] = None) -> str:
        """
        下载Anthropic HH-RLHF数据集
        
        Args:
            subset: 数据子集 (helpful-base, harmless-base, helpful-online, harmless-online)
            max_samples: 最大样本数量
        
        Returns:
            保存的文件路径
        """
        logger.info(f"正在下载HH-RLHF数据集 - {subset}...")
        
        try:
            # 加载数据集
            dataset = load_dataset("Anthropic/hh-rlhf", data_dir=subset)
            
            # 处理训练集
            train_data = []
            for i, item in enumerate(tqdm(dataset['train'], desc="处理训练数据")):
                if max_samples and i >= max_samples:
                    break
                    
                # 解析对话
                chosen_conversation = item['chosen']
                rejected_conversation = item['rejected']
                
                # 提取prompt（通常是第一轮对话）
                prompt = self._extract_prompt_from_conversation(chosen_conversation)
                chosen_response = self._extract_response_from_conversation(chosen_conversation)
                rejected_response = self._extract_response_from_conversation(rejected_conversation)
                
                train_data.append({
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response
                })
            
            # 保存数据
            output_file = os.path.join(self.output_dir, f"hh_rlhf_{subset}_train.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"HH-RLHF {subset} 训练数据已保存到: {output_file}")
            logger.info(f"样本数量: {len(train_data)}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"下载HH-RLHF数据集失败: {e}")
            return None
    
    def download_belle_data(self, max_samples: Optional[int] = None) -> str:
        """
        下载BELLE中文指令数据集
        
        Args:
            max_samples: 最大样本数量
        
        Returns:
            保存的文件路径
        """
        logger.info("正在下载BELLE中文指令数据集...")
        
        try:
            # 加载BELLE数据集
            dataset = load_dataset("BelleGroup/train_2M_CN")
            
            # 处理数据
            train_data = []
            for i, item in enumerate(tqdm(dataset['train'], desc="处理BELLE数据")):
                if max_samples and i >= max_samples:
                    break
                
                # BELLE数据格式: {"instruction": "", "input": "", "output": ""}
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # 构建prompt
                if input_text:
                    prompt = f"{instruction}\n\n{input_text}"
                else:
                    prompt = instruction
                
                train_data.append({
                    "prompt": prompt,
                    "response": output_text
                })
            
            # 保存数据
            output_file = os.path.join(self.output_dir, "belle_sft_train.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"BELLE数据已保存到: {output_file}")
            logger.info(f"样本数量: {len(train_data)}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"下载BELLE数据集失败: {e}")
            return None
    
    def download_alpaca_chinese(self, max_samples: Optional[int] = None) -> str:
        """
        下载Alpaca中文数据集
        
        Args:
            max_samples: 最大样本数量
        
        Returns:
            保存的文件路径
        """
        logger.info("正在下载Alpaca中文数据集...")
        
        try:
            # 加载Alpaca中文数据集
            dataset = load_dataset("shibing624/alpaca-zh")
            
            # 处理数据
            train_data = []
            for i, item in enumerate(tqdm(dataset['train'], desc="处理Alpaca中文数据")):
                if max_samples and i >= max_samples:
                    break
                
                # Alpaca格式: {"instruction": "", "input": "", "output": ""}
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # 构建prompt
                if input_text:
                    prompt = f"{instruction}\n\n{input_text}"
                else:
                    prompt = instruction
                
                train_data.append({
                    "prompt": prompt,
                    "response": output_text
                })
            
            # 保存数据
            output_file = os.path.join(self.output_dir, "alpaca_chinese_sft_train.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Alpaca中文数据已保存到: {output_file}")
            logger.info(f"样本数量: {len(train_data)}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"下载Alpaca中文数据集失败: {e}")
            return None
    
    def create_synthetic_preference_data(self, sft_file: str, num_samples: int = 1000) -> str:
        """
        从SFT数据创建合成偏好数据
        
        Args:
            sft_file: SFT数据文件路径
            num_samples: 生成的偏好样本数量
        
        Returns:
            保存的偏好数据文件路径
        """
        logger.info(f"正在从{sft_file}创建合成偏好数据...")
        
        try:
            # 加载SFT数据
            with open(sft_file, 'r', encoding='utf-8') as f:
                sft_data = json.load(f)
            
            # 创建偏好数据
            preference_data = []
            for i in range(min(num_samples, len(sft_data))):
                item = sft_data[i]
                prompt = item['prompt']
                chosen = item['response']
                
                # 创建一个较差的回答（简单策略：截断或添加无关内容）
                if len(chosen) > 50:
                    rejected = chosen[:len(chosen)//2] + "..."
                else:
                    rejected = "我不知道。"
                
                preference_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
            
            # 保存偏好数据
            output_file = os.path.join(self.output_dir, "synthetic_preference_train.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(preference_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"合成偏好数据已保存到: {output_file}")
            logger.info(f"样本数量: {len(preference_data)}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"创建合成偏好数据失败: {e}")
            return None
    
    def _extract_prompt_from_conversation(self, conversation: str) -> str:
        """从对话中提取prompt"""
        lines = conversation.strip().split('\n')
        for line in lines:
            if line.startswith('Human:'):
                return line.replace('Human:', '').strip()
        return conversation.split('\n')[0].strip()
    
    def _extract_response_from_conversation(self, conversation: str) -> str:
        """从对话中提取回答"""
        lines = conversation.strip().split('\n')
        response_lines = []
        in_assistant = False
        
        for line in lines:
            if line.startswith('Assistant:'):
                in_assistant = True
                response_lines.append(line.replace('Assistant:', '').strip())
            elif in_assistant and not line.startswith('Human:'):
                response_lines.append(line.strip())
            elif line.startswith('Human:'):
                in_assistant = False
        
        return '\n'.join(response_lines).strip()
    
    def validate_data(self, file_path: str, data_type: str = "sft") -> Dict[str, Any]:
        """
        验证数据集质量
        
        Args:
            file_path: 数据文件路径
            data_type: 数据类型 (sft, preference)
        
        Returns:
            验证结果统计
        """
        logger.info(f"正在验证数据集: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats = {
                "total_samples": len(data),
                "avg_prompt_length": 0,
                "avg_response_length": 0,
                "empty_prompts": 0,
                "empty_responses": 0
            }
            
            prompt_lengths = []
            response_lengths = []
            
            for item in data:
                prompt = item.get('prompt', '')
                
                if data_type == "sft":
                    response = item.get('response', '')
                    if not response:
                        stats["empty_responses"] += 1
                    else:
                        response_lengths.append(len(response))
                        
                elif data_type == "preference":
                    chosen = item.get('chosen', '')
                    rejected = item.get('rejected', '')
                    if not chosen or not rejected:
                        stats["empty_responses"] += 1
                    else:
                        response_lengths.append(len(chosen))
                        response_lengths.append(len(rejected))
                
                if not prompt:
                    stats["empty_prompts"] += 1
                else:
                    prompt_lengths.append(len(prompt))
            
            if prompt_lengths:
                stats["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
            if response_lengths:
                stats["avg_response_length"] = sum(response_lengths) / len(response_lengths)
            
            logger.info(f"数据验证完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {}
    
    def create_train_config(self, datasets: List[str]) -> str:
        """
        创建训练配置文件
        
        Args:
            datasets: 数据集文件路径列表
        
        Returns:
            配置文件路径
        """
        config = {
            "model": {
                "name_or_path": "Qwen/Qwen2.5-3B",
                "cache_dir": "./cache"
            },
            "training": {
                "algorithm": "ppo",
                "output_dir": "./output",
                "num_epochs": 3,
                "learning_rate": 5e-5,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 500
            },
            "ppo": {
                "ppo_epochs": 4,
                "clip_range": 0.2,
                "value_loss_coef": 0.5,
                "entropy_coef": 0.01,
                "gamma": 0.99,
                "gae_lambda": 0.95
            },
            "data": {
                "datasets": datasets,
                "max_length": 512,
                "max_prompt_length": 256
            },
            "generation": {
                "max_new_tokens": 256,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0
            }
        }
        
        config_file = os.path.join(self.output_dir, "training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练配置已保存到: {config_file}")
        return config_file

def main():
    parser = argparse.ArgumentParser(description="准备RLHF训练数据集")
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B", help="分词器名称")
    parser.add_argument("--max_samples", type=int, default=None, help="每个数据集的最大样本数")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["hh-rlhf", "belle", "alpaca-chinese", "all"],
                       default=["all"], help="要下载的数据集")
    
    args = parser.parse_args()
    
    # 创建数据准备器
    preparer = DatasetPreparer(args.output_dir, args.tokenizer)
    
    downloaded_files = []
    
    # 下载数据集
    if "all" in args.datasets or "hh-rlhf" in args.datasets:
        logger.info("=== 下载HH-RLHF数据集 ===")
        hh_file = preparer.download_hh_rlhf("helpful-base", args.max_samples)
        if hh_file:
            downloaded_files.append(hh_file)
            preparer.validate_data(hh_file, "preference")
    
    if "all" in args.datasets or "belle" in args.datasets:
        logger.info("=== 下载BELLE数据集 ===")
        belle_file = preparer.download_belle_data(args.max_samples)
        if belle_file:
            downloaded_files.append(belle_file)
            preparer.validate_data(belle_file, "sft")
            
            # 从BELLE数据创建偏好数据
            pref_file = preparer.create_synthetic_preference_data(belle_file, 1000)
            if pref_file:
                downloaded_files.append(pref_file)
                preparer.validate_data(pref_file, "preference")
    
    if "all" in args.datasets or "alpaca-chinese" in args.datasets:
        logger.info("=== 下载Alpaca中文数据集 ===")
        alpaca_file = preparer.download_alpaca_chinese(args.max_samples)
        if alpaca_file:
            downloaded_files.append(alpaca_file)
            preparer.validate_data(alpaca_file, "sft")
    
    # 创建训练配置
    if downloaded_files:
        config_file = preparer.create_train_config(downloaded_files)
        logger.info(f"\n=== 数据准备完成 ===")
        logger.info(f"下载的数据集文件: {downloaded_files}")
        logger.info(f"训练配置文件: {config_file}")
        logger.info(f"\n可以使用以下命令开始训练:")
        logger.info(f"python main.py --config {config_file}")
    else:
        logger.error("没有成功下载任何数据集")

if __name__ == "__main__":
    main()