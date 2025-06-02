#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备脚本 - 为RL Learning Framework准备训练数据集

支持的数据集类型：
1. SFT (Supervised Fine-tuning) 数据
2. 偏好学习数据 (Preference Learning)
3. 对话数据 (Conversation Data)
4. 评估数据 (Evaluation Data)

支持的公开数据集：
- Anthropic HH-RLHF (英文偏好数据)
- BELLE (中文指令数据)
- Alpaca Chinese (中文指令数据)
- MOSS (中文对话数据)
- 本地样本数据生成
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
try:
    from datasets import load_dataset, Dataset
except ImportError:
    print("Warning: datasets library not installed. Some features may not work.")
    load_dataset = None
    Dataset = None

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers library not installed. Tokenizer features disabled.")
    AutoTokenizer = None

import pandas as pd
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """数据集准备器"""
    
    def __init__(self, output_dir: str = "./data", tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.output_dir = output_dir
        self.tokenizer_name = tokenizer_name
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/raw").mkdir(exist_ok=True)
        Path(f"{output_dir}/processed").mkdir(exist_ok=True)
        
        # 加载分词器
        if AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"无法加载分词器 {tokenizer_name}: {e}")
                self.tokenizer = None
        else:
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
    
    def create_sample_data(self, num_samples: int = 100) -> Dict[str, str]:
        """创建示例数据集"""
        logger.info(f"创建包含 {num_samples} 个样本的示例数据集...")
        
        # SFT 示例数据 - 更丰富的中文问答
        sft_samples = [
            {
                "prompt": "什么是人工智能？",
                "response": "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统，如学习、推理、感知、理解语言和解决问题。"
            },
            {
                "prompt": "解释机器学习的基本概念",
                "response": "机器学习是人工智能的一个子集，它使计算机能够从数据中学习并做出预测或决策，而无需明确编程每个可能的情况。它包括监督学习、无监督学习和强化学习等方法。"
            },
            {
                "prompt": "深度学习与传统机器学习的区别是什么？",
                "response": "深度学习使用多层神经网络来学习数据的复杂模式，能够自动提取特征，而传统机器学习通常需要手工特征工程。深度学习在图像、语音和自然语言处理等领域表现出色。"
            },
            {
                "prompt": "如何开始学习编程？",
                "response": "学习编程建议从以下步骤开始：1）选择一门适合初学者的语言如Python；2）学习基础语法和概念；3）通过小项目练习；4）阅读优秀代码；5）参与开源项目；6）持续练习和学习新技术。"
            },
            {
                "prompt": "什么是数据结构？",
                "response": "数据结构是计算机存储、组织数据的方式，包括数组、链表、栈、队列、树、图等。选择合适的数据结构能够提高程序的效率和性能，是编程的基础知识。"
            },
            {
                "prompt": "解释算法复杂度的概念",
                "response": "算法复杂度用于衡量算法执行效率，包括时间复杂度（执行时间）和空间复杂度（内存使用）。常用大O记号表示，如O(1)、O(n)、O(log n)等，帮助我们选择最优算法。"
            },
            {
                "prompt": "什么是云计算？",
                "response": "云计算是通过互联网提供计算服务的模式，包括服务器、存储、数据库、网络、软件等。它提供按需访问、弹性扩展、成本效益等优势，主要有IaaS、PaaS、SaaS三种服务模式。"
            },
            {
                "prompt": "区块链技术的基本原理是什么？",
                "response": "区块链是一种分布式账本技术，通过密码学方法将数据块按时间顺序链接。它具有去中心化、不可篡改、透明性等特点，广泛应用于加密货币、供应链管理、数字身份等领域。"
            },
            {
                "prompt": "前端开发需要掌握哪些技术？",
                "response": "前端开发主要需要掌握：1）基础技术：HTML、CSS、JavaScript；2）框架：React、Vue、Angular等；3）工具：Webpack、npm、Git等；4）响应式设计；5）性能优化；6）用户体验设计原则。"
            },
            {
                "prompt": "数据库的作用是什么？",
                "response": "数据库用于存储、管理和检索数据，提供数据的持久化存储、并发访问控制、数据完整性保证、备份恢复等功能。常见类型包括关系型数据库（MySQL、PostgreSQL）和非关系型数据库（MongoDB、Redis）。"
            }
        ]
        
        # 扩展到指定数量
        extended_sft = []
        for i in range(num_samples):
            base_sample = sft_samples[i % len(sft_samples)]
            if i >= len(sft_samples):
                # 添加变化避免完全重复
                extended_sft.append({
                    "prompt": f"请详细{base_sample['prompt']}",
                    "response": f"详细来说，{base_sample['response']}"
                })
            else:
                extended_sft.append(base_sample.copy())
        
        # 偏好数据示例 - 更丰富的对比
        preference_samples = [
            {
                "prompt": "如何学习编程？",
                "chosen": "学习编程需要循序渐进：首先选择一门适合的编程语言（如Python），然后学习基础语法，通过实际项目练习，阅读优秀代码，参与开源项目，并保持持续学习的习惯。",
                "rejected": "直接开始写复杂程序就行了，不需要学基础。"
            },
            {
                "prompt": "什么是数据结构？",
                "chosen": "数据结构是计算机存储、组织数据的方式，包括数组、链表、栈、队列、树、图等。选择合适的数据结构能够显著提高程序的效率和性能，是编程的重要基础知识。",
                "rejected": "数据结构就是存储数据的东西，没什么特别的。"
            },
            {
                "prompt": "解释机器学习的概念",
                "chosen": "机器学习是人工智能的一个重要分支，它使计算机能够从数据中自动学习模式和规律，并基于这些学习结果做出预测或决策，而无需为每种情况明确编程。",
                "rejected": "机器学习就是让机器自己学习，具体怎么学不重要。"
            },
            {
                "prompt": "云计算有什么优势？",
                "chosen": "云计算的主要优势包括：成本效益（按需付费）、弹性扩展（根据需求调整资源）、高可用性（多地备份）、易于维护（专业团队管理）、快速部署（即开即用）等。",
                "rejected": "云计算就是便宜一点，其他没什么特别的。"
            },
            {
                "prompt": "前端开发的核心技能是什么？",
                "chosen": "前端开发的核心技能包括：扎实的HTML/CSS/JavaScript基础、现代框架使用（React/Vue/Angular）、响应式设计、性能优化、用户体验设计、版本控制（Git）、构建工具使用等。",
                "rejected": "前端开发就是写写网页，会HTML就够了。"
            }
        ]
        
        # 扩展偏好数据
        extended_preference = []
        pref_count = max(num_samples // 2, 20)  # 至少20个偏好样本
        for i in range(pref_count):
            base_sample = preference_samples[i % len(preference_samples)]
            if i >= len(preference_samples):
                # 添加变化
                extended_preference.append({
                    "prompt": f"请解释{base_sample['prompt']}",
                    "chosen": base_sample['chosen'],
                    "rejected": base_sample['rejected']
                })
            else:
                extended_preference.append(base_sample.copy())
        
        # 创建评估数据
        eval_samples = [
            {
                "prompt": "什么是深度学习？",
                "reference": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂表示和模式。"
            },
            {
                "prompt": "如何提高编程技能？",
                "reference": "提高编程技能需要持续练习、学习新技术、参与项目、阅读优秀代码、与他人交流学习。"
            },
            {
                "prompt": "解释云计算的概念",
                "reference": "云计算是通过互联网提供按需计算资源和服务的模式，具有弹性、可扩展、成本效益等特点。"
            },
            {
                "prompt": "什么是区块链技术？",
                "reference": "区块链是一种分布式账本技术，通过密码学方法确保数据的安全性和不可篡改性。"
            },
            {
                "prompt": "前端开发的主要技术栈",
                "reference": "前端开发主要包括HTML、CSS、JavaScript以及现代框架如React、Vue、Angular等。"
            }
        ]
        
        # 扩展评估数据
        extended_eval = []
        eval_count = min(num_samples // 5, 20)  # 评估数据相对较少
        for i in range(eval_count):
            base_sample = eval_samples[i % len(eval_samples)]
            extended_eval.append(base_sample.copy())
        
        # 保存数据
        files = {}
        
        # SFT 数据
        sft_file = os.path.join(self.output_dir, "sft_train.json")
        with open(sft_file, 'w', encoding='utf-8') as f:
            json.dump(extended_sft, f, ensure_ascii=False, indent=2)
        files["sft"] = sft_file
        
        # 偏好数据
        pref_file = os.path.join(self.output_dir, "preference_train.json")
        with open(pref_file, 'w', encoding='utf-8') as f:
            json.dump(extended_preference, f, ensure_ascii=False, indent=2)
        files["preference"] = pref_file
        
        # 评估数据
        eval_file = os.path.join(self.output_dir, "eval.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(extended_eval, f, ensure_ascii=False, indent=2)
        files["eval"] = eval_file
        
        # 创建测试数据（与现有test.json兼容）
        test_file = os.path.join(self.output_dir, "test.json")
        if not os.path.exists(test_file):
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(extended_eval, f, ensure_ascii=False, indent=2)
            files["test"] = test_file
        
        logger.info(f"示例数据集创建完成：")
        logger.info(f"  SFT 数据: {sft_file} ({len(extended_sft)} 样本)")
        logger.info(f"  偏好数据: {pref_file} ({len(extended_preference)} 样本)")
        logger.info(f"  评估数据: {eval_file} ({len(extended_eval)} 样本)")
        
        return files
    
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
                "name_or_path": "Qwen/Qwen2.5-3B-Instruct",
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
                "max_length": 2048,
                "max_prompt_length": 1024
            },
            "generation": {
                "max_new_tokens": 1024,
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
    
    def validate_data(self, file_path: str, data_type: str) -> bool:
        """验证数据格式"""
        logger.info(f"验证 {data_type} 数据格式: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error(f"数据应该是列表格式，当前是 {type(data)}")
                return False
            
            if len(data) == 0:
                logger.error("数据为空")
                return False
            
            # 检查格式
            sample = data[0]
            
            if data_type == "sft":
                required_keys = ["prompt", "response"]
            elif data_type == "preference":
                required_keys = ["prompt", "chosen", "rejected"]
            elif data_type in ["eval", "test"]:
                required_keys = ["prompt"]
            else:
                required_keys = []
            
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                logger.error(f"缺少必需字段: {missing_keys}")
                return False
            
            logger.info(f"数据验证通过 ({len(data)} 样本)")
            return True
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="数据集准备脚本")
    parser.add_argument("--dataset", type=str, choices=["hh-rlhf", "belle", "alpaca-chinese", "moss", "sample", "all"], 
                       default="sample", help="要准备的数据集，使用'all'下载所有公开数据集")
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=100, help="样本数量")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="分词器名称")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--validate_only", action="store_true", help="仅验证现有数据")
    
    args = parser.parse_args()
    
    print("=== RL Learning Framework 数据集准备 ===")
    print(f"数据集: {args.dataset}")
    print(f"输出目录: {args.output_dir}")
    print(f"样本数量: {args.num_samples}")
    
    # 创建数据准备器
    preparer = DatasetPreparer(args.output_dir, args.tokenizer)
    
    if args.validate_only:
        # 仅验证现有数据
        files_to_validate = [
            ("sft_train.json", "sft"),
            ("preference_train.json", "preference"),
            ("eval.json", "eval"),
            ("test.json", "test")
        ]
        
        print("\n=== 验证现有数据 ===")
        for filename, data_type in files_to_validate:
            file_path = os.path.join(args.output_dir, filename)
            if os.path.exists(file_path):
                preparer.validate_data(file_path, data_type)
            else:
                print(f"⚠ 文件不存在: {file_path}")
        return
    
    if args.dataset == "sample":
        # 创建示例数据
        files = preparer.create_sample_data(args.num_samples)
        print("\n=== 示例数据集创建完成 ===")
        for data_type, file_path in files.items():
            print(f"{data_type}: {file_path}")
            # 验证创建的数据
            preparer.validate_data(file_path, data_type)
    
    elif args.dataset == "hh-rlhf":
        # 准备 HH-RLHF 数据集
        if load_dataset is None:
            print("错误: 需要安装 datasets 库来下载 HH-RLHF 数据集")
            print("请运行: pip install datasets")
            return
        
        files = preparer.prepare_hh_rlhf(args.num_samples, args.max_length)
        print("\n=== HH-RLHF 数据集准备完成 ===")
        for data_type, file_path in files.items():
            print(f"{data_type}: {file_path}")
    
    elif args.dataset == "belle":
        # 准备 BELLE 数据集
        if load_dataset is None:
            print("错误: 需要安装 datasets 库来下载 BELLE 数据集")
            print("请运行: pip install datasets")
            return
        
        files = preparer.prepare_belle(args.num_samples, args.max_length)
        print("\n=== BELLE 数据集准备完成 ===")
        for data_type, file_path in files.items():
            print(f"{data_type}: {file_path}")
    
    elif args.dataset == "alpaca-chinese":
        # 准备 Alpaca Chinese 数据集
        if load_dataset is None:
            print("错误: 需要安装 datasets 库来下载 Alpaca Chinese 数据集")
            print("请运行: pip install datasets")
            return
        
        files = preparer.prepare_alpaca_chinese(args.num_samples, args.max_length)
        print("\n=== Alpaca Chinese 数据集准备完成 ===")
        for data_type, file_path in files.items():
            print(f"{data_type}: {file_path}")
    
    elif args.dataset == "moss":
        # 准备 MOSS 数据集
        if load_dataset is None:
            print("错误: 需要安装 datasets 库来下载 MOSS 数据集")
            print("请运行: pip install datasets")
            return
        
        files = preparer.prepare_moss(args.num_samples, args.max_length)
        print("\n=== MOSS 数据集准备完成 ===")
        for data_type, file_path in files.items():
            print(f"{data_type}: {file_path}")
    
    elif args.dataset == "all":
        # 准备所有公开数据集
        if load_dataset is None:
            print("错误: 需要安装 datasets 库来下载公开数据集")
            print("请运行: pip install datasets")
            return
        
        print("\n=== 开始下载所有公开数据集 ===")
        all_files = {}
        
        # 下载 HH-RLHF 数据集
        print("\n1/4 准备 HH-RLHF 数据集...")
        try:
            files = preparer.prepare_hh_rlhf(args.num_samples, args.max_length)
            all_files.update({f"hh-rlhf_{k}": v for k, v in files.items()})
            print("✓ HH-RLHF 数据集准备完成")
        except Exception as e:
            print(f"✗ HH-RLHF 数据集准备失败: {e}")
        
        # 下载 BELLE 数据集
        print("\n2/4 准备 BELLE 数据集...")
        try:
            files = preparer.prepare_belle(args.num_samples, args.max_length)
            all_files.update({f"belle_{k}": v for k, v in files.items()})
            print("✓ BELLE 数据集准备完成")
        except Exception as e:
            print(f"✗ BELLE 数据集准备失败: {e}")
        
        # 下载 Alpaca Chinese 数据集
        print("\n3/4 准备 Alpaca Chinese 数据集...")
        try:
            files = preparer.prepare_alpaca_chinese(args.num_samples, args.max_length)
            all_files.update({f"alpaca-chinese_{k}": v for k, v in files.items()})
            print("✓ Alpaca Chinese 数据集准备完成")
        except Exception as e:
            print(f"✗ Alpaca Chinese 数据集准备失败: {e}")
        
        # 下载 MOSS 数据集
        print("\n4/4 准备 MOSS 数据集...")
        try:
            files = preparer.prepare_moss(args.num_samples, args.max_length)
            all_files.update({f"moss_{k}": v for k, v in files.items()})
            print("✓ MOSS 数据集准备完成")
        except Exception as e:
            print(f"✗ MOSS 数据集准备失败: {e}")
        
        print("\n=== 所有数据集准备完成 ===")
        print(f"成功准备了 {len(all_files)} 个数据文件:")
        for data_type, file_path in all_files.items():
            print(f"  {data_type}: {file_path}")
    
    print("\n=== 数据准备完成 ===")
    print("\n可用的数据文件:")
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith('.json'):
                print(f"  - {os.path.join(root, file)}")
    
    print("\n现在可以开始训练了！")
    print("\n使用示例：")
    print("  # 快速测试系统")
    print("  python quick_test.py")
    print("  ")
    print("  # 完整训练流程")
    print("  python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 50")
    print("  ")
    print("  # 使用配置文件训练")
    print("  python main.py --config config.yaml")

if __name__ == "__main__":
    main()