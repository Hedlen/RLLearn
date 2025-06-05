#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件验证脚本
"""

import yaml
import os
import sys
import argparse
from pathlib import Path

def validate_config(config_path="config.yaml"):
    """验证配置文件的完整性和正确性"""
    print(f"=== 验证配置文件: {config_path} ===")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
            
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"❌ 配置文件为空或格式错误")
            return False
        
        # 检查必需的配置项
        required_keys = ['model', 'training', 'data']
        missing_keys = []
        
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ 缺少必需配置项: {', '.join(missing_keys)}")
            return False
        
        print("✅ 基础配置项检查通过")
        
        # 检查模型配置
        model_config = config.get('model', {})
        if 'model_name_or_path' not in model_config:
            print("❌ 缺少模型路径配置 (model.model_name_or_path)")
            return False
        
        model_path = model_config['model_name_or_path']
        if os.path.exists(model_path):
            print(f"✅ 本地模型路径存在: {model_path}")
        elif '/' in model_path:
            print(f"✅ 使用HuggingFace模型: {model_path}")
        else:
            print(f"⚠️  模型路径可能不存在: {model_path}")
        
        # 检查训练配置
        training_config = config.get('training', {})
        required_training_keys = ['output_dir', 'learning_rate']
        
        for key in required_training_keys:
            if key not in training_config:
                print(f"⚠️  建议添加训练配置: training.{key}")
        
        # 检查输出目录
        output_dir = training_config.get('output_dir', './output')
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"📁 将创建输出目录: {output_dir}")
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ 输出目录存在: {output_dir}")
        
        # 检查数据配置
        data_config = config.get('data', {})
        if 'datasets' in data_config:
            datasets = data_config['datasets']
            if not isinstance(datasets, list):
                print("❌ datasets配置应该是列表格式")
                return False
            
            missing_files = []
            for i, dataset in enumerate(datasets):
                if 'path' in dataset:
                    data_path = dataset['path']
                    if not os.path.exists(data_path):
                        missing_files.append(f"数据集 {i+1}: {data_path}")
                    else:
                        size = os.path.getsize(data_path) / 1024**2
                        print(f"✅ 数据集 {i+1}: {data_path} ({size:.1f}MB)")
                else:
                    print(f"⚠️  数据集 {i+1} 缺少path配置")
            
            if missing_files:
                print("❌ 以下数据文件不存在:")
                for file_info in missing_files:
                    print(f"   {file_info}")
                return False
        else:
            print("⚠️  未配置数据集 (data.datasets)")
        
        # 检查生成配置
        generation_config = config.get('generation', {})
        if not generation_config:
            print("⚠️  建议添加生成配置 (generation)")
        
        # 检查日志配置
        logging_config = config.get('logging', {})
        if 'log_file' in logging_config:
            log_file = logging_config['log_file']
            log_dir = Path(log_file).parent
            if not log_dir.exists():
                print(f"📁 将创建日志目录: {log_dir}")
                log_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n✅ 配置文件验证通过")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ YAML格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False

def generate_sample_config(output_path="config_sample.yaml"):
    """生成示例配置文件"""
    sample_config = {
        'model': {
            'model_name_or_path': 'Qwen/Qwen2.5-3B-Instruct',
            'trust_remote_code': True
        },
        'training': {
            'output_dir': './output',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 5e-6,
            'warmup_steps': 100,
            'logging_steps': 10,
            'save_steps': 500,
            'eval_steps': 500,
            'fp16': True,
            'dataloader_num_workers': 4,
            'remove_unused_columns': False,
            'report_to': ['tensorboard']
        },
        'data': {
            'max_length': 1024,
            'datasets': [
                {
                    'name': 'sample_dataset',
                    'path': './data/sample.json',
                    'type': 'sft'
                }
            ]
        },
        'generation': {
            'max_new_tokens': 256,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9
        },
        'logging': {
            'log_level': 'INFO',
            'log_file': './logs/training.log'
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 示例配置文件已生成: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="配置文件验证工具")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--generate_sample", action="store_true", help="生成示例配置文件")
    parser.add_argument("--output", default="config_sample.yaml", help="示例配置文件输出路径")
    
    args = parser.parse_args()
    
    if args.generate_sample:
        generate_sample_config(args.output)
        return True
    else:
        return validate_config(args.config)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)