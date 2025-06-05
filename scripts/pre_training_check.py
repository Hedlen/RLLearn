#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练前置检查脚本
"""

import torch
import psutil
import os
import sys
import yaml
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

def check_system_resources():
    """检查系统资源"""
    print("=== 系统资源检查 ===")
    
    # GPU检查
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU")
        
        total_gpu_memory = 0
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            total_gpu_memory += gpu_memory
            
            # 检查GPU使用情况
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"      已分配: {allocated:.1f}GB, 已缓存: {cached:.1f}GB")
            
            if allocated / gpu_memory > 0.8:
                print(f"      ⚠️  GPU {i} 使用率较高 ({allocated/gpu_memory*100:.1f}%)")
        
        print(f"总GPU内存: {total_gpu_memory:.1f}GB")
        
        # 建议模型大小
        if total_gpu_memory >= 24:
            print("📊 建议模型: 7B或更大模型")
        elif total_gpu_memory >= 16:
            print("📊 建议模型: 3B-7B模型")
        elif total_gpu_memory >= 8:
            print("📊 建议模型: 1B-3B模型")
        else:
            print("📊 建议模型: 小于1B模型")
            
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
        print("   建议安装CUDA版本的PyTorch以获得更好性能")
    
    # 内存检查
    memory = psutil.virtual_memory()
    print(f"💾 系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
    
    if memory.available / 1024**3 < 4:
        print("⚠️  可用内存不足，可能影响训练")
        return False
    
    # 磁盘空间检查
    disk = psutil.disk_usage('.')
    print(f"💿 磁盘空间: {disk.free / 1024**3:.1f}GB 可用")
    
    if disk.free / 1024**3 < 10:
        print("⚠️  磁盘空间不足，建议至少保留10GB空间")
        return False
    
    return True

def check_model_accessibility(model_name):
    """检查模型是否可访问"""
    print(f"\n=== 模型访问检查: {model_name} ===")
    
    try:
        # 检查分词器
        print("检查分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ 分词器加载成功")
        print(f"   词汇表大小: {len(tokenizer)}")
        print(f"   特殊token: pad={tokenizer.pad_token}, eos={tokenizer.eos_token}")
        
        # 检查模型配置
        print("检查模型配置...")
        config = AutoConfig.from_pretrained(model_name)
        print(f"✅ 模型配置加载成功")
        print(f"   模型类型: {config.model_type}")
        print(f"   隐藏层大小: {config.hidden_size}")
        print(f"   注意力头数: {config.num_attention_heads}")
        print(f"   层数: {config.num_hidden_layers}")
        
        # 估算模型大小
        if hasattr(config, 'num_parameters'):
            params = config.num_parameters
        else:
            # 粗略估算
            params = config.hidden_size * config.num_hidden_layers * 12  # 简化估算
        
        params_b = params / 1e9
        print(f"   估算参数量: {params_b:.1f}B")
        
        # 估算内存需求
        memory_gb = params_b * 4 * 1.2  # 4字节/参数 + 20%开销
        print(f"   估算内存需求: {memory_gb:.1f}GB (仅推理)")
        print(f"   训练内存需求: {memory_gb * 3:.1f}GB (包含梯度和优化器)")
        
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "timeout" in error_msg:
            print(f"❌ 网络连接问题: {e}")
            print("   建议解决方案:")
            print("   1. 检查网络连接")
            print("   2. 使用镜像源: export HF_ENDPOINT=https://hf-mirror.com")
            print("   3. 手动下载模型到本地")
        else:
            print(f"❌ 模型访问失败: {e}")
        return False

def check_data_files(config_path="config.yaml"):
    """检查数据文件"""
    print(f"\n=== 数据文件检查 ===")
    
    try:
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在: {config_path}")
            return True  # 可能使用命令行参数
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'data' not in config:
            print("⚠️  配置中未找到data部分")
            return True
        
        data_config = config['data']
        
        if 'datasets' not in data_config:
            print("⚠️  配置中未找到datasets")
            return True
        
        datasets = data_config['datasets']
        if not isinstance(datasets, list):
            print("❌ datasets应该是列表格式")
            return False
        
        total_size = 0
        valid_datasets = 0
        
        for i, dataset in enumerate(datasets):
            if 'path' not in dataset:
                print(f"⚠️  数据集 {i+1}: 缺少path配置")
                continue
            
            path = dataset['path']
            if not os.path.exists(path):
                print(f"❌ 数据集 {i+1}: {path} 不存在")
                return False
            
            size = os.path.getsize(path) / 1024**2
            total_size += size
            valid_datasets += 1
            
            # 检查文件格式
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    if path.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            print(f"✅ 数据集 {i+1}: {path} ({size:.1f}MB, {len(data)} 条记录)")
                            
                            # 检查数据格式
                            sample = data[0]
                            if isinstance(sample, dict):
                                keys = list(sample.keys())
                                print(f"      数据字段: {keys}")
                            else:
                                print(f"      ⚠️  数据格式可能不正确")
                        else:
                            print(f"⚠️  数据集 {i+1}: {path} 格式可能不正确")
                    else:
                        print(f"✅ 数据集 {i+1}: {path} ({size:.1f}MB)")
            except Exception as e:
                print(f"⚠️  数据集 {i+1}: {path} 读取失败 - {e}")
        
        print(f"\n📊 数据统计:")
        print(f"   有效数据集: {valid_datasets}")
        print(f"   总数据大小: {total_size:.1f}MB")
        
        if valid_datasets == 0:
            print("❌ 没有有效的数据集")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据文件检查失败: {e}")
        return False

def check_training_config(config_path="config.yaml"):
    """检查训练配置"""
    print(f"\n=== 训练配置检查 ===")
    
    try:
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在: {config_path}")
            return True
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'training' not in config:
            print("⚠️  配置中未找到training部分")
            return True
        
        training_config = config['training']
        
        # 检查输出目录
        output_dir = training_config.get('output_dir', './output')
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"📁 将创建输出目录: {output_dir}")
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ 输出目录存在: {output_dir}")
        
        # 检查关键参数
        batch_size = training_config.get('per_device_train_batch_size', 4)
        grad_accum = training_config.get('gradient_accumulation_steps', 1)
        effective_batch = batch_size * grad_accum
        
        print(f"📊 训练参数:")
        print(f"   批次大小: {batch_size}")
        print(f"   梯度累积: {grad_accum}")
        print(f"   有效批次: {effective_batch}")
        print(f"   学习率: {training_config.get('learning_rate', 'N/A')}")
        print(f"   训练轮数: {training_config.get('num_train_epochs', 'N/A')}")
        
        # 检查是否启用了内存优化
        fp16 = training_config.get('fp16', False)
        gradient_checkpointing = training_config.get('gradient_checkpointing', False)
        
        print(f"🔧 内存优化:")
        print(f"   混合精度(fp16): {fp16}")
        print(f"   梯度检查点: {gradient_checkpointing}")
        
        # 根据GPU内存给出建议
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory < 8 and not fp16:
                print("💡 建议启用fp16以节省内存")
            
            if gpu_memory < 12 and batch_size > 2:
                print(f"💡 建议减少批次大小到2或更小")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练配置检查失败: {e}")
        return False

def estimate_training_time(config_path="config.yaml", model_name=None):
    """估算训练时间"""
    print(f"\n=== 训练时间估算 ===")
    
    try:
        # 基础参数
        base_time_per_step = 1.0  # 秒/步 (基准值)
        
        # GPU加速因子
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 24:
                gpu_factor = 0.3
            elif gpu_memory >= 16:
                gpu_factor = 0.5
            elif gpu_memory >= 8:
                gpu_factor = 0.8
            else:
                gpu_factor = 1.2
            
            gpu_factor /= gpu_count  # 多GPU加速
        else:
            gpu_factor = 10.0  # CPU训练很慢
        
        # 从配置文件读取参数
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            training_config = config.get('training', {})
            data_config = config.get('data', {})
            
            epochs = training_config.get('num_train_epochs', 3)
            batch_size = training_config.get('per_device_train_batch_size', 4)
            max_steps = training_config.get('max_steps')
            
            # 估算数据量
            total_samples = 0
            if 'datasets' in data_config:
                for dataset in data_config['datasets']:
                    if 'path' in dataset and os.path.exists(dataset['path']):
                        try:
                            import json
                            with open(dataset['path'], 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    total_samples += len(data)
                        except:
                            total_samples += 1000  # 默认估算
            
            if total_samples == 0:
                total_samples = 10000  # 默认估算
            
            # 计算步数
            if max_steps:
                total_steps = max_steps
            else:
                steps_per_epoch = total_samples // batch_size
                total_steps = steps_per_epoch * epochs
            
            # 估算时间
            time_per_step = base_time_per_step * gpu_factor
            total_time_seconds = total_steps * time_per_step
            
            hours = total_time_seconds // 3600
            minutes = (total_time_seconds % 3600) // 60
            
            print(f"📊 训练估算:")
            print(f"   总样本数: {total_samples:,}")
            print(f"   总步数: {total_steps:,}")
            print(f"   每步时间: {time_per_step:.1f}秒")
            print(f"   预计总时间: {hours:.0f}小时 {minutes:.0f}分钟")
            
            if hours > 24:
                print("⚠️  训练时间较长，建议考虑:")
                print("   1. 减少训练轮数")
                print("   2. 增加批次大小")
                print("   3. 使用更强的GPU")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练时间估算失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="训练前置检查")
    parser.add_argument("model_name", nargs='?', default="Qwen/Qwen2.5-3B-Instruct", help="模型名称")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--skip_model_check", action="store_true", help="跳过模型检查")
    
    args = parser.parse_args()
    
    print("🔍 开始训练前置检查...\n")
    
    checks = [
        lambda: check_system_resources(),
        lambda: check_data_files(args.config),
        lambda: check_training_config(args.config),
        lambda: estimate_training_time(args.config, args.model_name)
    ]
    
    if not args.skip_model_check:
        checks.insert(1, lambda: check_model_accessibility(args.model_name))
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "="*50)
    print("🎯 训练前置检查总结")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("✅ 所有检查通过，可以开始训练！")
        print("🚀 建议运行命令:")
        if args.config and os.path.exists(args.config):
            print(f"   python main.py --config {args.config}")
        else:
            print(f"   python example_training.py --model_name {args.model_name}")
        return True
    else:
        print(f"⚠️  {total - passed} 项检查未通过")
        print("请根据上述提示解决问题后重新检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)