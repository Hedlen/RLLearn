#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练工具：错误处理和性能优化
"""

import torch
import psutil
import gc
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import traceback
from datetime import datetime

class TrainingErrorHandler:
    """训练错误处理器"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_log = self.output_dir / "error_log.json"
        self.errors = []
    
    def handle_cuda_oom(self, error: Exception, config: Dict = None) -> Dict[str, Any]:
        """处理CUDA内存不足错误"""
        print("\n❌ CUDA内存不足错误")
        print(f"错误信息: {str(error)}")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            # 显示当前GPU使用情况
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB (缓存: {cached:.1f}GB)")
        
        # 生成优化建议
        suggestions = self._generate_memory_optimization(config)
        
        # 记录错误
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "CUDA_OOM",
            "error_message": str(error),
            "suggestions": suggestions,
            "config": config
        }
        
        self.errors.append(error_info)
        self._save_error_log()
        
        print("\n💡 内存优化建议:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
        
        return suggestions
    
    def handle_device_assertion_error(self, error: Exception) -> Dict[str, Any]:
        """处理设备断言错误"""
        print("\n❌ 设备断言错误")
        print(f"错误信息: {str(error)}")
        
        suggestions = [
            "检查所有张量是否在同一设备上",
            "确保模型和数据在相同的GPU上",
            "检查是否有张量仍在CPU上",
            "使用 .to(device) 将张量移动到正确设备",
            "检查多GPU设置是否正确"
        ]
        
        # 显示当前设备信息
        if torch.cuda.is_available():
            print(f"\n🔍 当前设备信息:")
            print(f"   CUDA设备数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                print(f"   设备 {i}: {torch.cuda.get_device_name(i)}")
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "DEVICE_ASSERTION",
            "error_message": str(error),
            "suggestions": suggestions
        }
        
        self.errors.append(error_info)
        self._save_error_log()
        
        print("\n💡 解决建议:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
        
        return {"suggestions": suggestions}
    
    def handle_model_loading_error(self, error: Exception, model_name: str) -> Dict[str, Any]:
        """处理模型加载错误"""
        print(f"\n❌ 模型加载错误: {model_name}")
        print(f"错误信息: {str(error)}")
        
        error_msg = str(error).lower()
        suggestions = []
        
        if "connection" in error_msg or "timeout" in error_msg:
            suggestions.extend([
                "检查网络连接",
                "设置HuggingFace镜像: export HF_ENDPOINT=https://hf-mirror.com",
                "使用代理或VPN",
                "手动下载模型到本地",
                "增加超时时间"
            ])
        elif "not found" in error_msg or "does not exist" in error_msg:
            suggestions.extend([
                "检查模型名称是否正确",
                "确认模型在HuggingFace Hub上存在",
                "检查本地路径是否正确",
                "尝试使用完整的模型路径"
            ])
        elif "permission" in error_msg or "access" in error_msg:
            suggestions.extend([
                "检查文件权限",
                "确认有访问模型的权限",
                "登录HuggingFace: huggingface-cli login",
                "检查私有模型的访问权限"
            ])
        else:
            suggestions.extend([
                "检查模型格式是否兼容",
                "更新transformers库版本",
                "尝试使用不同的模型版本",
                "检查磁盘空间是否充足"
            ])
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "MODEL_LOADING",
            "model_name": model_name,
            "error_message": str(error),
            "suggestions": suggestions
        }
        
        self.errors.append(error_info)
        self._save_error_log()
        
        print("\n💡 解决建议:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
        
        return {"suggestions": suggestions}
    
    def _generate_memory_optimization(self, config: Dict = None) -> List[str]:
        """生成内存优化建议"""
        suggestions = []
        
        if config:
            batch_size = config.get('training', {}).get('per_device_train_batch_size', 4)
            if batch_size > 2:
                suggestions.append(f"减少批次大小: 当前 {batch_size} -> 建议 {max(1, batch_size // 2)}")
            
            if not config.get('training', {}).get('fp16', False):
                suggestions.append("启用混合精度训练: fp16=True")
            
            if not config.get('training', {}).get('gradient_checkpointing', False):
                suggestions.append("启用梯度检查点: gradient_checkpointing=True")
            
            grad_accum = config.get('training', {}).get('gradient_accumulation_steps', 1)
            if grad_accum < 4:
                suggestions.append(f"增加梯度累积步数: 当前 {grad_accum} -> 建议 {grad_accum * 2}")
        
        # 通用建议
        suggestions.extend([
            "使用更小的模型",
            "减少序列长度",
            "启用CPU卸载",
            "使用DeepSpeed ZeRO",
            "清理不必要的变量"
        ])
        
        return suggestions
    
    def generate_optimized_config(self, original_config: Dict, error_type: str = "CUDA_OOM") -> Dict:
        """生成优化后的配置"""
        optimized_config = original_config.copy()
        
        if error_type == "CUDA_OOM":
            training_config = optimized_config.setdefault('training', {})
            
            # 减少批次大小
            current_batch = training_config.get('per_device_train_batch_size', 4)
            training_config['per_device_train_batch_size'] = max(1, current_batch // 2)
            
            # 启用内存优化
            training_config['fp16'] = True
            training_config['gradient_checkpointing'] = True
            training_config['dataloader_pin_memory'] = False
            
            # 增加梯度累积
            current_accum = training_config.get('gradient_accumulation_steps', 1)
            training_config['gradient_accumulation_steps'] = current_accum * 2
            
            # 减少序列长度
            if 'max_length' in training_config:
                current_length = training_config['max_length']
                training_config['max_length'] = min(current_length, 1024)
        
        # 保存优化配置
        optimized_path = self.output_dir / "optimized_config.yaml"
        with open(optimized_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n💾 优化配置已保存: {optimized_path}")
        return optimized_config
    
    def _save_error_log(self):
        """保存错误日志"""
        try:
            with open(self.error_log, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存错误日志失败: {e}")
    
    def print_error_summary(self):
        """打印错误摘要"""
        if not self.errors:
            print("✅ 没有记录的错误")
            return
        
        print(f"\n📋 错误摘要 (共 {len(self.errors)} 个错误):")
        
        error_types = {}
        for error in self.errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            print(f"   {error_type}: {count} 次")
        
        print(f"\n最近的错误:")
        for error in self.errors[-3:]:
            print(f"   [{error['timestamp']}] {error['error_type']}: {error['error_message'][:100]}...")

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024**3,
            "gpu_count": 0,
            "gpu_memory_gb": 0,
            "gpu_names": []
        }
        
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            total_gpu_memory = 0
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                info["gpu_names"].append(gpu_name)
                total_gpu_memory += gpu_memory
            
            info["gpu_memory_gb"] = total_gpu_memory
        
        return info
    
    def optimize_training_config(self, config: Dict = None) -> Dict:
        """优化训练配置"""
        if config is None:
            config = {}
        
        optimized = config.copy()
        training_config = optimized.setdefault('training', {})
        
        # 根据GPU内存优化批次大小
        if self.system_info["gpu_count"] > 0:
            gpu_memory = self.system_info["gpu_memory_gb"]
            
            if gpu_memory >= 24:
                # 高端GPU
                training_config.setdefault('per_device_train_batch_size', 8)
                training_config.setdefault('gradient_accumulation_steps', 2)
                training_config.setdefault('max_length', 2048)
            elif gpu_memory >= 16:
                # 中端GPU
                training_config.setdefault('per_device_train_batch_size', 4)
                training_config.setdefault('gradient_accumulation_steps', 4)
                training_config.setdefault('max_length', 1536)
            elif gpu_memory >= 8:
                # 入门GPU
                training_config.setdefault('per_device_train_batch_size', 2)
                training_config.setdefault('gradient_accumulation_steps', 8)
                training_config.setdefault('max_length', 1024)
                training_config['fp16'] = True
                training_config['gradient_checkpointing'] = True
            else:
                # 低端GPU
                training_config.setdefault('per_device_train_batch_size', 1)
                training_config.setdefault('gradient_accumulation_steps', 16)
                training_config.setdefault('max_length', 512)
                training_config['fp16'] = True
                training_config['gradient_checkpointing'] = True
                training_config['dataloader_pin_memory'] = False
        else:
            # CPU训练
            training_config.setdefault('per_device_train_batch_size', 1)
            training_config.setdefault('gradient_accumulation_steps', 32)
            training_config.setdefault('max_length', 512)
            training_config['dataloader_num_workers'] = min(4, self.system_info["cpu_count"])
        
        # 根据CPU核心数优化数据加载
        cpu_count = self.system_info["cpu_count"]
        if cpu_count >= 16:
            training_config.setdefault('dataloader_num_workers', 8)
        elif cpu_count >= 8:
            training_config.setdefault('dataloader_num_workers', 4)
        else:
            training_config.setdefault('dataloader_num_workers', 2)
        
        # 内存优化
        memory_gb = self.system_info["memory_gb"]
        if memory_gb < 16:
            training_config['dataloader_pin_memory'] = False
            training_config.setdefault('max_length', 512)
        
        return optimized
    
    def suggest_model_size(self) -> Dict[str, str]:
        """建议模型大小"""
        gpu_memory = self.system_info["gpu_memory_gb"]
        
        if gpu_memory >= 40:
            return {
                "recommended": "13B-30B",
                "models": ["Qwen2.5-14B", "Llama-2-13B", "Baichuan2-13B"],
                "reason": "高端GPU，可以训练大型模型"
            }
        elif gpu_memory >= 24:
            return {
                "recommended": "7B-13B",
                "models": ["Qwen2.5-7B", "Llama-2-7B", "ChatGLM3-6B"],
                "reason": "中高端GPU，适合中大型模型"
            }
        elif gpu_memory >= 16:
            return {
                "recommended": "3B-7B",
                "models": ["Qwen2.5-3B", "Phi-3-mini", "ChatGLM3-6B"],
                "reason": "中端GPU，适合中型模型"
            }
        elif gpu_memory >= 8:
            return {
                "recommended": "1B-3B",
                "models": ["Qwen2.5-1.5B", "Phi-3-mini", "TinyLlama-1.1B"],
                "reason": "入门GPU，建议小型模型"
            }
        else:
            return {
                "recommended": "<1B",
                "models": ["TinyLlama-1.1B", "DistilBERT"],
                "reason": "GPU内存不足，只能使用很小的模型"
            }
    
    def print_optimization_report(self, config: Dict = None):
        """打印优化报告"""
        print("\n🔧 性能优化报告")
        print("=" * 50)
        
        # 系统信息
        print(f"💻 系统配置:")
        print(f"   CPU: {self.system_info['cpu_count']} 核心")
        print(f"   内存: {self.system_info['memory_gb']:.1f}GB")
        
        if self.system_info["gpu_count"] > 0:
            print(f"   GPU: {self.system_info['gpu_count']} 个")
            print(f"   GPU内存: {self.system_info['gpu_memory_gb']:.1f}GB")
            for i, name in enumerate(self.system_info["gpu_names"]):
                print(f"     GPU {i}: {name}")
        else:
            print(f"   GPU: 无")
        
        # 模型建议
        model_suggestion = self.suggest_model_size()
        print(f"\n🤖 推荐模型大小: {model_suggestion['recommended']}")
        print(f"   原因: {model_suggestion['reason']}")
        print(f"   推荐模型:")
        for model in model_suggestion['models']:
            print(f"     • {model}")
        
        # 配置优化
        if config:
            optimized = self.optimize_training_config(config)
            training_config = optimized.get('training', {})
            
            print(f"\n⚙️  优化后的训练配置:")
            print(f"   批次大小: {training_config.get('per_device_train_batch_size', 'N/A')}")
            print(f"   梯度累积: {training_config.get('gradient_accumulation_steps', 'N/A')}")
            print(f"   最大长度: {training_config.get('max_length', 'N/A')}")
            print(f"   数据加载器工作进程: {training_config.get('dataloader_num_workers', 'N/A')}")
            print(f"   混合精度: {training_config.get('fp16', False)}")
            print(f"   梯度检查点: {training_config.get('gradient_checkpointing', False)}")
            
            # 计算有效批次大小
            batch_size = training_config.get('per_device_train_batch_size', 1)
            grad_accum = training_config.get('gradient_accumulation_steps', 1)
            gpu_count = max(1, self.system_info["gpu_count"])
            effective_batch = batch_size * grad_accum * gpu_count
            
            print(f"   有效批次大小: {effective_batch}")
        
        # 性能提示
        print(f"\n💡 性能优化提示:")
        
        if self.system_info["gpu_count"] == 0:
            print(f"   • 考虑使用GPU加速训练")
            print(f"   • CPU训练速度较慢，建议使用小模型")
        elif self.system_info["gpu_memory_gb"] < 8:
            print(f"   • GPU内存较小，启用fp16和梯度检查点")
            print(f"   • 考虑使用更小的批次大小")
        
        if self.system_info["memory_gb"] < 16:
            print(f"   • 系统内存较小，关闭pin_memory")
            print(f"   • 减少数据加载器工作进程数")
        
        if self.system_info["cpu_count"] < 4:
            print(f"   • CPU核心数较少，可能影响数据加载速度")
        
        print(f"   • 定期清理GPU缓存: torch.cuda.empty_cache()")
        print(f"   • 使用TensorBoard监控训练进度")
        print(f"   • 考虑使用DeepSpeed进一步优化")

def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练工具测试")
    parser.add_argument("--action", choices=["optimize", "error_test"], default="optimize", help="操作类型")
    parser.add_argument("--config", help="配置文件路径")
    
    args = parser.parse_args()
    
    if args.action == "optimize":
        optimizer = PerformanceOptimizer()
        
        config = None
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        optimizer.print_optimization_report(config)
        
        if config:
            optimized = optimizer.optimize_training_config(config)
            print(f"\n💾 优化后的配置:")
            print(yaml.dump(optimized, default_flow_style=False, allow_unicode=True))
    
    elif args.action == "error_test":
        handler = TrainingErrorHandler()
        
        # 模拟CUDA OOM错误
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        except Exception as e:
            handler.handle_cuda_oom(e, {"training": {"per_device_train_batch_size": 8}})
        
        handler.print_error_summary()

if __name__ == "__main__":
    main()