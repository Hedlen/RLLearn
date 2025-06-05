#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEFT模型管理脚本

功能:
1. 保存PEFT适配器 (LoRA/QLoRA)
2. 加载PEFT适配器
3. 合并PEFT适配器到基础模型
4. 验证模型兼容性

使用方法:
    python scripts/peft_model_manager.py --action save --model_path ./outputs/sft_lora/checkpoint-1000
    python scripts/peft_model_manager.py --action merge --base_model Qwen/Qwen2.5-7B-Instruct --adapter_path ./outputs/sft_lora/checkpoint-1000 --output_path ./merged_model
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)

try:
    from peft import (
        PeftModel,
        PeftConfig,
        LoraConfig,
        get_peft_model,
        TaskType
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT library not available. Please install with: pip install peft")

from src.utils import setup_logger


class PEFTModelManager:
    """PEFT模型管理器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("peft_manager")
        
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required. Please install with: pip install peft")
    
    def save_peft_adapter(self, 
                         model: Union[PeftModel, torch.nn.Module],
                         save_path: str,
                         tokenizer: Optional[AutoTokenizer] = None,
                         safe_serialization: bool = True) -> None:
        """
        保存PEFT适配器
        
        Args:
            model: PEFT模型或包含PEFT的模型
            save_path: 保存路径
            tokenizer: 分词器（可选）
            safe_serialization: 是否使用安全序列化
        """
        self.logger.info(f"Saving PEFT adapter to: {save_path}")
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 检查是否是PEFT模型
        if hasattr(model, 'save_pretrained'):
            # 如果是PEFT模型，直接保存适配器
            if isinstance(model, PeftModel):
                model.save_pretrained(
                    save_path,
                    safe_serialization=safe_serialization
                )
                self.logger.info("PEFT adapter saved successfully")
            else:
                # 如果不是PEFT模型，尝试保存整个模型
                self.logger.warning("Model is not a PEFT model, saving full model state")
                model.save_pretrained(
                    save_path,
                    safe_serialization=safe_serialization
                )
        else:
            # 如果没有save_pretrained方法，保存state_dict
            self.logger.warning("Model doesn't have save_pretrained method, saving state_dict")
            torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # 保存分词器
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
            self.logger.info("Tokenizer saved successfully")
        
        # 保存模型信息
        model_info = {
            "model_type": type(model).__name__,
            "is_peft_model": isinstance(model, PeftModel),
            "save_time": str(torch.utils.data.get_worker_info() or "unknown")
        }
        
        if isinstance(model, PeftModel):
            model_info["peft_config"] = model.peft_config
            model_info["base_model_name"] = getattr(model.base_model, 'name_or_path', 'unknown')
        
        with open(os.path.join(save_path, "model_info.json"), 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Model info saved to: {os.path.join(save_path, 'model_info.json')}")
    
    def load_peft_adapter(self,
                         base_model_path: str,
                         adapter_path: str,
                         device: Optional[str] = None,
                         torch_dtype: Optional[torch.dtype] = None) -> tuple:
        """
        加载PEFT适配器到基础模型
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: 适配器路径
            device: 设备
            torch_dtype: 数据类型
            
        Returns:
            (model, tokenizer) 元组
        """
        self.logger.info(f"Loading base model from: {base_model_path}")
        self.logger.info(f"Loading PEFT adapter from: {adapter_path}")
        
        # 加载基础模型
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype
        if device is not None:
            model_kwargs['device_map'] = device
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载PEFT适配器
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        self.logger.info("PEFT adapter loaded successfully")
        return model, tokenizer
    
    def merge_and_save(self,
                      base_model_path: str,
                      adapter_path: str,
                      output_path: str,
                      device: Optional[str] = None,
                      torch_dtype: Optional[torch.dtype] = None,
                      safe_serialization: bool = True,
                      max_shard_size: str = "5GB") -> None:
        """
        合并PEFT适配器到基础模型并保存
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: 适配器路径
            output_path: 输出路径
            device: 设备
            torch_dtype: 数据类型
            safe_serialization: 是否使用安全序列化
            max_shard_size: 最大分片大小
        """
        self.logger.info("Starting model merging process...")
        
        # 加载模型和适配器
        model, tokenizer = self.load_peft_adapter(
            base_model_path, adapter_path, device, torch_dtype
        )
        
        # 合并适配器
        self.logger.info("Merging PEFT adapter with base model...")
        merged_model = model.merge_and_unload()
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 保存合并后的模型
        self.logger.info(f"Saving merged model to: {output_path}")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size
        )
        
        # 保存分词器
        tokenizer.save_pretrained(output_path)
        
        # 保存合并信息
        merge_info = {
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "output_path": output_path,
            "merge_time": str(torch.utils.data.get_worker_info() or "unknown"),
            "model_size_gb": self._get_model_size(merged_model)
        }
        
        with open(os.path.join(output_path, "merge_info.json"), 'w', encoding='utf-8') as f:
            json.dump(merge_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Model merging completed successfully!")
        self.logger.info(f"Merged model saved to: {output_path}")
    
    def validate_peft_model(self,
                           model_path: str,
                           sample_text: str = "Hello, this is a test.") -> Dict[str, Any]:
        """
        验证PEFT模型
        
        Args:
            model_path: 模型路径
            sample_text: 测试文本
            
        Returns:
            验证结果字典
        """
        self.logger.info(f"Validating model at: {model_path}")
        
        results = {
            "model_path": model_path,
            "is_peft_model": False,
            "can_load": False,
            "can_generate": False,
            "error": None
        }
        
        try:
            # 尝试加载模型
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                # 这是一个PEFT适配器
                results["is_peft_model"] = True
                self.logger.info("Detected PEFT adapter")
                
                # 需要基础模型路径来完全验证
                with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")
                    if base_model_name:
                        self.logger.info(f"Base model: {base_model_name}")
                        results["base_model_name"] = base_model_name
            else:
                # 这是一个完整模型
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                results["can_load"] = True
                
                # 测试生成
                inputs = tokenizer(sample_text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 10,
                        do_sample=False
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results["can_generate"] = True
                results["sample_output"] = generated_text
                
                self.logger.info("Model validation successful")
        
        except Exception as e:
            results["error"] = str(e)
            self.logger.error(f"Model validation failed: {e}")
        
        return results
    
    def _get_model_size(self, model) -> float:
        """
        获取模型大小（GB）
        
        Args:
            model: 模型
            
        Returns:
            模型大小（GB）
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_gb = (param_size + buffer_size) / 1024**3
        return round(size_gb, 2)


def main():
    parser = argparse.ArgumentParser(description="PEFT模型管理工具")
    parser.add_argument(
        "--action",
        type=str,
        choices=["save", "load", "merge", "validate"],
        required=True,
        help="操作类型: save(保存适配器), load(加载适配器), merge(合并模型), validate(验证模型)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="基础模型路径（用于load和merge操作）"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="适配器路径"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="模型路径（用于save和validate操作）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="输出路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备 (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="PyTorch数据类型"
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="使用安全序列化"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger("peft_manager", level="INFO")
    
    # 设置数据类型
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.float16)
    
    # 创建管理器
    manager = PEFTModelManager(logger)
    
    try:
        if args.action == "save":
            if not args.model_path or not args.output_path:
                raise ValueError("save操作需要--model_path和--output_path参数")
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
            # 保存适配器
            manager.save_peft_adapter(model, args.output_path, tokenizer, args.safe_serialization)
        
        elif args.action == "load":
            if not args.base_model or not args.adapter_path:
                raise ValueError("load操作需要--base_model和--adapter_path参数")
            
            # 加载适配器
            model, tokenizer = manager.load_peft_adapter(
                args.base_model, args.adapter_path, args.device, torch_dtype
            )
            logger.info("模型和适配器加载成功")
        
        elif args.action == "merge":
            if not args.base_model or not args.adapter_path or not args.output_path:
                raise ValueError("merge操作需要--base_model、--adapter_path和--output_path参数")
            
            # 合并模型
            manager.merge_and_save(
                args.base_model,
                args.adapter_path,
                args.output_path,
                args.device,
                torch_dtype,
                args.safe_serialization
            )
        
        elif args.action == "validate":
            if not args.model_path:
                raise ValueError("validate操作需要--model_path参数")
            
            # 验证模型
            results = manager.validate_peft_model(args.model_path)
            logger.info(f"验证结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
    
    except Exception as e:
        logger.error(f"操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()