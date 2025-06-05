#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的PEFT模型合并脚本

这个脚本帮助您:
1. 将训练好的LoRA/QLoRA适配器与基础模型合并
2. 保存为完整的模型，可以直接用于推理
3. 验证合并后的模型是否正常工作

使用示例:
    # 合并LoRA适配器
    python merge_peft_model.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_path ./outputs/sft_lora/checkpoint-1000 --output_path ./merged_model
    
    # 验证合并后的模型
    python merge_peft_model.py --validate ./merged_model
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("❌ PEFT库未安装，请运行: pip install peft")
    sys.exit(1)


def merge_peft_model(base_model_path: str, 
                    adapter_path: str, 
                    output_path: str,
                    device: str = "auto",
                    torch_dtype: str = "float16"):
    """
    合并PEFT适配器到基础模型
    
    Args:
        base_model_path: 基础模型路径或HuggingFace模型名
        adapter_path: LoRA适配器路径
        output_path: 合并后模型的保存路径
        device: 设备 (cpu, cuda, auto)
        torch_dtype: 数据类型 (float16, bfloat16, float32)
    """
    print("🚀 开始合并PEFT模型...")
    print(f"📁 基础模型: {base_model_path}")
    print(f"🔧 适配器路径: {adapter_path}")
    print(f"💾 输出路径: {output_path}")
    
    # 设置数据类型
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(torch_dtype, torch.float16)
    
    try:
        # 1. 加载基础模型
        print("\n📥 加载基础模型...")
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True
        }
        
        if device != "cpu":
            model_kwargs["device_map"] = "auto" if device == "auto" else device
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        print("✅ 基础模型加载成功")
        
        # 2. 加载分词器
        print("📝 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ 分词器加载成功")
        
        # 3. 加载PEFT适配器
        print("🔧 加载PEFT适配器...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ PEFT适配器加载成功")
        
        # 4. 合并适配器
        print("🔄 合并适配器到基础模型...")
        merged_model = model.merge_and_unload()
        print("✅ 模型合并成功")
        
        # 5. 保存合并后的模型
        print(f"💾 保存合并后的模型到: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        tokenizer.save_pretrained(output_path)
        
        # 6. 保存合并信息
        merge_info = {
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "output_path": output_path,
            "torch_dtype": str(torch_dtype),
            "device": device,
            "status": "success"
        }
        
        with open(os.path.join(output_path, "merge_info.json"), 'w', encoding='utf-8') as f:
            json.dump(merge_info, f, indent=2, ensure_ascii=False)
        
        print("\n🎉 模型合并完成!")
        print(f"📁 合并后的模型保存在: {output_path}")
        print("\n💡 使用方法:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{output_path}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{output_path}')")
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        raise


def validate_model(model_path: str, sample_text: str = "你好，请介绍一下自己。"):
    """
    验证模型是否可以正常工作
    
    Args:
        model_path: 模型路径
        sample_text: 测试文本
    """
    print(f"🔍 验证模型: {model_path}")
    
    try:
        # 加载模型和分词器
        print("📥 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ 模型加载成功")
        
        # 测试生成
        print(f"🧪 测试生成，输入: {sample_text}")
        inputs = tokenizer(sample_text, return_tensors="pt")
        
        # 移动到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 生成结果: {generated_text}")
        print("✅ 模型验证成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        return False


def check_adapter_info(adapter_path: str):
    """
    检查适配器信息
    
    Args:
        adapter_path: 适配器路径
    """
    print(f"🔍 检查适配器信息: {adapter_path}")
    
    # 检查适配器配置
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("📋 适配器配置:")
        print(f"   类型: {config.get('peft_type', 'Unknown')}")
        print(f"   任务类型: {config.get('task_type', 'Unknown')}")
        print(f"   基础模型: {config.get('base_model_name_or_path', 'Unknown')}")
        
        if 'r' in config:
            print(f"   LoRA rank: {config['r']}")
        if 'lora_alpha' in config:
            print(f"   LoRA alpha: {config['lora_alpha']}")
        if 'target_modules' in config:
            print(f"   目标模块: {config['target_modules']}")
        
        return config.get('base_model_name_or_path')
    else:
        print("❌ 未找到adapter_config.json文件")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="PEFT模型合并工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 合并LoRA适配器
  python merge_peft_model.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_path ./outputs/sft_lora/checkpoint-1000 --output_path ./merged_model
  
  # 验证合并后的模型
  python merge_peft_model.py --validate ./merged_model
  
  # 检查适配器信息
  python merge_peft_model.py --check_adapter ./outputs/sft_lora/checkpoint-1000
"""
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        help="基础模型路径或HuggingFace模型名 (如: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="LoRA适配器路径 (如: ./outputs/sft_lora/checkpoint-1000)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="合并后模型的保存路径 (如: ./merged_model)"
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="验证指定路径的模型是否正常工作"
    )
    parser.add_argument(
        "--check_adapter",
        type=str,
        help="检查指定路径的适配器信息"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="设备选择 (默认: auto)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch数据类型 (默认: float16)"
    )
    parser.add_argument(
        "--sample_text",
        type=str,
        default="你好，请介绍一下自己。",
        help="验证时使用的测试文本"
    )
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            # 验证模型
            validate_model(args.validate, args.sample_text)
        
        elif args.check_adapter:
            # 检查适配器信息
            base_model = check_adapter_info(args.check_adapter)
            if base_model:
                print(f"\n💡 建议的合并命令:")
                print(f"python merge_peft_model.py --base_model {base_model} --adapter_path {args.check_adapter} --output_path ./merged_model")
        
        else:
            # 合并模型
            if not args.base_model or not args.adapter_path or not args.output_path:
                print("❌ 合并操作需要指定 --base_model, --adapter_path 和 --output_path")
                print("使用 --help 查看详细用法")
                sys.exit(1)
            
            # 检查适配器路径
            if not os.path.exists(args.adapter_path):
                print(f"❌ 适配器路径不存在: {args.adapter_path}")
                sys.exit(1)
            
            # 检查适配器信息
            print("\n" + "="*50)
            base_model_from_config = check_adapter_info(args.adapter_path)
            
            if base_model_from_config and base_model_from_config != args.base_model:
                print(f"⚠️  警告: 适配器的基础模型 ({base_model_from_config}) 与指定的基础模型 ({args.base_model}) 不匹配")
                response = input("是否继续? (y/N): ")
                if response.lower() != 'y':
                    print("操作已取消")
                    sys.exit(0)
            
            print("="*50 + "\n")
            
            # 执行合并
            merge_peft_model(
                args.base_model,
                args.adapter_path,
                args.output_path,
                args.device,
                args.torch_dtype
            )
            
            # 询问是否验证
            response = input("\n是否验证合并后的模型? (Y/n): ")
            if response.lower() != 'n':
                print("\n" + "="*50)
                validate_model(args.output_path, args.sample_text)
    
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()