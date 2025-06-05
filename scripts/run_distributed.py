#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式训练启动脚本

使用 torch.distributed.launch 或 torchrun 启动多GPU分布式训练。
支持单机多卡和多机多卡训练。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_available_gpus():
    """获取可用的GPU数量"""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def run_single_node_training(args):
    """运行单节点多GPU训练"""
    print(f"启动单节点训练，使用 {args.nproc_per_node} 个GPU")
    
    # 检查是否使用DeepSpeed
    config_path = project_root / args.config
    use_deepspeed = False
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            training_config = config.get('training', {})
            hardware_config = config.get('hardware', {})
            distributed_config = hardware_config.get('distributed', {})
            
            # 获取分布式策略，优先使用新的配置结构
            distributed_strategy = distributed_config.get('strategy') or training_config.get('distributed_strategy', 'ddp')
            use_deepspeed = distributed_strategy == 'deepspeed'
            
            if use_deepspeed:
                print("检测到DeepSpeed配置，使用DeepSpeed启动")
        except Exception as e:
            print(f"读取配置文件失败: {e}，使用默认DDP模式")
    
    if use_deepspeed:
        # 使用deepspeed命令启动
        cmd = [
            "deepspeed",
            f"--num_gpus={args.nproc_per_node}",
            f"--master_port={args.master_port}",
            str(project_root / "main.py"),
            "--config", args.config,
            "--mode", "train",
            "--algorithm", args.algorithm
        ]
    else:
        # 使用torchrun命令启动
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
            str(project_root / "main.py"),
            "--config", args.config,
            "--mode", "train",
            "--algorithm", args.algorithm
        ]
    
    # 添加额外参数
    if args.extra_args:
        cmd.extend(args.extra_args.split())
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        print("训练完成！")
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        sys.exit(1)


def run_distributed_training(args):
    """运行分布式训练"""
    # 检查GPU可用性
    num_gpus = get_available_gpus()
    if num_gpus == 0:
        print("错误: 未检测到可用的GPU")
        return 1
    
    if args.nproc_per_node > num_gpus:
        print(f"警告: 请求的进程数 ({args.nproc_per_node}) 超过可用GPU数 ({num_gpus})")
        print(f"将使用 {num_gpus} 个GPU")
        args.nproc_per_node = num_gpus
    
    # 检查是否使用DeepSpeed
    config_path = project_root / args.config
    use_deepspeed = False
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            training_config = config.get('training', {})
            hardware_config = config.get('hardware', {})
            distributed_config = hardware_config.get('distributed', {})
            
            # 获取分布式策略，优先使用新的配置结构
            distributed_strategy = distributed_config.get('strategy') or training_config.get('distributed_strategy', 'ddp')
            use_deepspeed = distributed_strategy == 'deepspeed'
            
            if use_deepspeed:
                print("检测到DeepSpeed配置，使用DeepSpeed启动")
        except Exception as e:
            print(f"读取配置文件失败: {e}，使用默认DDP模式")
    
    # 构建命令
    if use_deepspeed:
        # 使用deepspeed命令启动多节点训练
        cmd = [
            "deepspeed",
            f"--num_nodes={args.nnodes}",
            f"--node_rank={args.node_rank}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
            f"--num_gpus={args.nproc_per_node}",
            str(project_root / "main.py"),
            "--config", args.config,
            "--mode", "train"
        ]
        
        if args.algorithm:
            cmd.extend(["--algorithm", args.algorithm])
    
    elif args.use_torchrun:
        # 使用 torchrun (推荐，PyTorch 1.10+)
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--nnodes={args.nnodes}",
            f"--node_rank={args.node_rank}",
        ]
        
        if args.master_addr:
            cmd.append(f"--master_addr={args.master_addr}")
        if args.master_port:
            cmd.append(f"--master_port={args.master_port}")
        
        cmd.extend([
            str(project_root / "main.py"),
            "--mode", "train",
            "--config", args.config
        ])
        
        if args.algorithm:
            cmd.extend(["--algorithm", args.algorithm])
    
    else:
        # 使用 torch.distributed.launch (兼容性)
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--nnodes={args.nnodes}",
            f"--node_rank={args.node_rank}",
        ]
        
        if args.master_addr:
            cmd.append(f"--master_addr={args.master_addr}")
        if args.master_port:
            cmd.append(f"--master_port={args.master_port}")
        
        cmd.extend([
            str(project_root / "main.py"),
            "--mode", "train",
            "--config", args.config
        ])
        
        if args.algorithm:
            cmd.extend(["--algorithm", args.algorithm])
    
    # 添加其他参数
    if args.extra_args:
        cmd.extend(args.extra_args.split())
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.nproc_per_node)))
    
    if args.backend:
        env['NCCL_BACKEND'] = args.backend
    
    # 打印命令信息
    print("=" * 60)
    print("启动分布式训练")
    print("=" * 60)
    print(f"GPU数量: {args.nproc_per_node}")
    print(f"节点数量: {args.nnodes}")
    print(f"当前节点rank: {args.node_rank}")
    print(f"配置文件: {args.config}")
    print(f"算法: {args.algorithm or '从配置文件读取'}")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 运行命令
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"训练失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="分布式训练启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 单机4卡训练:
   python scripts/run_distributed.py --nproc_per_node 4 --config config.yaml

2. 单机8卡训练，指定算法:
   python scripts/run_distributed.py --nproc_per_node 8 --config config.yaml --algorithm sft

3. 多机训练 (节点0):
   python scripts/run_distributed.py --nproc_per_node 4 --nnodes 2 --node_rank 0 \
          --master_addr 192.168.1.100 --config config.yaml

4. 多机训练 (节点1):
   python scripts/run_distributed.py --nproc_per_node 4 --nnodes 2 --node_rank 1 \
          --master_addr 192.168.1.100 --config config.yaml

5. 使用torchrun (推荐):
   python scripts/run_distributed.py --use_torchrun --nproc_per_node 4 --config config.yaml
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="训练配置文件路径"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["sft", "reward", "ppo", "dpo", "grpo"],
        help="训练算法 (如果不指定，从配置文件读取)"
    )
    
    # 分布式参数
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="每个节点的进程数 (通常等于GPU数，默认为可用GPU数)"
    )
    
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="节点总数 (默认: 1)"
    )
    
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="当前节点的rank (默认: 0)"
    )
    
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="主节点地址 (默认: localhost)"
    )
    
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="主节点端口 (默认: 29500)"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["nccl", "gloo"],
        default="nccl",
        help="分布式后端 (默认: nccl)"
    )
    
    # 启动方式
    parser.add_argument(
        "--use_torchrun",
        action="store_true",
        help="使用torchrun而不是torch.distributed.launch (推荐)"
    )
    
    # 其他参数
    parser.add_argument(
        "--extra_args",
        type=str,
        help="传递给main.py的额外参数 (用空格分隔)"
    )
    
    args = parser.parse_args()
    
    # 设置默认GPU数量
    if args.nproc_per_node is None:
        args.nproc_per_node = get_available_gpus()
        if args.nproc_per_node == 0:
            print("错误: 未检测到可用的GPU")
            return 1
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return 1
    
    return run_distributed_training(args)


if __name__ == "__main__":
    sys.exit(main())