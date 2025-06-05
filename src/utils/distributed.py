#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式训练工具模块

提供多GPU分布式训练的初始化、设置和管理功能。
支持 torch.distributed 和 accelerate 两种方式。
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
import logging


def is_distributed() -> bool:
    """检查是否在分布式环境中运行"""
    return (
        dist.is_available() and 
        dist.is_initialized() and 
        dist.get_world_size() > 1
    )


def get_rank() -> int:
    """获取当前进程的rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """获取本地rank（单机内的rank）"""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """检查是否为主进程"""
    return get_rank() == 0


def setup_distributed(backend: str = 'nccl') -> Dict[str, Any]:
    """初始化分布式训练环境
    
    Args:
        backend: 分布式后端，默认为'nccl'（GPU）或'gloo'（CPU）
        
    Returns:
        包含分布式信息的字典
    """
    logger = logging.getLogger(__name__)
    
    # 检查环境变量
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        logger.info("未检测到分布式环境变量，使用单GPU训练")
        return {
            'rank': 0,
            'local_rank': 0,
            'world_size': 1,
            'distributed': False,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
    
    # 获取分布式参数
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if backend == 'auto':
            backend = 'nccl'
    else:
        device = torch.device('cpu')
        if backend == 'auto':
            backend = 'gloo'
    
    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        logger.info(f"初始化分布式训练: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'distributed': True,
        'device': device,
        'backend': backend
    }


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    """同步所有进程"""
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """在所有进程间进行all-reduce操作"""
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """在所有进程间收集张量"""
    if not is_distributed():
        return tensor
    
    world_size = get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def reduce_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """对字典中的所有张量进行reduce操作"""
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    reduced_dict = {}
    
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            reduced_value = value.clone()
            dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)
            reduced_dict[key] = reduced_value / world_size
        else:
            reduced_dict[key] = value
    
    return reduced_dict


def save_on_master(state_dict: Dict[str, Any], filepath: str):
    """只在主进程保存模型"""
    if is_main_process():
        torch.save(state_dict, filepath)


def print_on_master(*args, **kwargs):
    """只在主进程打印信息"""
    if is_main_process():
        print(*args, **kwargs)


def log_on_master(logger: logging.Logger, level: int, msg: str, *args, **kwargs):
    """只在主进程记录日志"""
    if is_main_process():
        logger.log(level, msg, *args, **kwargs)


class DistributedSampler:
    """分布式数据采样器包装器"""
    
    def __init__(self, dataset, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        if is_distributed():
            from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler
            self.sampler = TorchDistributedSampler(
                dataset=dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle,
                drop_last=drop_last
            )
        else:
            from torch.utils.data import RandomSampler, SequentialSampler
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)
    
    def set_epoch(self, epoch: int):
        """设置epoch（用于分布式训练的随机性）"""
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        return iter(self.sampler)
    
    def __len__(self):
        return len(self.sampler)