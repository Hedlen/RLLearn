#!/usr/bin/env python3
"""
数据集合并工具
支持多个数据集的合并，包括权重采样、平衡采样等策略
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import numpy as np


class DatasetMerger:
    """数据集合并器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据集合并器
        
        Args:
            config: 配置字典，包含merge_config等信息
        """
        self.config = config
        self.merge_config = config.get('data', {}).get('merge_config', {})
        self.strategy = self.merge_config.get('strategy', 'concat')
        self.seed = self.merge_config.get('seed', 42)
        self.shuffle = self.merge_config.get('shuffle', True)
        self.max_merged_samples = self.merge_config.get('max_merged_samples')
        self.save_merge_stats = self.merge_config.get('save_merge_stats', True)
        
        # 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.logger = logging.getLogger(__name__)
    
    def load_dataset_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载数据集文件
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            数据集列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.warning(f"数据集文件不存在: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # 自动检测格式
                if file_path.suffix == '.jsonl' or not content.startswith('['):
                    # JSONL格式或者不是以[开头的文件
                    f.seek(0)
                    data = [json.loads(line.strip()) for line in f if line.strip()]
                else:
                    # JSON数组格式
                    data = json.loads(content)
            
            self.logger.info(f"成功加载数据集: {file_path}, 样本数: {len(data)}")
            return data
        except Exception as e:
            self.logger.error(f"加载数据集失败: {file_path}, 错误: {e}")
            return []
    
    def merge_datasets(
        self, 
        dataset_configs: List[Dict[str, Any]], 
        algorithm_type: str = 'sft'
    ) -> List[Dict[str, Any]]:
        """
        合并多个数据集
        
        Args:
            dataset_configs: 数据集配置列表，每个包含path, weight, name等信息
            algorithm_type: 算法类型 ('sft', 'reward', 'rlhf')
            
        Returns:
            合并后的数据集
        """
        if not dataset_configs:
            self.logger.warning("没有提供数据集配置")
            return []
        
        # 加载所有数据集
        datasets = []
        dataset_info = []
        
        for config in dataset_configs:
            file_path = config.get('path')
            weight = config.get('weight', 1.0)
            name = config.get('name', Path(file_path).stem)
            
            if not file_path:
                self.logger.warning(f"数据集配置缺少path字段: {config}")
                continue
            
            data = self.load_dataset_file(file_path)
            if data:
                datasets.append(data)
                dataset_info.append({
                    'name': name,
                    'path': file_path,
                    'weight': weight,
                    'size': len(data),
                    'data': data
                })
        
        if not datasets:
            self.logger.error("没有成功加载任何数据集")
            return []
        
        # 根据策略合并数据集
        if self.strategy == 'concat':
            merged_data = self._concat_merge(dataset_info)
        elif self.strategy == 'weighted_sampling':
            merged_data = self._weighted_sampling_merge(dataset_info)
        elif self.strategy == 'balanced':
            merged_data = self._balanced_merge(dataset_info)
        else:
            self.logger.warning(f"未知的合并策略: {self.strategy}, 使用concat策略")
            merged_data = self._concat_merge(dataset_info)
        
        # 限制样本数量
        if self.max_merged_samples and len(merged_data) > self.max_merged_samples:
            if self.shuffle:
                random.shuffle(merged_data)
            merged_data = merged_data[:self.max_merged_samples]
            self.logger.info(f"限制合并后样本数量为: {self.max_merged_samples}")
        
        # 打乱数据
        if self.shuffle:
            random.shuffle(merged_data)
            self.logger.info("已打乱合并后的数据")
        
        # 保存合并统计信息
        if self.save_merge_stats:
            self._save_merge_stats(dataset_info, merged_data, algorithm_type)
        
        self.logger.info(f"数据集合并完成，最终样本数: {len(merged_data)}")
        return merged_data
    
    def _concat_merge(self, dataset_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        简单拼接合并策略
        """
        merged_data = []
        for info in dataset_info:
            data = info['data']
            # 为每个样本添加数据集来源信息
            for item in data:
                item['_dataset_source'] = info['name']
                merged_data.append(item)
        
        self.logger.info(f"使用concat策略合并，总样本数: {len(merged_data)}")
        return merged_data
    
    def _weighted_sampling_merge(self, dataset_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        权重采样合并策略
        """
        # 计算总权重
        total_weight = sum(info['weight'] for info in dataset_info)
        
        # 计算每个数据集应该采样的数量
        total_samples = sum(info['size'] for info in dataset_info)
        merged_data = []
        
        for info in dataset_info:
            weight_ratio = info['weight'] / total_weight
            target_samples = int(total_samples * weight_ratio)
            
            data = info['data'].copy()
            if len(data) >= target_samples:
                # 随机采样
                sampled_data = random.sample(data, target_samples)
            else:
                # 重复采样
                sampled_data = data * (target_samples // len(data))
                remaining = target_samples % len(data)
                if remaining > 0:
                    sampled_data.extend(random.sample(data, remaining))
            
            # 添加数据集来源信息
            for item in sampled_data:
                item['_dataset_source'] = info['name']
                merged_data.append(item)
            
            self.logger.info(f"数据集 {info['name']}: 权重={info['weight']:.2f}, "
                           f"原始样本={info['size']}, 采样样本={len(sampled_data)}")
        
        return merged_data
    
    def _balanced_merge(self, dataset_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        平衡合并策略 - 每个数据集贡献相同数量的样本
        """
        # 找到最小的数据集大小
        min_size = min(info['size'] for info in dataset_info)
        merged_data = []
        
        for info in dataset_info:
            data = info['data'].copy()
            # 随机采样到最小大小
            if len(data) > min_size:
                sampled_data = random.sample(data, min_size)
            else:
                sampled_data = data
            
            # 添加数据集来源信息
            for item in sampled_data:
                item['_dataset_source'] = info['name']
                merged_data.append(item)
            
            self.logger.info(f"数据集 {info['name']}: 原始样本={info['size']}, "
                           f"平衡采样={len(sampled_data)}")
        
        return merged_data
    
    def _save_merge_stats(self, dataset_info: List[Dict[str, Any]], 
                         merged_data: List[Dict[str, Any]], algorithm_type: str):
        """
        保存合并统计信息
        """
        stats = {
            'merge_strategy': self.strategy,
            'algorithm_type': algorithm_type,
            'total_merged_samples': len(merged_data),
            'datasets': []
        }
        
        # 统计每个数据集的信息
        source_counts = defaultdict(int)
        for item in merged_data:
            source = item.get('_dataset_source', 'unknown')
            source_counts[source] += 1
        
        for info in dataset_info:
            dataset_stats = {
                'name': info['name'],
                'path': info['path'],
                'weight': info['weight'],
                'original_size': info['size'],
                'merged_size': source_counts.get(info['name'], 0),
                'contribution_ratio': source_counts.get(info['name'], 0) / len(merged_data) if merged_data else 0
            }
            stats['datasets'].append(dataset_stats)
        
        # 保存统计信息
        cache_dir = Path(self.config.get('data', {}).get('cache_dir', './cache'))
        cache_dir.mkdir(exist_ok=True)
        
        stats_file = cache_dir / f'merge_stats_{algorithm_type}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"合并统计信息已保存到: {stats_file}")
    
    def merge_and_save(
        self, 
        dataset_configs: List[Dict[str, Any]], 
        output_path: str,
        algorithm_type: str = 'sft'
    ) -> bool:
        """
        合并数据集并保存到文件
        
        Args:
            dataset_configs: 数据集配置列表
            output_path: 输出文件路径
            algorithm_type: 算法类型
            
        Returns:
            是否成功
        """
        try:
            merged_data = self.merge_datasets(dataset_configs, algorithm_type)
            if not merged_data:
                self.logger.error("合并后的数据集为空")
                return False
            
            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存合并后的数据集
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"合并后的数据集已保存到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"合并和保存数据集失败: {e}")
            return False
    
    def get_merged_dataset_path(
        self, 
        algorithm_type: str, 
        dataset_configs: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        获取合并后数据集的缓存路径
        
        Args:
            algorithm_type: 算法类型
            dataset_configs: 数据集配置列表
            
        Returns:
            缓存文件路径，如果不需要合并则返回None
        """
        data_config = self.config.get('data', {})
        datasets_config = data_config.get('datasets', {})
        algo_config = datasets_config.get(algorithm_type, {})
        
        # 检查是否需要合并
        if not algo_config.get('merge_datasets', False):
            return None
        
        # 如果只有一个数据集，不需要合并
        if len(dataset_configs) <= 1:
            return None
        
        # 获取缓存路径
        cache_path = algo_config.get('merged_cache_path')
        if cache_path:
            return cache_path
        
        # 生成默认缓存路径
        cache_dir = data_config.get('cache_dir', './cache')
        return f"{cache_dir}/merged_{algorithm_type}_train.json"


def merge_datasets_for_algorithm(config: Dict[str, Any], algorithm_type: str) -> Optional[str]:
    """
    为指定算法合并数据集
    
    Args:
        config: 完整配置
        algorithm_type: 算法类型 ('sft', 'reward', 'rlhf')
        
    Returns:
        合并后的数据集路径，如果不需要合并则返回None
    """
    logger = logging.getLogger(__name__)
    
    data_config = config.get('data', {})
    datasets_config = data_config.get('datasets', {})
    algo_config = datasets_config.get(algorithm_type, {})
    
    # 获取训练文件配置
    train_files = algo_config.get('train_files', [])
    if not train_files:
        logger.info(f"没有为算法 {algorithm_type} 配置训练文件")
        return None
    
    # 检查是否需要合并
    if not algo_config.get('merge_datasets', False) or len(train_files) <= 1:
        # 不需要合并，返回第一个文件路径
        if train_files:
            return train_files[0].get('path')
        return None
    
    # 创建合并器
    merger = DatasetMerger(config)
    
    # 获取缓存路径
    cache_path = merger.get_merged_dataset_path(algorithm_type, train_files)
    if not cache_path:
        logger.warning(f"无法确定算法 {algorithm_type} 的缓存路径")
        return None
    
    # 检查缓存是否存在且是最新的
    cache_file = Path(cache_path)
    if cache_file.exists():
        # 检查缓存是否比源文件新
        cache_mtime = cache_file.stat().st_mtime
        source_files_newer = False
        
        for file_config in train_files:
            source_file = Path(file_config.get('path', ''))
            if source_file.exists() and source_file.stat().st_mtime > cache_mtime:
                source_files_newer = True
                break
        
        if not source_files_newer:
            logger.info(f"使用缓存的合并数据集: {cache_path}")
            return str(cache_path)
    
    # 执行合并
    logger.info(f"开始合并算法 {algorithm_type} 的数据集...")
    success = merger.merge_and_save(train_files, cache_path, algorithm_type)
    
    if success:
        return str(cache_path)
    else:
        logger.error(f"合并算法 {algorithm_type} 的数据集失败")
        return None


def merge_validation_datasets_for_algorithm(config: Dict[str, Any], algorithm_type: str) -> Optional[str]:
    """
    为指定算法合并验证数据集
    
    Args:
        config: 完整配置
        algorithm_type: 算法类型 ('sft', 'reward', 'rlhf')
        
    Returns:
        合并后的验证数据集路径，如果不需要合并则返回None
    """
    logger = logging.getLogger(__name__)
    
    data_config = config.get('data', {})
    datasets_config = data_config.get('datasets', {})
    algo_config = datasets_config.get(algorithm_type, {})
    
    # 获取验证文件配置
    validation_files = algo_config.get('validation_files', [])
    if not validation_files:
        logger.info(f"没有为算法 {algorithm_type} 配置验证文件")
        return None
    
    # 检查是否需要合并
    if len(validation_files) <= 1:
        # 不需要合并，返回第一个文件路径
        if validation_files:
            return validation_files[0].get('path')
        return None
    
    # 创建合并器
    merger = DatasetMerger(config)
    
    # 获取缓存路径
    cache_path = algo_config.get('merged_validation_cache_path', 
                               f"./cache/merged_{algorithm_type}_validation.json")
    
    # 检查缓存是否存在且是最新的
    cache_file = Path(cache_path)
    if cache_file.exists():
        # 检查缓存是否比源文件新
        cache_mtime = cache_file.stat().st_mtime
        source_files_newer = False
        
        for file_config in validation_files:
            source_file = Path(file_config.get('path', ''))
            if source_file.exists() and source_file.stat().st_mtime > cache_mtime:
                source_files_newer = True
                break
        
        if not source_files_newer:
            logger.info(f"使用缓存的合并验证数据集: {cache_path}")
            return str(cache_path)
    
    # 执行合并
    logger.info(f"开始合并算法 {algorithm_type} 的验证数据集...")
    success = merger.merge_and_save(validation_files, cache_path, algorithm_type)
    
    if success:
        return str(cache_path)
    else:
        logger.error(f"合并算法 {algorithm_type} 的验证数据集失败")
        return None