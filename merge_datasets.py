#!/usr/bin/env python3
"""
数据集合并脚本
用于手动合并多个数据集，支持不同的合并策略

使用方法:
python merge_datasets.py --config config.yaml --algorithm sft
python merge_datasets.py --config config.yaml --algorithm reward --strategy weighted_sampling
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils import load_config, setup_logger
from src.data.merger import DatasetMerger, merge_datasets_for_algorithm


def main():
    parser = argparse.ArgumentParser(description="数据集合并工具")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["sft", "reward", "rlhf"],
        required=True,
        help="算法类型"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["concat", "weighted_sampling", "balanced"],
        help="合并策略（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新合并，即使缓存文件存在"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示合并计划，不实际执行"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        config = load_config(args.config)
        logger.info(f"已加载配置文件: {args.config}")
        
        # 覆盖合并策略
        if args.strategy:
            if 'data' not in config:
                config['data'] = {}
            if 'merge_config' not in config['data']:
                config['data']['merge_config'] = {}
            config['data']['merge_config']['strategy'] = args.strategy
            logger.info(f"使用指定的合并策略: {args.strategy}")
        
        # 获取数据集配置
        data_config = config.get('data', {})
        datasets_config = data_config.get('datasets', {})
        algo_config = datasets_config.get(args.algorithm, {})
        
        if not algo_config:
            logger.error(f"配置文件中没有找到算法 {args.algorithm} 的数据集配置")
            sys.exit(1)
        
        train_files = algo_config.get('train_files', [])
        if not train_files:
            logger.error(f"算法 {args.algorithm} 没有配置训练文件")
            sys.exit(1)
        
        logger.info(f"找到 {len(train_files)} 个训练数据集:")
        for i, file_config in enumerate(train_files, 1):
            path = file_config.get('path', 'N/A')
            weight = file_config.get('weight', 1.0)
            name = file_config.get('name', 'N/A')
            logger.info(f"  {i}. {name}: {path} (权重: {weight})")
        
        # 检查是否需要合并
        if len(train_files) <= 1:
            logger.info("只有一个数据集，无需合并")
            if train_files:
                logger.info(f"数据集路径: {train_files[0].get('path')}")
            return
        
        if not algo_config.get('merge_datasets', False):
            logger.warning(f"算法 {args.algorithm} 的 merge_datasets 设置为 false，跳过合并")
            return
        
        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            output_path = algo_config.get('merged_cache_path')
            if not output_path:
                cache_dir = data_config.get('cache_dir', './cache')
                output_path = f"{cache_dir}/merged_{args.algorithm}_train.json"
        
        logger.info(f"输出路径: {output_path}")
        
        # 检查输出文件是否存在
        output_file = Path(output_path)
        if output_file.exists() and not args.force:
            logger.info(f"输出文件已存在: {output_path}")
            logger.info("使用 --force 参数强制重新合并")
            return
        
        if args.dry_run:
            logger.info("干运行模式，不实际执行合并")
            
            # 显示合并计划
            merger = DatasetMerger(config)
            strategy = merger.strategy
            logger.info(f"合并策略: {strategy}")
            
            # 加载数据集信息
            total_samples = 0
            for file_config in train_files:
                file_path = file_config.get('path')
                if Path(file_path).exists():
                    try:
                        data = merger.load_dataset_file(file_path)
                        samples = len(data)
                        total_samples += samples
                        logger.info(f"  {file_config.get('name')}: {samples} 样本")
                    except Exception as e:
                        logger.error(f"  {file_config.get('name')}: 加载失败 - {e}")
                else:
                    logger.warning(f"  {file_config.get('name')}: 文件不存在 - {file_path}")
            
            logger.info(f"总样本数: {total_samples}")
            return
        
        # 执行合并
        logger.info(f"开始合并算法 {args.algorithm} 的数据集...")
        
        merger = DatasetMerger(config)
        success = merger.merge_and_save(train_files, output_path, args.algorithm)
        
        if success:
            logger.info(f"数据集合并成功！输出文件: {output_path}")
            
            # 显示统计信息
            cache_dir = Path(data_config.get('cache_dir', './cache'))
            stats_file = cache_dir / f'merge_stats_{args.algorithm}.json'
            if stats_file.exists():
                logger.info(f"合并统计信息: {stats_file}")
        else:
            logger.error("数据集合并失败")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"执行失败: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()