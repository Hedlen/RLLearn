# 多数据集配置和合并指南

本指南介绍如何在RL Learning框架中配置和使用多个数据集进行统一训练。

## 功能特性

- 🔄 **多数据集合并**: 支持合并多个SFT、Reward、RLHF数据集
- ⚖️ **权重采样**: 为不同数据集设置不同的采样权重
- 📊 **多种合并策略**: 支持拼接、权重采样、平衡采样等策略
- 💾 **智能缓存**: 自动缓存合并结果，避免重复计算
- 📈 **统计信息**: 详细的合并统计和数据集贡献分析
- 🔧 **向后兼容**: 完全兼容原有的单文件配置方式

## 配置方式

### 1. 基本配置结构

在 `config.yaml` 中，新的数据集配置结构如下：

```yaml
data:
  # 多数据集配置
  datasets:
    sft:  # SFT训练数据集
      train_files:
        - path: "./data/alpaca_chinese_sft.json"
          weight: 1.0
          name: "alpaca_chinese"
        - path: "./data/belle_sft.json"
          weight: 0.8
          name: "belle_conversation"
      validation_files:
        - path: "./data/alpaca_chinese_eval.json"
          name: "alpaca_chinese_eval"
      merge_datasets: true
      merged_cache_path: "./cache/merged_sft_train.json"
    
    reward:  # 奖励模型数据集
      train_files:
        - path: "./data/preference_train.json"
          weight: 1.0
          name: "preference_data"
      merge_datasets: true
      merged_cache_path: "./cache/merged_reward_train.json"
    
    rlhf:  # RLHF数据集（PPO/DPO等）
      train_files:
        - path: "./data/rlhf_train.json"
          weight: 1.0
          name: "rlhf_data"
      merge_datasets: false
      merged_cache_path: "./cache/merged_rlhf_train.json"
  
  # 数据合并配置
  merge_config:
    strategy: "weighted_sampling"  # 合并策略
    seed: 42                       # 随机种子
    shuffle: true                  # 是否打乱数据
    max_merged_samples: null       # 最大样本数限制
    save_merge_stats: true         # 保存统计信息
```

### 2. 数据集文件配置

每个数据集文件配置包含以下字段：

- `path`: 数据集文件路径（必需）
- `weight`: 采样权重，默认为1.0（可选）
- `name`: 数据集名称，用于标识和统计（可选）

### 3. 合并策略

支持三种合并策略：

#### concat（拼接）
- 简单地将所有数据集拼接在一起
- 保持每个数据集的原始大小比例
- 适用于数据集大小相近的情况

#### weighted_sampling（权重采样）
- 根据设置的权重对数据集进行采样
- 权重高的数据集会贡献更多样本
- 适用于希望某些数据集有更大影响的情况

#### balanced（平衡采样）
- 每个数据集贡献相同数量的样本
- 以最小数据集的大小为准
- 适用于希望各数据集影响相等的情况

## 使用方法

### 1. 训练时自动合并

使用标准训练命令，框架会自动检测并合并配置的数据集：

```bash
# SFT训练
python main.py --mode train --algorithm sft --config config.yaml

# 奖励模型训练
python main.py --mode train --algorithm reward --config config.yaml

# PPO训练
python main.py --mode train --algorithm ppo --config config.yaml
```

### 2. 手动合并数据集

使用专门的合并脚本：

```bash
# 合并SFT数据集
python merge_datasets.py --config config.yaml --algorithm sft

# 使用特定策略合并
python merge_datasets.py --config config.yaml --algorithm sft --strategy balanced

# 指定输出路径
python merge_datasets.py --config config.yaml --algorithm reward --output ./my_merged_data.json

# 干运行（只显示合并计划）
python merge_datasets.py --config config.yaml --algorithm sft --dry-run

# 强制重新合并
python merge_datasets.py --config config.yaml --algorithm sft --force
```

### 3. 查看合并统计

合并完成后，统计信息会保存在缓存目录中：

```bash
# 查看SFT合并统计
cat ./cache/merge_stats_sft.json

# 查看奖励模型合并统计
cat ./cache/merge_stats_reward.json
```

## 配置示例

### 示例1：多个SFT数据集

```yaml
data:
  datasets:
    sft:
      train_files:
        # 主要指令数据集
        - path: "./data/alpaca_chinese_sft.json"
          weight: 1.0
          name: "alpaca_chinese"
        
        # 对话数据集
        - path: "./data/belle_sft.json"
          weight: 0.8
          name: "belle_conversation"
        
        # 代码生成数据集
        - path: "./data/code_alpaca_sft.json"
          weight: 0.5
          name: "code_generation"
        
        # 数学推理数据集
        - path: "./data/math_sft.json"
          weight: 0.6
          name: "math_reasoning"
      
      merge_datasets: true
      merged_cache_path: "./cache/merged_sft_train.json"
  
  merge_config:
    strategy: "weighted_sampling"
    shuffle: true
    save_merge_stats: true
```

### 示例2：平衡多个偏好数据集

```yaml
data:
  datasets:
    reward:
      train_files:
        - path: "./data/hh_rlhf_preference.json"
          weight: 1.0
          name: "hh_rlhf"
        
        - path: "./data/custom_preference.json"
          weight: 1.0
          name: "custom_pref"
      
      merge_datasets: true
      merged_cache_path: "./cache/merged_reward_train.json"
  
  merge_config:
    strategy: "balanced"  # 平衡采样
    shuffle: true
```

## 向后兼容性

新的多数据集配置完全兼容原有的单文件配置方式：

```yaml
data:
  # 传统单文件配置（仍然支持）
  train_file: "./data/alpaca_chinese_sft.json"
  validation_file: "./data/eval.json"
  
  # 如果设置了train_file，将覆盖datasets配置
```

## 最佳实践

### 1. 数据集权重设置

- **高质量数据集**: 权重设为1.0或更高
- **辅助数据集**: 权重设为0.5-0.8
- **特定领域数据**: 根据需要调整权重

### 2. 合并策略选择

- **数据集质量相近**: 使用 `concat` 策略
- **数据集质量差异大**: 使用 `weighted_sampling` 策略
- **希望各数据集影响相等**: 使用 `balanced` 策略

### 3. 缓存管理

- 合并后的数据集会自动缓存，避免重复计算
- 如果源数据集更新，缓存会自动重新生成
- 使用 `--force` 参数可以强制重新合并

### 4. 内存优化

- 对于大型数据集，考虑设置 `max_merged_samples` 限制
- 使用 `preprocessing_num_workers` 加速数据处理

## 故障排除

### 常见问题

1. **数据集文件不存在**
   - 检查文件路径是否正确
   - 确保文件格式正确（JSON/JSONL）

2. **内存不足**
   - 设置 `max_merged_samples` 限制样本数量
   - 减少 `preprocessing_num_workers` 数量

3. **合并结果不符合预期**
   - 检查权重设置是否合理
   - 查看合并统计信息了解实际贡献比例
   - 尝试不同的合并策略

### 调试技巧

1. **使用干运行模式**
   ```bash
   python merge_datasets.py --config config.yaml --algorithm sft --dry-run
   ```

2. **查看详细日志**
   ```bash
   python merge_datasets.py --config config.yaml --algorithm sft --log-level DEBUG
   ```

3. **检查统计信息**
   ```bash
   cat ./cache/merge_stats_sft.json
   ```

## 高级功能

### 1. 自定义数据处理

可以在 `DatasetMerger` 类中添加自定义的数据处理逻辑：

```python
from src.data.merger import DatasetMerger

class CustomDatasetMerger(DatasetMerger):
    def _preprocess_data(self, data, dataset_name):
        # 自定义数据预处理逻辑
        return processed_data
```

### 2. 动态权重调整

可以根据训练进度动态调整数据集权重：

```python
def adjust_weights_by_epoch(epoch, base_weights):
    # 根据训练轮数调整权重
    adjusted_weights = {}
    for name, weight in base_weights.items():
        if epoch > 5:  # 后期减少某些数据集的权重
            adjusted_weights[name] = weight * 0.8
        else:
            adjusted_weights[name] = weight
    return adjusted_weights
```

## 总结

多数据集配置功能为RL Learning框架提供了强大的数据管理能力，支持：

- ✅ 灵活的多数据集配置
- ✅ 多种合并策略
- ✅ 智能缓存机制
- ✅ 详细的统计分析
- ✅ 完全向后兼容

通过合理配置和使用这些功能，可以显著提升模型训练的效果和效率。