# 验证数据集配置指南

本指南介绍如何在 RL Learning Framework 中配置和使用验证数据集（validation datasets）功能。

## 功能概述

新增的验证数据集功能包括：

1. **多验证数据集合并**：支持配置多个验证数据集文件，自动合并为单个验证集
2. **数据集拆分工具**：提供工具将单个数据集拆分为训练集和验证集
3. **配置文件自动处理**：根据配置文件自动拆分所有数据集并生成验证集配置

## 配置文件格式

### 基本配置

在配置文件中，每个算法类型（`sft`, `reward`, `rlhf`）都可以配置 `validation_files` 字段：

```yaml
data:
  datasets:
    sft:
      # 训练文件配置
      train_files:
        - path: "./data/sft_dataset1_train.json"
          weight: 1.0
          name: "general_sft"
        - path: "./data/sft_dataset2_train.json"
          weight: 0.8
          name: "domain_specific"
      
      # 验证文件配置 (新增)
      validation_files:
        - path: "./data/sft_dataset1_val.json"
          weight: 1.0
          name: "general_sft_val"
        - path: "./data/sft_dataset2_val.json"
          weight: 0.8
          name: "domain_specific_val"
      
      # 验证集合并缓存路径 (可选)
      merged_validation_cache_path: "./cache/merged_sft_validation.json"
```

### 字段说明

- `validation_files`: 验证数据集文件列表
  - `path`: 验证数据集文件路径
  - `weight`: 数据集权重（用于合并时的采样比例）
  - `name`: 数据集名称（用于日志和调试）
- `merged_validation_cache_path`: 合并后验证集的缓存路径（可选）

## 使用方法

### 1. 直接配置验证文件

如果你已经有分离的训练集和验证集文件，可以直接在配置文件中指定：

```yaml
data:
  datasets:
    sft:
      train_files:
        - path: "./data/train_data.json"
          weight: 1.0
          name: "training_data"
      
      validation_files:
        - path: "./data/val_data.json"
          weight: 1.0
          name: "validation_data"
```

### 2. 使用数据集拆分工具

#### 拆分单个数据集文件

```bash
# 将单个数据集文件拆分为训练集和验证集
python prepare_datasets.py --split_dataset ./data/full_dataset.json --train_ratio 0.8
```

参数说明：
- `--split_dataset`: 要拆分的数据集文件路径
- `--train_ratio`: 训练集比例（默认 0.8）
- `--shuffle`: 是否打乱数据（默认 True）
- `--random_seed`: 随机种子（默认 42）

#### 根据配置文件自动拆分

```bash
# 根据配置文件自动拆分所有数据集
python prepare_datasets.py --split_config ./config.yaml --algorithm_type sft --train_ratio 0.8
```

参数说明：
- `--split_config`: 配置文件路径
- `--algorithm_type`: 算法类型（sft, reward, rlhf）
- `--train_ratio`: 训练集比例
- `--shuffle`: 是否打乱数据
- `--random_seed`: 随机种子

这个命令会：
1. 读取配置文件中指定算法的 `train_files`
2. 将每个训练文件拆分为训练集和验证集
3. 更新配置文件，添加 `validation_files` 配置
4. 生成新的配置文件（原文件名 + `_split` 后缀）

### 3. 训练时的自动处理

在训练过程中，框架会自动：

1. 检查是否配置了 `validation_files`
2. 如果配置了多个验证文件，自动合并为单个验证集
3. 使用合并后的验证集进行模型评估

## 示例工作流程

### 场景1：从零开始准备数据集

```bash
# 1. 创建示例数据
python prepare_datasets.py --dataset sample --num_samples 1000

# 2. 拆分数据集
python prepare_datasets.py --split_dataset ./data/sft_train.json --train_ratio 0.8

# 3. 手动配置 validation_files 或使用配置文件拆分功能
```

### 场景2：已有配置文件，需要添加验证集

```bash
# 根据现有配置自动拆分并生成新配置
python prepare_datasets.py --split_config ./config.yaml --algorithm_type sft

# 使用生成的新配置文件进行训练
python main.py --config ./config_split.yaml --algorithm sft
```

### 场景3：多数据集合并验证

```yaml
# 配置多个验证数据集
data:
  datasets:
    sft:
      validation_files:
        - path: "./data/general_val.json"
          weight: 1.0
          name: "general"
        - path: "./data/domain_val.json"
          weight: 0.8
          name: "domain_specific"
        - path: "./data/quality_val.json"
          weight: 1.2
          name: "high_quality"
```

训练时会自动合并这些验证集，权重用于控制各数据集的采样比例。

## 注意事项

1. **数据格式一致性**：确保所有验证数据集文件的格式与训练数据集一致
2. **权重设置**：验证集权重影响合并时的采样比例，建议根据数据质量和重要性设置
3. **缓存管理**：合并后的验证集会缓存到指定路径，避免重复合并
4. **文件路径**：确保所有配置的文件路径都存在且可访问

## 兼容性

- 新功能完全向后兼容，不影响现有的单验证文件配置
- 如果没有配置 `validation_files`，系统会回退到原有的验证文件查找逻辑
- 支持所有训练算法：SFT、Reward Model、RLHF

## 故障排除

### 常见问题

1. **验证文件不存在**
   - 检查文件路径是否正确
   - 确保文件权限允许读取

2. **合并失败**
   - 检查数据格式是否一致
   - 查看日志中的详细错误信息

3. **缓存问题**
   - 删除缓存文件重新生成
   - 检查缓存目录的写入权限

### 调试技巧

- 设置日志级别为 DEBUG 查看详细信息
- 使用 `--validate_only` 参数验证数据格式
- 检查生成的合并文件内容是否正确