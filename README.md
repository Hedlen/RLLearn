# 大模型强化学习框架 (RL Learning Framework)

一个专为大语言模型设计的强化学习训练框架，支持PPO、DPO、GRPO等主流算法，适用于RLHF（人类反馈强化学习）场景。

## 🚀 特性

- **多算法支持**: PPO、DPO、GRPO等强化学习算法
- **模块化设计**: 灵活的组件架构，易于扩展和定制
- **完整训练流程**: 从数据处理到模型训练的端到端解决方案
- **多种模型**: 策略模型、价值模型、奖励模型
- **丰富的数据处理**: 支持SFT、偏好学习、对话等多种数据格式
- **分布式训练支持**: 支持单节点多GPU和多节点分布式训练
  - **DDP (DistributedDataParallel)**: PyTorch原生分布式训练
  - **DeepSpeed**: 支持ZeRO优化器，内存高效训练大模型
  - **自动策略选择**: 根据模型大小自动选择最佳分布式策略
- **实时监控**: TensorBoard集成，实时训练监控
- **检查点管理**: 自动保存和恢复训练状态

## 📁 项目结构

```
rl_learning/
├── config.yaml              # 主配置文件
├── config_deepspeed_example.yaml # DeepSpeed配置示例
├── config_distributed_example.yaml # 分布式训练配置示例
├── config_examples.yaml     # 配置示例文件
├── config_rlhf_example.yaml # RLHF配置示例
├── main.py                   # 生产级训练框架主入口
├── example_training.py       # 完整训练示例脚本
├── prepare_datasets.py       # 数据集准备脚本
├── merge_datasets.py         # 数据集合并脚本
├── merge_peft_model.py       # PEFT模型合并脚本
├── quick_test.py            # 快速测试脚本
├── test_models.py           # 模型测试脚本
├── test_training.py         # 训练测试脚本
├── requirements.txt          # 依赖包列表
├── README.md                # 项目文档
├── TRAINING_MODES.md        # 训练模式说明
├── LICENSE                  # 开源协议
├── .gitignore              # Git忽略文件
├── configs/               # 配置文件目录
│   └── deepspeed/         # DeepSpeed配置
├── scripts/               # 脚本文件
│   ├── run_distributed.py # 分布式训练启动
│   ├── check_environment.py # 环境检查
│   ├── validate_config.py # 配置验证
│   ├── training_monitor.py # 训练监控
│   ├── pre_training_check.py # 训练前检查
│   ├── peft_model_manager.py # PEFT模型管理
│   └── training_utils.py   # 训练工具
├── data/                  # 数据目录
│   ├── processed/          # 处理后的数据
│   └── raw/               # 原始数据
├── test_data/             # 测试数据目录
│   ├── processed/          # 处理后的测试数据
│   └── raw/               # 原始测试数据
├── docs/                  # 文档目录
│   ├── DISTRIBUTED_TRAINING_GUIDE.md # 分布式训练指南
│   ├── DEEPSPEED_INTEGRATION.md # DeepSpeed集成
│   ├── PEFT_USAGE_GUIDE.md # PEFT使用指南
│   ├── LORA_QLORA_GUIDE.md # LoRA/QLoRA指南
│   ├── CUSTOM_REWARD_FUNCTIONS.md # 自定义奖励函数
│   ├── multi_datasets_guide.md # 多数据集指南
│   └── validation_datasets_guide.md # 验证数据集指南
└── src/                    # 核心源代码
    ├── algorithms/          # 强化学习算法
    │   ├── __init__.py
    │   ├── base.py         # 基础算法类
    │   ├── ppo.py          # PPO算法
    │   ├── dpo.py          # DPO算法
    │   ├── grpo.py         # GRPO算法
    │   └── utils.py        # 算法工具函数
    ├── data/               # 数据处理模块
    │   ├── __init__.py
    │   ├── processor.py    # 数据处理器
    │   ├── dataset.py      # 数据集类
    │   ├── collator.py     # 数据整理器
    │   ├── merger.py       # 数据合并器
    │   └── utils.py        # 数据工具函数
    ├── evaluators/         # 评估模块
    │   ├── __init__.py
    │   ├── model_evaluator.py    # 模型评估器
    │   ├── automatic_evaluator.py # 自动评估器
    │   ├── human_evaluator.py    # 人工评估器
    │   └── metrics.py      # 评估指标
    ├── models/             # 模型定义
    │   ├── __init__.py
    │   ├── policy_model.py # 策略模型
    │   ├── value_model.py  # 价值模型
    │   ├── reward_model.py # 奖励模型
    │   └── model_utils.py  # 模型工具函数
    ├── rewards/            # 奖励函数模块
    │   └── custom_rewards.py # 自定义奖励函数
    ├── trainers/           # 训练器
    │   ├── __init__.py
    │   ├── base_trainer.py # 基础训练器
    │   ├── sft_trainer.py  # SFT训练器
    │   ├── reward_trainer.py # 奖励模型训练器
    │   ├── ppo_trainer.py  # PPO训练器
    │   ├── dpo_trainer.py  # DPO训练器
    │   ├── grpo_trainer.py # GRPO训练器
    │   └── trainer_utils.py # 训练工具函数
    └── utils/              # 工具模块
        ├── __init__.py
        ├── logger.py       # 日志记录
        ├── config.py       # 配置管理
        ├── metrics.py      # 评估指标
        ├── checkpoint.py   # 检查点管理
        ├── distributed.py  # 分布式工具
        └── deepspeed_utils.py # DeepSpeed工具
```

## 🛠️ 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0 (可选，用于GPU加速)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd rl_learning
```

2. **创建虚拟环境**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
# 基础安装
pip install -r requirements.txt

# 验证安装是否成功
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"

# 环境检查脚本
python scripts/check_environment.py
```

### 🔧 配置文件验证

在开始训练前，建议验证配置文件的正确性：

```python
# scripts/validate_config.py
import yaml
import os
from pathlib import Path

def validate_config(config_path="config.yaml"):
    """验证配置文件的完整性和正确性"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查必需的配置项
        required_keys = ['model', 'training', 'data']
        for key in required_keys:
            if key not in config:
                print(f"❌ 缺少必需配置项: {key}")
                return False
        
        # 检查模型路径
        if 'model_name_or_path' in config['model']:
            model_path = config['model']['model_name_or_path']
            if not os.path.exists(model_path) and '/' not in model_path:
                print(f"⚠️  模型路径可能不存在: {model_path}")
        
        # 检查数据路径
        if 'datasets' in config['data']:
            for dataset in config['data']['datasets']:
                if 'path' in dataset:
                    data_path = dataset['path']
                    if not os.path.exists(data_path):
                        print(f"❌ 数据文件不存在: {data_path}")
                        return False
        
        print("✅ 配置文件验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False

if __name__ == "__main__":
    validate_config()
```

```bash
# 验证配置文件
python scripts/validate_config.py
```

## ⚙️ 配置

### 主配置文件 (config.yaml)

```yaml
# RL Learning Framework Configuration

# 模型配置
model:
  name_or_path: "Qwen/Qwen2.5-3B-Instruct"           # 基础模型路径
  cache_dir: "./cache"                       # 模型缓存目录

# 训练配置
training:
  algorithm: "ppo"                          # 算法类型: ppo, dpo, grpo
  output_dir: "./output"                    # 输出目录
  num_epochs: 3                             # 训练轮数
  learning_rate: 5e-5                       # 学习率
  per_device_train_batch_size: 4            # 每设备训练批大小
  gradient_accumulation_steps: 1            # 梯度累积步数
  max_grad_norm: 1.0                        # 梯度裁剪
  warmup_steps: 100                         # 预热步数
  logging_steps: 10                         # 日志记录间隔
  save_steps: 500                           # 保存间隔
  eval_steps: 500                           # 评估间隔

# PPO特定配置
ppo:
  ppo_epochs: 4                             # PPO训练轮数
  clip_range: 0.2                           # 裁剪范围
  value_loss_coef: 0.5                      # 价值损失系数
  entropy_coef: 0.01                        # 熵损失系数
  gamma: 0.99                               # 折扣因子
  gae_lambda: 0.95                          # GAE参数

# DPO特定配置
dpo:
  beta: 0.1                                 # DPO温度参数
  loss_type: "sigmoid"                      # 损失类型
  label_smoothing: 0.0                      # 标签平滑

# 数据配置
data:
  dataset_path: "./data"                    # 数据集路径
  max_length: 2048                          # 最大序列长度
  max_prompt_length: 1024                   # 最大提示长度

# 生成配置
generation:
  max_new_tokens: 1024                       # 最大生成长度
  temperature: 1.0                          # 采样温度
  top_k: 50                                 # Top-k采样
  top_p: 1.0                                # Top-p采样

# 日志配置
logging:
  level: "INFO"                             # 日志级别
  log_dir: "./logs"                         # 日志目录
  tensorboard: true                         # 启用TensorBoard```

## 🚀 快速开始

### ⚡ 零配置快速体验（推荐首次使用）

最快速的体验方式，无需任何配置，5分钟内完成完整流程：

```bash
# 1. 环境检查和依赖验证
python quick_test.py --check_env

# 2. 零配置快速体验（自动生成数据、快速训练、评估）
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 10 --quick_test

# 3. 查看训练结果
python main.py --config config.yaml --mode eval --quick_eval
```

### 方式一：使用示例脚本（推荐新手）

`example_training.py` 提供了完整的端到端训练示例，包含数据准备、模型训练和评估：

```bash
# 快速测试（使用小模型和少量步数）
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 50

# 使用Qwen/Qwen2.5-3B-Instruct进行完整训练
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 1000

# 仅运行SFT训练
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 100 --skip_reward --skip_rl

# 跳过训练，仅进行评估
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --skip_training
```

**example_training.py 特点**：
- ✅ 自动生成示例数据
- ✅ 完整的RLHF训练流程（SFT → Reward Model → PPO/DPO/GRPO）
- ✅ 内置模型评估和样本生成
- ✅ 适合学习和快速测试

### 方式二：使用生产级框架

`main.py` 是生产级训练框架，支持配置文件驱动和高级功能：

```bash
# 数据处理
python main.py --config config.yaml --mode data_process

# 1. SFT训练
python main.py --config config.yaml --mode train --algorithm sft

# 2. 奖励模型训练
python main.py --config config.yaml --mode train --algorithm reward

# 3. 强化学习训练（PPO/DPO/GRPO）
python main.py --config config.yaml --mode train --algorithm ppo
python main.py --config config.yaml --mode train --algorithm dpo
python main.py --config config.yaml --mode train --algorithm grpo

# 评估模型
python main.py --config config.yaml --mode eval
```

**main.py 特点**：
- ✅ 配置文件驱动，灵活性高
- ✅ 支持断点续训
- ✅ 生产级错误处理和日志
- ✅ 适合正式项目和大规模训练

#### 多数据集配置 🆕

现在支持配置多个数据集进行合并训练！在 `config.yaml` 中配置：

```yaml
data:
  datasets:
    sft:
      train_files:
        - path: "./data/alpaca_chinese_sft.json"
          weight: 1.0
          name: "alpaca_chinese"
        - path: "./data/belle_sft.json"
          weight: 0.8
          name: "belle_conversation"
      validation_files:  # 新增：验证数据集支持
        - path: "./data/alpaca_chinese_val.json"
          weight: 1.0
          name: "alpaca_chinese_val"
        - path: "./data/belle_val.json"
          weight: 0.8
          name: "belle_val"
      merge_datasets: true
      merged_cache_path: "./cache/merged_sft_train.json"
      merged_validation_cache_path: "./cache/merged_sft_validation.json"  # 新增：验证集缓存
  merge_config:
    strategy: "weighted_sampling"  # concat, weighted_sampling, balanced
    shuffle: true
```

详细配置请参考 [多数据集配置指南](docs/multi_datasets_guide.md)

#### 数据集合并（可选）

手动合并多个数据集：

```bash
# 合并SFT数据集
python merge_datasets.py --config config.yaml --algorithm sft

# 查看合并计划（不实际执行）
python merge_datasets.py --config config.yaml --algorithm sft --dry-run

# 使用特定策略合并
python merge_datasets.py --config config.yaml --algorithm sft --strategy balanced
```

### 快速测试系统

```bash
# 运行快速测试，验证环境配置
python quick_test.py
```

## 📊 数据准备

### 📋 支持的数据集

#### 1. 内置数据集

**Sample (示例数据)**
- **描述**: 内置的中文技术问答样本数据，用于快速测试
- **数据量**: 可配置（默认100个SFT样本，50个偏好样本，20个评估样本）
- **格式**: 包含SFT、偏好学习、评估三种格式
- **用途**: 快速开始、系统测试、算法验证
- **语言**: 中文

#### 2. 英文数据集

**HH-RLHF (Anthropic Human Feedback)**
- **描述**: Anthropic发布的人类偏好数据集，包含有用性和无害性两个维度
- **数据量**: ~161,000条对话
- **格式**: 偏好对比数据 (chosen vs rejected)
- **用途**: 奖励模型训练、DPO训练、GRPO训练
- **语言**: 英文

#### 3. 中文数据集

**BELLE**
- **描述**: 中文指令微调数据集
- **数据量**: ~2,000,000条指令-回答对
- **格式**: 指令跟随数据
- **用途**: SFT训练
- **语言**: 中文

**Alpaca Chinese**
- **描述**: Alpaca数据集的中文版本
- **数据量**: ~52,000条指令-回答对
- **格式**: 指令跟随数据
- **用途**: SFT训练
- **语言**: 中文

**MOSS**
- **描述**: 中文对话数据集
- **数据量**: 变量（根据配置）
- **格式**: 对话数据
- **用途**: SFT训练、对话能力提升
- **语言**: 中文

### 🚀 快速开始

#### 安装依赖

```bash
pip install datasets transformers pandas tqdm
```

#### 快速开始（推荐）

```bash
# 生成示例数据（最快方式，适合测试）
python prepare_datasets.py --dataset sample --num_samples 100 --output_dir ./data

# 验证数据格式
python prepare_datasets.py --validate_only

# 新增：数据集拆分功能
# 将单个数据集拆分为训练集和验证集
python prepare_datasets.py --split_dataset ./data/large_dataset.json --train_ratio 0.8 --shuffle --random_seed 42

# 根据配置文件批量拆分数据集
python prepare_datasets.py --split_config config.yaml --algorithm_type sft
```

#### 下载公开数据集

```bash
# 下载所有公开数据集（推荐）
python prepare_datasets.py --dataset all --num_samples 5000 --output_dir ./data

# 或者单独下载特定数据集
# 下载中文数据集
python prepare_datasets.py --dataset belle --num_samples 10000 --output_dir ./data
python prepare_datasets.py --dataset alpaca-chinese --num_samples 5000 --output_dir ./data
python prepare_datasets.py --dataset moss --num_samples 3000 --output_dir ./data

# 下载英文偏好数据
python prepare_datasets.py --dataset hh-rlhf --num_samples 5000 --output_dir ./data
```

**支持的数据集选项**：
- **sample**: 内置样本数据（中文技术问答）
- **all**: 下载所有公开数据集（HH-RLHF + BELLE + Alpaca Chinese + MOSS）
- **hh-rlhf**: Anthropic的英文偏好数据集（16万条）
- **belle**: 中文指令数据集（200万条）
- **alpaca-chinese**: 中文版Alpaca数据集（5万条）
- **moss**: 中文对话数据集

### 📊 数据集详细说明

#### HH-RLHF数据集

**数据格式示例**:
```json
{
  "prompt": "请解释什么是机器学习？",
  "chosen": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以识别模式并做出预测。",
  "rejected": "机器学习就是让机器学习。"
}
```

**使用场景**:
- DPO (Direct Preference Optimization) 训练
- 奖励模型训练
- PPO训练中的偏好评估

#### BELLE数据集

**数据格式示例**:
```json
{
  "prompt": "请介绍一下中国的传统节日春节。",
  "response": "春节是中国最重要的传统节日，也被称为农历新年。它标志着农历年的开始，通常在1月或2月举行。春节期间，人们会进行各种庆祝活动，如贴春联、放鞭炮、吃团圆饭、给红包等。这个节日象征着新的开始和家庭团聚。"
}
```

**使用场景**:
- 监督微调 (SFT)
- 生成合成偏好数据
- 中文对话能力训练

#### Alpaca Chinese数据集

**数据格式示例**:
```json
{
  "prompt": "解释以下概念：深度学习\n\n深度学习是什么？",
  "response": "深度学习是机器学习的一个子领域，它基于人工神经网络，特别是深层神经网络。深度学习模型能够自动学习数据的层次化表示，从简单的特征到复杂的概念。它在图像识别、自然语言处理、语音识别等领域取得了突破性进展。"
}
```

**使用场景**:
- 监督微调 (SFT)
- 中文指令跟随能力训练
- 基础对话能力建立

### 🔧 数据处理流程

#### 1. 数据下载和预处理

```python
from prepare_datasets import DatasetPreparer

# 创建数据准备器
preparer = DatasetPreparer(
    output_dir="./data",
    tokenizer_name="Qwen/Qwen2.5-3B-Instruct"
)

# 下载BELLE数据集
belle_file = preparer.download_belle_data(max_samples=5000)

# 验证数据质量
stats = preparer.validate_data(belle_file, "sft")
print(f"数据统计: {stats}")
```

#### 2. 创建偏好数据

```python
# 从SFT数据创建合成偏好数据
preference_file = preparer.create_synthetic_preference_data(
    sft_file=belle_file,
    num_samples=1000
)
```

#### 3. 数据验证

```python
# 验证偏好数据
stats = preparer.validate_data(preference_file, "preference")
print(f"偏好数据统计: {stats}")
```

### 📈 训练数据配置

#### SFT训练配置

```yaml
# config.yaml (SFT训练配置)
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  model_name_or_path: "Qwen/Qwen2.5-3B-Instruct"
  cache_dir: "./cache"

training:
  algorithm: "sft"
  output_dir: "./output/sft"
  num_epochs: 3
  learning_rate: 5e-5
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2

sft:
  max_length: 2048

data:
  dataset_path: "./data/belle_sft_train.json"
  max_length: 2048
  max_prompt_length: 1024
```

#### PPO训练配置

```yaml
# config.yaml (PPO训练配置)
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  model_name_or_path: "./output/sft/final_model"  # 使用SFT训练后的模型
  cache_dir: "./cache"

training:
  algorithm: "ppo"
  output_dir: "./output/ppo"
  num_epochs: 2
  learning_rate: 1e-5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4

ppo:
  ppo_epochs: 4
  clip_range: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  reward_model_path: "./output/reward/final_model"

data:
  dataset_path: "./data/synthetic_preference_train.json"
  max_length: 2048
```

#### DPO训练配置

```yaml
# config.yaml (DPO训练配置)
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  model_name_or_path: "./output/sft/final_model"
  cache_dir: "./cache"

training:
  algorithm: "dpo"
  output_dir: "./output/dpo"
  num_epochs: 1
  learning_rate: 5e-6
  per_device_train_batch_size: 2

dpo:
  beta: 0.1
  loss_type: "sigmoid"
  label_smoothing: 0.0

data:
  dataset_path: "./data/hh_rlhf_helpful-base_train.json"
  max_length: 2048
```

### 🎯 训练流程建议

#### 阶段1: 监督微调 (SFT)

```bash
# 1. 准备所有数据（推荐）
python prepare_datasets.py --dataset all --num_samples 10000 --output_dir ./data

# 或者单独准备SFT数据
# python prepare_datasets.py --dataset belle --num_samples 10000 --output_dir ./data
# python prepare_datasets.py --dataset alpaca-chinese --num_samples 5000 --output_dir ./data

# 2. 开始SFT训练
python main.py --config config.yaml --mode train --algorithm sft
```

**目标**: 让模型学会基本的对话和指令跟随能力

#### 阶段2: 奖励模型训练

```bash
# 1. 准备偏好数据（如果之前没有使用 --dataset all）
python prepare_datasets.py --dataset hh-rlhf --num_samples 5000 --output_dir ./data

# 2. 训练奖励模型
python main.py --config config.yaml --mode train --algorithm reward
```

**目标**: 训练奖励模型来评估回答质量

#### 阶段3: 偏好优化 (PPO/DPO/GRPO)

**选择PPO**:
```bash
# PPO训练（需要奖励模型）
python main.py --config config.yaml --mode train --algorithm ppo
```

**选择DPO**:
```bash
# DPO训练（无需奖励模型）
python main.py --config config.yaml --mode train --algorithm dpo
```

**选择GRPO**:
```bash
# GRPO训练（需要奖励模型）
python main.py --config config.yaml --mode train --algorithm grpo
```

**目标**: 让模型学会人类偏好，提高回答质量

### 📊 数据质量评估

#### 自动评估指标

```python
from src.utils import MetricsTracker

# 评估数据质量
metrics = {
    "avg_prompt_length": 45.2,
    "avg_response_length": 128.7,
    "empty_samples": 0,
    "total_samples": 5000
}

print(f"数据质量报告:")
print(f"- 平均提示长度: {metrics['avg_prompt_length']:.1f} 字符")
print(f"- 平均回答长度: {metrics['avg_response_length']:.1f} 字符")
print(f"- 空样本数量: {metrics['empty_samples']}")
print(f"- 总样本数量: {metrics['total_samples']}")
```

#### 手动质量检查

```python
import json
import random

# 随机检查数据样本
with open('./data/belle_sft_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 随机选择5个样本进行检查
samples = random.sample(data, 5)
for i, sample in enumerate(samples):
    print(f"\n=== 样本 {i+1} ===")
    print(f"提示: {sample['prompt']}")
    print(f"回答: {sample['response']}")
    print("-" * 50)
```

### 📚 进阶用法

#### 自定义数据集

```python
# 创建自定义数据集
custom_data = [
    {
        "prompt": "你的自定义问题",
        "response": "期望的回答"
    },
    # 更多数据...
]

# 保存为JSON格式
with open('./data/custom_sft_train.json', 'w', encoding='utf-8') as f:
    json.dump(custom_data, f, ensure_ascii=False, indent=2)
```

#### 数据混合策略

```python
# 混合多个数据集
def merge_datasets(file_paths, output_path, weights=None):
    all_data = []
    
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 应用权重
        if weights:
            data = data[:int(len(data) * weights[i])]
        
        all_data.extend(data)
    
    # 随机打乱
    random.shuffle(all_data)
    
    # 保存混合数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"混合数据集已保存: {output_path}")
    print(f"总样本数: {len(all_data)}")

# 使用示例
merge_datasets(
    ['./data/belle_sft_train.json', './data/alpaca_chinese_sft_train.json'],
    './data/mixed_sft_train.json',
    weights=[0.7, 0.3]  # BELLE占70%，Alpaca占30%
)
```

#### 数据增强

```python
# 简单的数据增强策略
def augment_data(input_file, output_file, augment_ratio=0.2):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    augmented_data = data.copy()
    
    # 随机选择一些样本进行增强
    samples_to_augment = random.sample(data, int(len(data) * augment_ratio))
    
    for sample in samples_to_augment:
        # 简单的增强：在prompt前添加礼貌用语
        augmented_sample = sample.copy()
        augmented_sample['prompt'] = f"请问，{sample['prompt']}"
        augmented_data.append(augmented_sample)
    
    # 保存增强后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据增强完成: {len(data)} -> {len(augmented_data)} 样本")

# 使用示例
augment_data('./data/belle_sft_train.json', './data/belle_sft_augmented.json')
```

### 🔍 数据相关故障排除

#### 常见问题

1. **下载失败**
   ```bash
   # 设置代理（如果需要）
   export HF_ENDPOINT=https://hf-mirror.com
   python prepare_datasets.py --dataset belle --num_samples 10000 --output_dir ./data
   ```

2. **内存不足**
   ```bash
   # 减少样本数量
   python prepare_datasets.py --dataset sample --num_samples 1000 --output_dir ./data
   ```

3. **分词器加载失败**
   ```bash
   # 使用本地分词器
   python prepare_datasets.py --tokenizer ./local_tokenizer
   ```

#### 数据格式验证

```python
# 验证数据格式是否正确
def validate_format(file_path, expected_format):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if expected_format == "sft":
        required_keys = ["prompt", "response"]
    elif expected_format == "preference":
        required_keys = ["prompt", "chosen", "rejected"]
    
    for i, item in enumerate(data[:5]):  # 检查前5个样本
        for key in required_keys:
            if key not in item:
                print(f"错误: 样本 {i} 缺少字段 '{key}'")
                return False
    
    print(f"数据格式验证通过: {expected_format}")
    return True

# 使用示例
validate_format('./data/belle_sft_train.json', 'sft')
validate_format('./data/hh_rlhf_helpful-base_train.json', 'preference')
```

### 数据格式

框架支持多种数据格式：

#### 1. SFT (Supervised Fine-tuning) 数据
```json
{
  "prompt": "用户问题或指令",
  "response": "期望的回答"
}
```

#### 2. 偏好学习数据
```json
{
  "prompt": "用户问题",
  "chosen": "更好的回答",
  "rejected": "较差的回答"
}
```

#### 3. 对话数据
```json
{
  "conversations": [
    {"role": "user", "content": "用户消息"},
    {"role": "assistant", "content": "助手回复"}
  ]
}
```

### 验证数据集功能

框架现在支持独立的验证数据集配置，提供更好的训练监控和模型评估：

#### 功能特性

- **多验证文件支持**: 可配置多个验证数据集文件
- **自动合并**: 验证数据集可自动合并并缓存
- **算法兼容**: 支持SFT、Reward和RLHF所有训练算法
- **数据集拆分**: 提供工具将大数据集拆分为训练集和验证集

#### 配置示例

```yaml
# config.yaml
data:
  sft:
    train_files:
      - "./data/sft_train1.json"
      - "./data/sft_train2.json"
    validation_files:  # 验证数据集配置
      - "./data/sft_val1.json"
      - "./data/sft_val2.json"
    merge_datasets: true
    merged_cache_path: "./data/merged_sft_train.json"
    merged_validation_cache_path: "./data/merged_sft_validation.json"
```

#### 使用数据集拆分工具

```bash
# 将单个数据集拆分为训练集和验证集
python prepare_datasets.py --split_dataset ./data/large_dataset.json \
    --train_ratio 0.8 --shuffle --random_seed 42

# 根据配置文件批量拆分
python prepare_datasets.py --split_config config.yaml --algorithm_type sft
```

详细使用指南请参考 `docs/validation_datasets_guide.md`。

### 数据预处理

```python
from src.data import DataProcessor

# 创建数据处理器
processor = DataProcessor(
    tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
    max_length=2048
)

# 加载和预处理数据
train_dataset = processor.load_dataset(
    "./data/belle_sft_train.json",
    dataset_type="sft"
)

eval_dataset = processor.load_dataset(
    "./data/hh_rlhf_helpful-base_train.json",
    dataset_type="preference"
)
```

## 🏋️ 训练

### 🔍 训练前置检查

在开始训练前，建议运行以下检查脚本：

```python
# scripts/pre_training_check.py
import torch
import psutil
import os
from transformers import AutoTokenizer, AutoModel

def check_system_resources():
    """检查系统资源"""
    print("=== 系统资源检查 ===")
    
    # GPU检查
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    # 内存检查
    memory = psutil.virtual_memory()
    print(f"💾 系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
    
    # 磁盘空间检查
    disk = psutil.disk_usage('.')
    print(f"💿 磁盘空间: {disk.free / 1024**3:.1f}GB 可用")
    
    if disk.free / 1024**3 < 10:
        print("⚠️  磁盘空间不足，建议至少保留10GB空间")

def check_model_accessibility(model_name):
    """检查模型是否可访问"""
    print(f"\n=== 模型访问检查: {model_name} ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ 分词器加载成功")
        
        # 尝试加载模型配置（不加载权重）
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        print(f"✅ 模型配置加载成功")
        print(f"   模型类型: {config.model_type}")
        print(f"   词汇表大小: {config.vocab_size}")
        
        return True
    except Exception as e:
        print(f"❌ 模型访问失败: {e}")
        return False

def check_data_files(config_path="config.yaml"):
    """检查数据文件"""
    print(f"\n=== 数据文件检查 ===")
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'datasets' in config.get('data', {}):
            for i, dataset in enumerate(config['data']['datasets']):
                if 'path' in dataset:
                    path = dataset['path']
                    if os.path.exists(path):
                        size = os.path.getsize(path) / 1024**2
                        print(f"✅ 数据集 {i+1}: {path} ({size:.1f}MB)")
                    else:
                        print(f"❌ 数据集 {i+1}: {path} 不存在")
                        return False
        return True
    except Exception as e:
        print(f"❌ 数据文件检查失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
    
    check_system_resources()
    check_model_accessibility(model_name)
    check_data_files()
    
    print("\n=== 检查完成 ===")
```

```bash
# 运行训练前检查
python scripts/pre_training_check.py Qwen/Qwen2.5-3B-Instruct
```

### 训练流程概述

本框架支持完整的RLHF训练流程，**必须按照以下顺序进行训练**：

1. **SFT (Supervised Fine-tuning)** - 监督微调基础模型
2. **Reward Model** - 训练奖励模型用于评估响应质量  
3. **强化学习训练** - 使用PPO、DPO或GRPO算法进一步优化模型

> ⚠️ **重要提示**: 
> - PPO和GRPO训练依赖奖励模型，必须先完成奖励模型训练！
> - DPO可以独立工作，不需要奖励模型
> - 推荐使用 `example_training.py` 进行完整流程训练

### 🌐 分布式训练支持

框架全面支持分布式训练，可显著加速大模型训练：

#### 支持的分布式策略

- **DDP (DistributedDataParallel)**: 适用于中小型模型（<1B参数）
- **DeepSpeed ZeRO**: 适用于大型模型，支持内存优化
  - ZeRO Stage 1: 优化器状态分片
  - ZeRO Stage 2: 梯度分片
  - ZeRO Stage 3: 参数分片，支持CPU/NVMe卸载
- **自动策略选择**: 根据模型大小自动选择最佳策略

#### 快速开始分布式训练

```bash
# 单节点4GPU训练（自动选择策略）
python scripts/run_distributed.py \
    --nproc_per_node 4 \
    --config config.yaml \
    --algorithm sft

# 使用DeepSpeed训练大模型
python scripts/run_distributed.py \
    --nproc_per_node 4 \
    --config config_deepspeed_example.yaml \
    --algorithm sft

# 多节点训练（节点1）
python scripts/run_distributed.py \
    --nproc_per_node 4 --nnodes 2 --node_rank 0 \
    --master_addr 192.168.1.100 --master_port 29500 \
    --config config.yaml
```

#### 配置示例

```yaml
# 启用分布式训练
distributed: true
distributed_strategy: "auto"  # 自动选择：ddp, deepspeed, auto
distributed_backend: "nccl"   # GPU使用nccl，CPU使用gloo

# DeepSpeed配置（可选）
deepspeed_config: "configs/deepspeed/zero2.json"
deepspeed_zero_stage: 2
deepspeed_cpu_offload: true
```

> 📖 **详细指南**: 完整的分布式训练配置和优化指南请参考 [`docs/DISTRIBUTED_TRAINING_GUIDE.md`](docs/DISTRIBUTED_TRAINING_GUIDE.md)
>
> 🚀 **DeepSpeed集成**: DeepSpeed功能详细说明请参考 [`DEEPSPEED_INTEGRATION.md`](DEEPSPEED_INTEGRATION.md)

### 使用示例脚本训练（推荐）

```bash
# 完整RLHF训练流程
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 1000

# 仅SFT训练
python example_training.py --model_name gpt2 --max_steps 100 --skip_reward --skip_rl

# 从SFT开始，跳过奖励模型，直接DPO训练
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 100 --skip_reward --algorithm dpo

# 使用更大模型
python example_training.py --model_name Qwen/Qwen2.5-7B --max_steps 500
```

### 命令行训练（高级用户）

使用 `main.py` 进行生产级训练，支持完整的RLHF流程：

#### 1. SFT训练

首先进行监督微调，为后续训练提供基础模型：

```bash
# 使用main.py进行SFT训练
python main.py --config config.yaml --mode train --algorithm sft
```

或者使用传统方式：

```python
from src.trainers import SFTTrainer, SFTTrainingConfig
from src.models import create_policy_model
from transformers import AutoTokenizer

# 加载模型和分词器
model, tokenizer = create_policy_model("Qwen/Qwen2.5-3B-Instruct")

# 配置训练参数
config = SFTTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    output_dir="./output/sft",
    num_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    max_length=2048
)

# 创建训练器
trainer = SFTTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset
)

# 开始训练
training_result = trainer.train()

# 保存模型
trainer.save_model("./output/sft/final_model")
```

#### 2. Reward Model训练

使用偏好数据训练奖励模型：

```bash
# 使用main.py进行奖励模型训练
python main.py --config config.yaml --mode train --algorithm reward
```

或者使用传统方式：

```python
from src.trainers import RewardModelTrainer, RewardTrainingConfig
from src.models import create_reward_model

# 加载模型（基于SFT模型或基础模型）
reward_model, tokenizer = create_reward_model("./output/sft/final_model")  # 推荐使用SFT模型

# 配置训练参数
config = RewardTrainingConfig(
    model_name_or_path="./output/sft/final_model",
    output_dir="./output/reward",
    num_epochs=3,
    learning_rate=5e-5,
    margin=0.0,
    loss_type="ranking"
)

# 创建训练器
trainer = RewardModelTrainer(
    config=config,
    model=reward_model,
    tokenizer=tokenizer,
    train_dataset=preference_dataset
)

# 开始训练
training_result = trainer.train()

# 保存模型
trainer.save_model("./output/reward/final_model")
```

#### 3. 强化学习训练

完成SFT和奖励模型训练后，可以选择以下算法进行强化学习训练：

**PPO训练**

```bash
# 使用main.py进行PPO训练
python main.py --config config.yaml --mode train --algorithm ppo
```

或者使用传统方式：

```python
from src.trainers import PPOTrainer, PPOTrainingConfig
from src.models import create_policy_model, create_value_model, create_reward_model
from transformers import AutoTokenizer

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
policy_model, _ = create_policy_model("./output/sft/final_model")  # 使用SFT模型作为初始策略模型
value_model, _ = create_value_model("Qwen/Qwen2.5-3B-Instruct")
reward_model, _ = create_reward_model("./output/reward/final_model")  # 加载训练好的奖励模型

# 配置训练参数
config = PPOTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    output_dir="./output/ppo",
    num_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    ppo_epochs=4,
    clip_range=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    reward_model_path="./output/reward/final_model"  # 奖励模型路径
)

# 创建训练器
trainer = PPOTrainer(
    config=config,
    policy_model=policy_model,
    value_model=value_model,
    tokenizer=tokenizer,
    reward_model=reward_model,  # 传入奖励模型
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader
)

# 开始训练
training_result = trainer.train()

# 保存模型
trainer.save_model("./output/ppo/final_model")
```

**DPO训练**

```bash
# 使用main.py进行DPO训练
python main.py --config config.yaml --mode train --algorithm dpo
```

DPO不需要奖励模型，可以直接使用偏好数据进行训练。或者使用传统方式：

```python
from src.trainers import DPOTrainer, DPOTrainingConfig
from src.models import create_policy_model

# 加载模型（使用SFT模型作为基础）
model, tokenizer = create_policy_model("./output/sft/final_model")
reference_model, _ = create_policy_model("./output/sft/final_model")  # 参考模型

# 配置训练参数
config = DPOTrainingConfig(
    model_name_or_path="./output/sft/final_model",
    output_dir="./output/dpo",
    num_epochs=3,
    learning_rate=5e-5,
    beta=0.1,
    loss_type="sigmoid"
)

# 创建训练器
trainer = DPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    reference_model=reference_model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader
)

# 开始训练
training_result = trainer.train()

# 保存模型
trainer.save_model("./output/dpo/final_model")
```

**GRPO训练**

```bash
# 使用main.py进行GRPO训练
python main.py --config config.yaml --mode train --algorithm grpo
```

GRPO需要奖励模型，确保先完成奖励模型训练。或者使用传统方式：

```python
from src.trainers import GRPOTrainer, GRPOConfig
from src.models import create_policy_model, create_reward_model

# 加载模型（使用SFT模型作为基础）
policy_model, tokenizer = create_policy_model("./output/sft/final_model")
reward_model, _ = create_reward_model("./output/reward/final_model")  # 加载训练好的奖励模型

# 配置训练参数
config = GRPOConfig(
    model_name_or_path="./output/sft/final_model",
    output_dir="./output/grpo",
    num_epochs=3,
    learning_rate=1e-5,
    group_size=4,
    beta=0.1,
    clip_range=0.2,
    vf_coef=0.5,
    ent_coef=0.01
)

# 创建训练器
trainer = GRPOTrainer(
    config=config,
    policy_model=policy_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader
)

# 开始训练
training_result = trainer.train()

# 保存模型
trainer.save_model("./output/grpo/final_model")
```

### 完整训练流程示例

以下是完整的RLHF训练流程示例：

```python
# 完整的RLHF训练流程
def complete_rlhf_training():
    # 1. SFT训练
    print("Step 1: Training SFT model...")
    sft_model_path = train_sft_model("Qwen/Qwen2.5-3B-Instruct")
    
    # 2. 奖励模型训练
    print("Step 2: Training Reward model...")
    reward_model_path = train_reward_model(sft_model_path)
    
    # 3. 选择强化学习算法进行训练
    print("Step 3: RL training...")
    
    # 选项A: PPO训练
    ppo_model_path = train_ppo_model(sft_model_path, reward_model_path)
    
    # 选项B: DPO训练（不需要奖励模型）
    # dpo_model_path = train_dpo_model(sft_model_path)
    
    # 选项C: GRPO训练
    # grpo_model_path = train_grpo_model(sft_model_path, reward_model_path)
    
    print("Training complete!")

if __name__ == "__main__":
    complete_rlhf_training()
```

### 快速开始训练

使用示例脚本进行完整的RLHF训练：

```bash
# 完整RLHF训练流程（SFT → Reward Model → PPO/DPO/GRPO）
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 100

# 仅进行SFT训练
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 100 --only_sft

# 跳过训练，仅评估现有模型
python example_training.py --skip_training

# 使用更大的模型
python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 500
```

### 命令行训练

```bash
# PPO训练（需要先完成SFT和Reward Model训练）
python main.py --config config.yaml --algorithm ppo

# DPO训练（需要先完成SFT训练）
python main.py --config config.yaml --algorithm dpo

# GRPO训练（需要先完成SFT和Reward Model训练）
python main.py --config config.yaml --algorithm grpo

# 指定GPU
python main.py --config config.yaml --algorithm ppo --device cuda:0

# 从检查点恢复
python main.py --config config.yaml --algorithm ppo --resume_from_checkpoint ./output/ppo/checkpoints/step-1000
```

> ⚠️ **训练顺序提醒**: 
> - PPO和GRPO训练前必须先完成SFT和Reward Model训练
> - DPO训练前只需完成SFT训练
> - 建议使用 `example_training.py` 进行完整流程训练

## 🧪 测试和评估

### 📊 训练进度跟踪

实时监控训练进度和预估完成时间：

```python
# scripts/training_monitor.py
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

class TrainingMonitor:
    def __init__(self, total_steps, log_file="training_progress.json"):
        self.total_steps = total_steps
        self.log_file = log_file
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, current_step, loss=None, metrics=None):
        """更新训练进度"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算平均步骤时间
        if current_step > 0:
            avg_step_time = elapsed_time / current_step
            remaining_steps = self.total_steps - current_step
            eta = remaining_steps * avg_step_time
            
            progress = {
                "current_step": current_step,
                "total_steps": self.total_steps,
                "progress_percent": (current_step / self.total_steps) * 100,
                "elapsed_time": elapsed_time,
                "eta_seconds": eta,
                "eta_formatted": str(timedelta(seconds=int(eta))),
                "avg_step_time": avg_step_time,
                "timestamp": datetime.now().isoformat(),
                "loss": loss,
                "metrics": metrics
            }
            
            # 保存进度
            with open(self.log_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            # 打印进度
            print(f"\r步骤 {current_step}/{self.total_steps} "
                  f"({progress['progress_percent']:.1f}%) "
                  f"- 预计剩余: {progress['eta_formatted']} "
                  f"- 损失: {loss:.4f if loss else 'N/A'}", end="")
            
            return progress
        
    def get_status(self):
        """获取当前状态"""
        if Path(self.log_file).exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return None

# 使用示例
if __name__ == "__main__":
    monitor = TrainingMonitor(total_steps=1000)
    
    # 模拟训练过程
    for step in range(1, 101):
        time.sleep(0.1)  # 模拟训练时间
        loss = 2.0 - (step * 0.01)  # 模拟损失下降
        monitor.update(step, loss=loss)
```

```bash
# 查看训练进度
python scripts/training_monitor.py

# 实时监控训练状态
watch -n 5 "python -c 'import json; print(json.dumps(json.load(open(\"training_progress.json\")), indent=2))'"
```

### 🔄 检查点管理和恢复

```python
# scripts/checkpoint_manager.py
import os
import json
from pathlib import Path
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint_info(self, step, loss, model_path, config):
        """保存检查点信息"""
        checkpoint_info = {
            "step": step,
            "loss": loss,
            "model_path": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "config": config
        }
        
        info_file = self.checkpoint_dir / f"checkpoint_{step}_info.json"
        with open(info_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
            
        # 更新最新检查点信息
        latest_file = self.checkpoint_dir / "latest_checkpoint.json"
        with open(latest_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
            
        print(f"✅ 检查点已保存: {info_file}")
        
    def list_checkpoints(self):
        """列出所有检查点"""
        checkpoints = []
        for info_file in self.checkpoint_dir.glob("checkpoint_*_info.json"):
            with open(info_file, 'r') as f:
                info = json.load(f)
                checkpoints.append(info)
        
        # 按步骤排序
        checkpoints.sort(key=lambda x: x['step'])
        return checkpoints
        
    def get_best_checkpoint(self, metric='loss', mode='min'):
        """获取最佳检查点"""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
            
        if mode == 'min':
            best = min(checkpoints, key=lambda x: x.get(metric, float('inf')))
        else:
            best = max(checkpoints, key=lambda x: x.get(metric, float('-inf')))
            
        return best
        
    def resume_from_checkpoint(self, checkpoint_path=None):
        """从检查点恢复训练"""
        if checkpoint_path is None:
            # 使用最新检查点
            latest_file = self.checkpoint_dir / "latest_checkpoint.json"
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    checkpoint_info = json.load(f)
                checkpoint_path = checkpoint_info['model_path']
            else:
                print("❌ 未找到检查点")
                return None
                
        if os.path.exists(checkpoint_path):
            print(f"✅ 从检查点恢复: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"❌ 检查点不存在: {checkpoint_path}")
            return None

# 使用示例
if __name__ == "__main__":
    manager = CheckpointManager()
    
    # 列出检查点
    checkpoints = manager.list_checkpoints()
    print("可用检查点:")
    for cp in checkpoints:
        print(f"  步骤 {cp['step']}: 损失 {cp['loss']:.4f} - {cp['timestamp']}")
    
    # 获取最佳检查点
    best = manager.get_best_checkpoint()
    if best:
        print(f"\n最佳检查点: 步骤 {best['step']}, 损失 {best['loss']:.4f}")
```

### 快速评估

```bash
# 评估训练好的模型
python main.py --config config.yaml --mode eval --algorithm ppo

# 评估DPO模型
python main.py --config config.yaml --mode eval --algorithm dpo

# 评估GRPO模型
python main.py --config config.yaml --mode eval --algorithm grpo

# 评估特定检查点
python main.py --config config.yaml --mode eval --resume_from_checkpoint ./output/ppo/checkpoint-1000

# 评估并生成报告
python main.py --config config.yaml --mode eval --algorithm dpo --output_dir ./eval_results
```

### 自动评估

```python
from src.evaluators import ModelEvaluator, AutomaticEvaluator
from transformers import AutoTokenizer

# 创建评估器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
evaluator = ModelEvaluator(tokenizer=tokenizer)

# 加载模型和数据
config = {
    'policy_model_path': './output/ppo/final_model',  # 或 './output/dpo/final_model', './output/grpo/final_model'
    'reward_model_path': './output/reward/final_model',
    'eval_dataset_path': './data/test.json',
    'batch_size': 8
}

# 运行评估
results = evaluator.evaluate(config)

# 查看结果
print(f"BLEU分数: {results['bleu_4']:.4f}")
print(f"ROUGE-L分数: {results['rouge_l']:.4f}")
print(f"平均奖励: {results['mean_reward']:.4f}")
print(f"困惑度: {results['perplexity']:.2f}")
```

### 生成质量评估

```python
from src.evaluators import AutomaticEvaluator
from src.models import create_policy_model

# 加载模型
policy_model, tokenizer = create_policy_model("./output/ppo/final_model")  # 或使用其他算法的模型
reward_model, _ = create_reward_model("./output/reward/final_model")

# 创建自动评估器
auto_evaluator = AutomaticEvaluator(
    tokenizer=tokenizer,
    output_dir="./eval_results"
)

# 测试提示
test_prompts = [
    "请解释什么是机器学习",
    "如何提高编程技能？",
    "描述一下深度学习的应用"
]

# 参考答案（可选）
reference_answers = [
    "机器学习是人工智能的一个分支...",
    "提高编程技能需要多练习...",
    "深度学习在图像识别、自然语言处理等领域有广泛应用..."
]

# 综合评估
results = auto_evaluator.evaluate_policy_model(
    policy_model=policy_model,
    reward_model=reward_model,
    prompts=test_prompts,
    references=reference_answers,
    generation_kwargs={
        'max_new_tokens': 128,
        'temperature': 0.7,
        'top_p': 0.9
    }
)

# 保存结果
result_file = auto_evaluator.save_results(results, "comprehensive_eval.json")
print(f"评估结果已保存到: {result_file}")
```

### 奖励模型评估

```python
from src.trainers import RewardModelTrainer
from src.evaluators import AutomaticEvaluator

# 加载奖励模型
reward_trainer = RewardModelTrainer.from_pretrained("./output/reward/final_model")

# 准备偏好数据
chosen_texts = ["这是一个很好的回答", "详细且准确的解释"]
rejected_texts = ["回答不够详细", "信息有误"]

# 评估奖励模型
auto_evaluator = AutomaticEvaluator(tokenizer=reward_trainer.tokenizer)
reward_metrics = auto_evaluator.evaluate_reward_model(
    reward_model=reward_trainer.model,
    chosen_texts=chosen_texts,
    rejected_texts=rejected_texts
)

print(f"偏好准确率: {reward_metrics['accuracy']:.4f}")
print(f"奖励差异: {reward_metrics['margin']:.4f}")
print(f"选择回答平均奖励: {reward_metrics['chosen_reward_mean']:.4f}")
print(f"拒绝回答平均奖励: {reward_metrics['rejected_reward_mean']:.4f}")
```

### 人工评估

```python
from src.evaluators import HumanEvaluator

# 创建人工评估器
human_evaluator = HumanEvaluator(output_dir="./human_eval")

# 准备对比数据
prompts = ["解释量子计算的原理", "如何学习Python编程？"]
model_a_responses = ["量子计算利用量子力学原理...", "学习Python可以从基础语法开始..."]
model_b_responses = ["量子计算是一种新型计算方式...", "Python是一门易学的编程语言..."]

# 创建对比评估任务
task_file = human_evaluator.create_evaluation_task(
    prompts=prompts,
    responses_a=model_a_responses,
    responses_b=model_b_responses,
    model_a_name="PPO模型",
    model_b_name="DPO模型",
    task_name="ppo_vs_dpo_comparison"
)

# 导出HTML评估界面
html_file = human_evaluator.export_evaluation_interface(task_file, "html")
print(f"人工评估界面: {html_file}")

# 分析评估结果（评估完成后）
# results = human_evaluator.analyze_evaluation_results(task_file)
# print(f"PPO胜率: {results['win_rates']['PPO模型']:.2%}")
```

### 批量测试脚本

```python
# test_models.py
import json
import argparse
from pathlib import Path
from src.evaluators import ModelEvaluator
from transformers import AutoTokenizer

def run_evaluation(model_path, test_data_path, output_dir):
    """运行模型评估"""
    
    # 加载测试数据
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 创建评估器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    evaluator = ModelEvaluator(tokenizer=tokenizer)
    
    # 配置评估
    config = {
        'policy_model_path': model_path,
        'eval_dataset_path': test_data_path,
        'batch_size': 8,
        'max_samples': len(test_data)
    }
    
    # 运行评估
    results = evaluator.evaluate(config)
    
    # 保存结果
    output_path = Path(output_dir) / f"eval_results_{Path(model_path).name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"评估完成，结果保存到: {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--test_data", required=True, help="测试数据路径")
    parser.add_argument("--output_dir", default="./eval_results", help="输出目录")
    
    args = parser.parse_args()
    
    results = run_evaluation(args.model_path, args.test_data, args.output_dir)
    
    # 打印关键指标
    print("\n=== 评估结果 ===")
    print(f"BLEU-4: {results.get('bleu_4', 0):.4f}")
    print(f"ROUGE-L: {results.get('rouge_l', 0):.4f}")
    print(f"平均奖励: {results.get('mean_reward', 0):.4f}")
    print(f"生成多样性: {results.get('distinct_2', 0):.4f}")
```

### 使用批量测试

```bash
# 测试PPO模型
python test_models.py --model_path ./output/ppo/final_model --test_data ./data/test.json

# 测试DPO模型
python test_models.py --model_path ./output/dpo/final_model --test_data ./data/test.json

# 对比多个模型
for model in ./output/*/final_model; do
    echo "Testing $model"
    python test_models.py --model_path "$model" --test_data ./data/test.json
done
```

## 📈 监控和可视化

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir ./output/logs

# 在浏览器中访问
# http://localhost:6006
```

### 训练指标

框架自动记录以下指标：

- **损失指标**: 总损失、策略损失、价值损失、熵损失
- **PPO指标**: 裁剪比例、KL散度、优势估计
- **DPO指标**: 偏好准确率、奖励差异、隐式奖励
- **生成指标**: 响应长度、生成质量
- **训练指标**: 学习率、梯度范数、训练速度

## 🔧 高级用法

### 自定义算法

```python
from src.algorithms import BaseRLAlgorithm

class CustomAlgorithm(BaseRLAlgorithm):
    def __init__(self, model, config):
        super().__init__(model, config)
        # 自定义初始化
    
    def compute_loss(self, batch):
        # 实现自定义损失计算
        pass
    
    def training_step(self, batch):
        # 实现自定义训练步骤
        pass
```

### 自定义数据集

```python
from src.data import RLDataset

class CustomDataset(RLDataset):
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__(tokenizer, max_length)
        # 加载自定义数据
    
    def __getitem__(self, idx):
        # 实现数据获取逻辑
        pass
```

### 自定义训练器

```python
from src.trainers import BaseTrainer

class CustomTrainer(BaseTrainer):
    def compute_loss(self, batch):
        # 实现自定义损失计算
        pass
    
    def evaluate(self):
        # 实现自定义评估逻辑
        pass
```

## 🐛 故障排除

### 🚨 智能错误处理和调试

```python
# scripts/error_handler.py
import traceback
import logging
import sys
from datetime import datetime
from pathlib import Path

class TrainingErrorHandler:
    def __init__(self, log_dir="./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置详细日志记录"""
        log_file = self.log_dir / f"training_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def handle_cuda_error(self, error):
        """处理CUDA相关错误"""
        error_msg = str(error).lower()
        
        if "out of memory" in error_msg:
            suggestions = [
                "🔧 减少批次大小 (batch_size)",
                "🔧 启用梯度累积 (gradient_accumulation_steps)",
                "🔧 使用混合精度训练 (fp16=True)",
                "🔧 减少序列长度 (max_length)",
                "🔧 使用CPU卸载 (offload_to_cpu=True)"
            ]
            
            self.logger.error(f"❌ CUDA内存不足: {error}")
            self.logger.info("💡 建议解决方案:")
            for suggestion in suggestions:
                self.logger.info(f"   {suggestion}")
                
            # 自动生成修复配置
            self.generate_memory_optimized_config()
            
        elif "device-side assert" in error_msg:
            self.logger.error("❌ CUDA设备断言错误，可能是数据问题")
            self.logger.info("💡 建议检查:")
            self.logger.info("   🔍 数据格式是否正确")
            self.logger.info("   🔍 标签范围是否合理")
            self.logger.info("   🔍 输入长度是否超限")
            
    def generate_memory_optimized_config(self):
        """生成内存优化配置"""
        optimized_config = {
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "fp16": True,
                "dataloader_num_workers": 0,
                "save_steps": 100,
                "eval_steps": 100
            },
            "model": {
                "gradient_checkpointing": True
            },
            "data": {
                "max_length": 512
            }
        }
        
        config_file = self.log_dir / "memory_optimized_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
            
        self.logger.info(f"✅ 已生成内存优化配置: {config_file}")

# 使用示例
if __name__ == "__main__":
    handler = TrainingErrorHandler()
    
    # 在训练代码中使用
    try:
        # 训练代码
        pass
    except RuntimeError as e:
        if "cuda" in str(e).lower():
            handler.handle_cuda_error(e)
        raise
```

### ⚡ 性能优化建议

```python
# scripts/performance_optimizer.py
import torch
import psutil

class PerformanceOptimizer:
    def __init__(self):
        self.gpu_memory = self.get_gpu_memory()
        self.cpu_count = psutil.cpu_count()
        self.system_memory = psutil.virtual_memory().total / 1024**3
        
    def get_gpu_memory(self):
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0
        
    def optimize_training_args(self, base_config):
        """根据硬件配置优化训练参数"""
        optimized = base_config.copy()
        
        # 根据GPU内存调整批次大小
        if self.gpu_memory > 0:
            if self.gpu_memory >= 24:  # 24GB+
                optimized['per_device_train_batch_size'] = 8
                optimized['gradient_accumulation_steps'] = 2
            elif self.gpu_memory >= 16:  # 16-24GB
                optimized['per_device_train_batch_size'] = 4
                optimized['gradient_accumulation_steps'] = 4
            elif self.gpu_memory >= 8:   # 8-16GB
                optimized['per_device_train_batch_size'] = 2
                optimized['gradient_accumulation_steps'] = 8
            else:  # <8GB
                optimized['per_device_train_batch_size'] = 1
                optimized['gradient_accumulation_steps'] = 16
                optimized['fp16'] = True
                optimized['gradient_checkpointing'] = True
        
        return optimized
        
    def print_optimization_report(self):
        """打印优化报告"""
        print("=== 性能优化报告 ===")
        print(f"🖥️  GPU内存: {self.gpu_memory:.1f}GB")
        print(f"💾 系统内存: {self.system_memory:.1f}GB")
        print(f"⚙️  CPU核心: {self.cpu_count}")
        
        if self.gpu_memory >= 24:
            print(f"📊 建议模型: 7B或更大模型")
        elif self.gpu_memory >= 16:
            print(f"📊 建议模型: 3B-7B模型")
        elif self.gpu_memory >= 8:
            print(f"📊 建议模型: 1B-3B模型")
        else:
            print(f"📊 建议模型: 小于1B模型或使用CPU")

# 使用示例
if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    optimizer.print_optimization_report()
```

```bash
# 运行性能优化分析
python scripts/performance_optimizer.py

# 使用优化后的配置训练
python scripts/error_handler.py --generate_config
python main.py --config logs/memory_optimized_config.yaml
```

### 常见问题

1. **内存不足**
   - 减少批大小 (`per_device_train_batch_size`)
   - 增加梯度累积步数 (`gradient_accumulation_steps`)
   - 使用混合精度训练 (`fp16: true`)
   - 使用自动优化: `python scripts/performance_optimizer.py`

2. **训练不收敛**
   - 调整学习率
   - 检查数据质量
   - 调整算法超参数
   - 使用错误处理器: `python scripts/error_handler.py`

3. **生成质量差**
   - 增加训练数据
   - 调整生成参数
   - 使用更好的基础模型

### 调试模式

```python
# 启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用小数据集测试
config.num_epochs = 1
config.max_train_samples = 100
config.max_eval_samples = 50
```

## 📚 API文档

### 核心类

- `BaseTrainer`: 基础训练器类
- `PPOTrainer`: PPO算法训练器
- `DPOTrainer`: DPO算法训练器
- `PolicyModel`: 策略模型
- `ValueModel`: 价值模型
- `RewardModel`: 奖励模型

### 工具函数

- `setup_logger()`: 设置日志记录
- `load_config()`: 加载配置文件
- `compute_advantages()`: 计算优势函数
- `save_checkpoint()`: 保存检查点

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Transformers](https://github.com/huggingface/transformers) - 预训练模型库
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 可视化工具

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 加入讨论群

---

**Happy Training! 🚀**
