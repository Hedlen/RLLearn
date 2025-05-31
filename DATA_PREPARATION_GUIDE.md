# 数据准备指南 - Qwen2.5 3B模型RLHF训练

本指南将帮助您为Qwen2.5 3B模型准备RLHF训练所需的数据集。我们提供了多种公开数据集的下载和处理方案。

## 📋 支持的数据集

### 1. 英文数据集

#### Anthropic HH-RLHF
- **描述**: Anthropic发布的人类偏好数据集，包含有用性和无害性两个维度
- **数据量**: 约16万条偏好对比数据
- **格式**: 偏好学习数据 (chosen vs rejected)
- **用途**: PPO/DPO训练的偏好数据
- **语言**: 英文

### 2. 中文数据集

#### BELLE
- **描述**: 中文指令微调数据集
- **数据量**: 200万条中文指令数据
- **格式**: SFT数据 (instruction + response)
- **用途**: 监督微调和偏好数据生成
- **语言**: 中文

#### Alpaca Chinese
- **描述**: Alpaca数据集的中文版本
- **数据量**: 约5万条中文指令数据
- **格式**: SFT数据 (instruction + input + output)
- **用途**: 监督微调
- **语言**: 中文

## 🚀 快速开始

### 安装依赖

```bash
pip install datasets transformers pandas tqdm
```

### 下载所有数据集

```bash
# 下载所有支持的数据集
python prepare_datasets.py --datasets all --output_dir ./data

# 限制每个数据集的样本数量（用于快速测试）
python prepare_datasets.py --datasets all --max_samples 1000 --output_dir ./data
```

### 下载特定数据集

```bash
# 只下载中文数据集
python prepare_datasets.py --datasets belle alpaca-chinese --output_dir ./data

# 只下载英文偏好数据
python prepare_datasets.py --datasets hh-rlhf --output_dir ./data
```

## 📊 数据集详细说明

### HH-RLHF数据集

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

**下载命令**:
```bash
python prepare_datasets.py --datasets hh-rlhf --max_samples 5000
```

### BELLE数据集

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

**下载命令**:
```bash
python prepare_datasets.py --datasets belle --max_samples 10000
```

### Alpaca Chinese数据集

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

**下载命令**:
```bash
python prepare_datasets.py --datasets alpaca-chinese --max_samples 20000
```

## 🔧 数据处理流程

### 1. 数据下载和预处理

```python
from prepare_datasets import DatasetPreparer

# 创建数据准备器
preparer = DatasetPreparer(
    output_dir="./data",
    tokenizer_name="Qwen/Qwen2.5-3B"
)

# 下载BELLE数据集
belle_file = preparer.download_belle_data(max_samples=5000)

# 验证数据质量
stats = preparer.validate_data(belle_file, "sft")
print(f"数据统计: {stats}")
```

### 2. 创建偏好数据

```python
# 从SFT数据创建合成偏好数据
preference_file = preparer.create_synthetic_preference_data(
    sft_file=belle_file,
    num_samples=1000
)
```

### 3. 数据验证

```python
# 验证偏好数据
stats = preparer.validate_data(preference_file, "preference")
print(f"偏好数据统计: {stats}")
```

## 📈 训练数据配置

### SFT训练配置

```yaml
# config_sft.yaml
model:
  name_or_path: "Qwen/Qwen2.5-3B"
  cache_dir: "./cache"

training:
  algorithm: "sft"
  output_dir: "./output/sft"
  num_epochs: 3
  learning_rate: 5e-5
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2

data:
  dataset_path: "./data/belle_sft_train.json"
  max_length: 512
  max_prompt_length: 256
```

### PPO训练配置

```yaml
# config_ppo.yaml
model:
  name_or_path: "./output/sft/final_model"  # 使用SFT训练后的模型
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

data:
  dataset_path: "./data/synthetic_preference_train.json"
  max_length: 512
```

### DPO训练配置

```yaml
# config_dpo.yaml
model:
  name_or_path: "./output/sft/final_model"
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
  max_length: 512
```

## 🎯 训练流程建议

### 阶段1: 监督微调 (SFT)

```bash
# 1. 准备SFT数据
python prepare_datasets.py --datasets belle alpaca-chinese --max_samples 10000

# 2. 开始SFT训练
python main.py --config config_sft.yaml
```

**目标**: 让模型学会基本的对话和指令跟随能力

### 阶段2: 偏好优化 (PPO/DPO)

```bash
# 1. 准备偏好数据
python prepare_datasets.py --datasets hh-rlhf --max_samples 5000

# 2. PPO训练
python main.py --config config_ppo.yaml

# 或者 DPO训练
python main.py --config config_dpo.yaml
```

**目标**: 让模型学会人类偏好，提高回答质量

## 📊 数据质量评估

### 自动评估指标

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

### 手动质量检查

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

## 🔍 故障排除

### 常见问题

1. **下载失败**
   ```bash
   # 设置代理（如果需要）
   export HF_ENDPOINT=https://hf-mirror.com
   python prepare_datasets.py --datasets belle
   ```

2. **内存不足**
   ```bash
   # 减少样本数量
   python prepare_datasets.py --datasets all --max_samples 1000
   ```

3. **分词器加载失败**
   ```bash
   # 使用本地分词器
   python prepare_datasets.py --tokenizer ./local_tokenizer
   ```

### 数据格式验证

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

## 📚 进阶用法

### 自定义数据集

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

### 数据混合策略

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

### 数据增强

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

## 🎉 总结

通过本指南，您可以：

1. **快速获取**多种高质量的公开数据集
2. **灵活配置**不同训练阶段的数据需求
3. **自动验证**数据质量和格式正确性
4. **轻松扩展**到自定义数据集

建议的训练流程：
1. 使用中文SFT数据（BELLE + Alpaca Chinese）进行监督微调
2. 使用偏好数据（HH-RLHF + 合成数据）进行PPO/DPO训练
3. 根据具体需求调整数据比例和训练参数

开始您的Qwen2.5 3B模型RLHF训练之旅吧！🚀