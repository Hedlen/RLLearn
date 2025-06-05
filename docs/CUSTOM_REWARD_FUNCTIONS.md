# 自定义奖励函数指南

本指南详细介绍如何在 GRPO (Generalized Reward Policy Optimization) 训练中使用自定义奖励函数来替代传统的奖励模型。

## 概述

传统的 GRPO 训练需要一个预训练的奖励模型来评估生成的响应质量。现在，您可以选择使用自定义奖励函数来替代奖励模型，这样可以：

- **降低计算成本**：无需训练和加载额外的奖励模型
- **提高灵活性**：可以根据特定任务需求定制奖励逻辑
- **简化流程**：直接通过规则或启发式方法计算奖励
- **快速迭代**：可以快速调整奖励策略而无需重新训练模型

## 配置方法

### 1. 基本配置

在 `config.yaml` 文件中添加 `reward_function` 配置：

```yaml
reward_function:
  module: "src.rewards.custom_rewards"  # Python 模块路径
  function: "length_based_reward"       # 函数名称
```

### 2. 优先级规则

系统按以下优先级选择奖励计算方式：
1. **自定义奖励函数** (如果配置了 `reward_function`)
2. **奖励模型** (如果没有配置奖励函数)
3. **虚拟奖励** (如果既没有奖励函数也没有奖励模型，返回 0.0)

## 内置奖励函数

我们提供了多种内置奖励函数，位于 `src/rewards/custom_rewards.py`：

### 1. 长度基础奖励 (`length_based_reward`)

根据响应长度给出奖励，适合控制生成文本的长度：

```python
def length_based_reward(prompt: str, response: str) -> float:
    """基于长度的奖励函数
    
    - 最优长度范围：50-200 词
    - 过短或过长都会被惩罚
    - 返回值：-1.0 到 1.0
    """
```

**适用场景**：
- 需要控制回答长度的对话系统
- 摘要生成任务
- 简洁回答要求的问答系统

### 2. 关键词基础奖励 (`keyword_based_reward`)

基于关键词出现情况给出奖励：

```python
def keyword_based_reward(prompt: str, response: str) -> float:
    """基于关键词的奖励函数
    
    - 奖励积极关键词：helpful, accurate, detailed 等
    - 惩罚消极关键词：harmful, offensive, inappropriate 等
    - 返回值：0.0 到 1.0
    """
```

**适用场景**：
- 安全性要求高的应用
- 特定领域的专业术语要求
- 情感倾向控制

### 3. 质量基础奖励 (`quality_based_reward`)

基于多个质量指标评估响应：

```python
def quality_based_reward(prompt: str, response: str) -> float:
    """基于质量的奖励函数
    
    评估指标：
    - 语法和标点符号
    - 连贯性和结构
    - 与提示的相关性
    - 返回值：0.0 到 1.0
    """
```

**适用场景**：
- 通用对话系统
- 内容生成任务
- 教育应用

### 4. 任务特定奖励 (`task_specific_reward`)

针对特定任务（如编程辅助）的奖励函数：

```python
def task_specific_reward(prompt: str, response: str) -> float:
    """任务特定奖励函数
    
    针对编程任务：
    - 奖励代码块和函数定义
    - 奖励解释和示例
    - 奖励适当的格式
    - 返回值：0.0 到 1.0
    """
```

**适用场景**：
- 代码生成和解释
- 技术文档编写
- 特定领域的专业应用

### 5. 正确性奖励 (`correctness_reward`)

基于答案正确性给出奖励，特别适合有标准答案的任务：

```python
def correctness_reward(prompt: str, response: str, expected_answer: str = None) -> float:
    """正确性奖励函数
    
    奖励策略：
    - 精确匹配：2.0 分
    - 数值等价：1.5 分
    - 部分匹配：1.0-1.8 分
    - 无匹配：0.0 分
    """
```

**适用场景**：
- 数学问题求解
- 编程题目评测
- 知识问答系统
- 标准化考试辅导

**特点**：
- 支持精确字符串匹配
- 支持数值等价检查（处理浮点精度）
- 支持数学表达式标准化
- 支持部分匹配和格式容错

### 6. 组合奖励 (`combined_reward`)

结合多种奖励标准：

```python
def combined_reward(prompt: str, response: str) -> float:
    """组合奖励函数
    
    权重分配：
    - 长度奖励：30%
    - 关键词奖励：30%
    - 质量奖励：40%
    - 返回值：0.0 到 1.0
    """
```

**适用场景**：
- 需要平衡多个因素的应用
- 通用性要求高的系统

### 7. 可配置奖励 (`ConfigurableReward`)

支持参数自定义的奖励类：

```python
class ConfigurableReward:
    def __init__(self, 
                 length_weight: float = 0.3,
                 keyword_weight: float = 0.3,
                 quality_weight: float = 0.4,
                 target_length_range: tuple = (50, 200),
                 positive_keywords: list = None,
                 negative_keywords: list = None):
```

## 自定义奖励函数

### 1. 函数签名要求

自定义奖励函数必须遵循以下签名：

```python
def your_reward_function(prompt: str, response: str) -> float:
    """自定义奖励函数
    
    Args:
        prompt: 输入提示
        response: 生成的响应
        
    Returns:
        奖励分数 (建议范围：0.0 到 1.0)
    """
    # 您的奖励逻辑
    return reward_score
```

### 2. 实现示例

```python
def sentiment_based_reward(prompt: str, response: str) -> float:
    """基于情感的奖励函数示例"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst']
    
    response_lower = response.lower()
    
    positive_count = sum(1 for word in positive_words if word in response_lower)
    negative_count = sum(1 for word in negative_words if word in response_lower)
    
    # 计算情感分数
    sentiment_score = (positive_count - negative_count) / max(1, len(response.split()))
    
    # 归一化到 0-1 范围
    return max(0.0, min(1.0, sentiment_score + 0.5))
```

### 3. 添加到项目

1. 将函数添加到 `src/rewards/custom_rewards.py`
2. 或创建新的模块文件
3. 在配置文件中指定模块和函数名

## 配置示例

### 示例 1：使用长度奖励 + LoRA

```yaml
model:
  model_name_or_path: "microsoft/DialoGPT-medium"
  use_peft: true
  peft_config:
    task_type: "CAUSAL_LM"
    r: 16
    lora_alpha: 32
    target_modules: ["c_attn", "c_proj"]

reward_function:
  module: "src.rewards.custom_rewards"
  function: "length_based_reward"

training:
  algorithm: "grpo"
  batch_size: 4
  learning_rate: 1e-5
```

### 示例 2：使用正确性奖励（适合数学/编程任务）

```yaml
model:
  model_name_or_path: "microsoft/DialoGPT-medium"
  use_peft: true
  peft_config:
    task_type: "CAUSAL_LM"
    r: 16
    lora_alpha: 32
    target_modules: ["c_attn", "c_proj"]

reward_function:
  module: "src.rewards.custom_rewards"
  function: "correctness_reward"
  # 注意：expected_answer 参数需要在数据集中提供或通过其他方式传递

training:
  algorithm: "grpo"
  batch_size: 4
  learning_rate: 1e-5
```

### 示例 3：使用质量奖励 + QLoRA

```yaml
model:
  model_name_or_path: "microsoft/DialoGPT-medium"
  use_peft: true
  peft_config:
    task_type: "CAUSAL_LM"
    r: 8
    lora_alpha: 16
    target_modules: ["c_attn", "c_proj"]
  quantization_config:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"

reward_function:
  module: "src.rewards.custom_rewards"
  function: "quality_based_reward"

training:
  algorithm: "grpo"
  batch_size: 2
  learning_rate: 1e-4
```

## 运行训练

使用自定义奖励函数运行 GRPO 训练：

```bash
# 使用长度奖励函数
python main.py --mode train --algorithm grpo --config config_length_reward.yaml

# 使用正确性奖励函数（适合数学/编程任务）
python main.py --mode train --algorithm grpo --config config_correctness_reward.yaml

# 使用关键词奖励函数
python main.py --mode train --algorithm grpo --config config_keyword_reward.yaml

# 使用组合奖励函数
python main.py --mode train --algorithm grpo --config config_combined_reward.yaml
```

## 性能对比

| 方法 | GPU 内存 | 训练速度 | 灵活性 | 准确性 |
|------|----------|----------|--------|--------|
| 奖励模型 | 高 | 慢 | 低 | 高 |
| 自定义函数 | 低 | 快 | 高 | 中等 |
| 组合方式 | 中等 | 中等 | 高 | 高 |

## 最佳实践

### 1. 奖励函数设计原则

- **明确目标**：确定您希望模型学习的具体行为
- **平衡性**：避免过度奖励某一方面而忽略其他
- **稳定性**：确保奖励函数在不同输入下表现稳定
- **可解释性**：奖励逻辑应该清晰易懂

### 2. 调试和优化

```python
def debug_reward_function(prompt: str, response: str) -> float:
    """带调试信息的奖励函数"""
    # 计算各项分数
    length_score = calculate_length_score(response)
    quality_score = calculate_quality_score(response)
    
    # 打印调试信息
    print(f"Length score: {length_score:.3f}")
    print(f"Quality score: {quality_score:.3f}")
    
    final_score = (length_score + quality_score) / 2
    print(f"Final score: {final_score:.3f}")
    
    return final_score
```

### 3. 错误处理

```python
def robust_reward_function(prompt: str, response: str) -> float:
    """健壮的奖励函数"""
    try:
        # 输入验证
        if not response or not response.strip():
            return 0.0
        
        # 奖励计算逻辑
        score = calculate_reward(prompt, response)
        
        # 确保返回值在合理范围内
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"Reward function error: {e}")
        return 0.0  # 默认奖励
```

### 4. 性能优化

- **缓存计算**：对于重复的计算结果进行缓存
- **批量处理**：如果可能，支持批量计算奖励
- **避免重复工作**：预计算常用的模式和规则

## 故障排除

### 常见问题

1. **模块导入错误**
   ```
   ModuleNotFoundError: No module named 'src.rewards.custom_rewards'
   ```
   **解决方案**：确保模块路径正确，检查 Python 路径设置

2. **函数不存在错误**
   ```
   AttributeError: module has no attribute 'your_function'
   ```
   **解决方案**：检查函数名拼写，确保函数已正确定义

3. **奖励值异常**
   ```
   Warning: Invalid reward type, using 0.0
   ```
   **解决方案**：确保奖励函数返回数值类型（int、float 或 tensor）

### 调试技巧

1. **启用详细日志**：在配置中设置 `logging_level: DEBUG`
2. **测试奖励函数**：在训练前单独测试奖励函数
3. **监控奖励分布**：观察训练过程中的奖励分布是否合理

## 总结

自定义奖励函数为 GRPO 训练提供了更大的灵活性和控制力。通过合理设计奖励函数，您可以：

- 降低计算成本和复杂性
- 快速适应不同的任务需求
- 实现更精确的行为控制
- 简化训练流程

选择使用奖励模型还是自定义奖励函数取决于您的具体需求：
- **奖励模型**：适合需要高精度评估的复杂任务
- **自定义函数**：适合规则明确、快速迭代的场景
- **组合使用**：在不同阶段使用不同方法

更多示例和配置请参考 `config_reward_function_examples.yaml` 文件。