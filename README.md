# 大模型强化学习框架 (RL Learning Framework)

一个专为大语言模型设计的强化学习训练框架，支持PPO、DPO、GRPO等主流算法，适用于RLHF（人类反馈强化学习）场景。

## 🚀 特性

- **多算法支持**: PPO、DPO、GRPO等强化学习算法
- **模块化设计**: 灵活的组件架构，易于扩展和定制
- **完整训练流程**: 从数据处理到模型训练的端到端解决方案
- **多种模型**: 策略模型、价值模型、奖励模型
- **丰富的数据处理**: 支持SFT、偏好学习、对话等多种数据格式
- **实时监控**: TensorBoard集成，实时训练监控
- **检查点管理**: 自动保存和恢复训练状态

## 📁 项目结构

```
rl_learning/
├── config.yaml              # 主配置文件
├── main.py                   # 主程序入口
├── requirements.txt          # 依赖包列表
├── README.md                # 项目文档
└── src/
    ├── algorithms/          # 强化学习算法
    │   ├── base.py         # 基础算法类
    │   ├── ppo.py          # PPO算法
    │   ├── dpo.py          # DPO算法
    │   ├── grpo.py         # GRPO算法
    │   └── utils.py        # 算法工具函数
    ├── data/               # 数据处理模块
    │   ├── processor.py    # 数据处理器
    │   ├── dataset.py      # 数据集类
    │   ├── collator.py     # 数据整理器
    │   └── utils.py        # 数据工具函数
    ├── models/             # 模型定义
    │   ├── policy_model.py # 策略模型
    │   ├── value_model.py  # 价值模型
    │   ├── reward_model.py # 奖励模型
    │   └── model_utils.py  # 模型工具函数
    ├── trainers/           # 训练器
    │   ├── base_trainer.py # 基础训练器
    │   ├── ppo_trainer.py  # PPO训练器
    │   └── dpo_trainer.py  # DPO训练器
    └── utils/              # 工具模块
        ├── logger.py       # 日志记录
        ├── config.py       # 配置管理
        ├── metrics.py      # 评估指标
        └── checkpoint.py   # 检查点管理
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
pip install -r requirements.txt
```

## ⚙️ 配置

### 主配置文件 (config.yaml)

```yaml
# RL Learning Framework Configuration

# 模型配置
model:
  name_or_path: "Qwen/Qwen2.5-3B"           # 基础模型路径
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
  max_length: 512                           # 最大序列长度
  max_prompt_length: 256                    # 最大提示长度

# 生成配置
generation:
  max_new_tokens: 256                       # 最大生成长度
  temperature: 1.0                          # 采样温度
  top_k: 50                                 # Top-k采样
  top_p: 1.0                                # Top-p采样

# 日志配置
logging:
  level: "INFO"                             # 日志级别
  log_dir: "./logs"                         # 日志目录
  tensorboard: true                         # 启用TensorBoard
```

## 📊 数据准备

### 快速开始 - 使用公开数据集

我们提供了自动化的数据准备脚本，支持多种公开数据集：

```bash
# 下载所有支持的数据集（推荐用于Qwen2.5 3B）
python prepare_datasets.py --datasets all --output_dir ./data

# 只下载中文数据集
python prepare_datasets.py --datasets belle alpaca-chinese --max_samples 10000

# 只下载英文偏好数据
python prepare_datasets.py --datasets hh-rlhf --max_samples 5000
```

**支持的数据集**：
- **HH-RLHF**: Anthropic的英文偏好数据集（16万条）
- **BELLE**: 中文指令数据集（200万条）
- **Alpaca Chinese**: 中文版Alpaca数据集（5万条）

> 📖 **详细指南**: 查看 [数据准备指南](DATA_PREPARATION_GUIDE.md) 了解完整的数据准备流程

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

### 数据预处理

```python
from src.data import DataProcessor

# 创建数据处理器
processor = DataProcessor(
    tokenizer_name="Qwen/Qwen2.5-3B",
    max_length=512
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

### PPO训练

```python
from src.trainers import PPOTrainer, PPOTrainingConfig
from src.models import create_policy_model, create_value_model
from transformers import AutoTokenizer

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
policy_model, _ = create_policy_model("Qwen/Qwen2.5-3B")
value_model, _ = create_value_model("Qwen/Qwen2.5-3B")

# 配置训练参数
config = PPOTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-3B",
    output_dir="./output/ppo",
    num_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    ppo_epochs=4,
    clip_range=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01
)

# 创建训练器
trainer = PPOTrainer(
    config=config,
    policy_model=policy_model,
    value_model=value_model,
    tokenizer=tokenizer,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader
)

# 开始训练
training_result = trainer.train()

# 保存模型
trainer.save_model("./output/ppo/final_model")
```

### DPO训练

```python
from src.trainers import DPOTrainer, DPOTrainingConfig
from src.models import create_policy_model

# 加载模型
model, tokenizer = create_policy_model("Qwen/Qwen2.5-3B")
reference_model, _ = create_policy_model("Qwen/Qwen2.5-3B")

# 配置训练参数
config = DPOTrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-3B",
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
```

### 命令行训练

```bash
# PPO训练
python main.py --config config.yaml --algorithm ppo

# DPO训练
python main.py --config config.yaml --algorithm dpo

# 指定GPU
python main.py --config config.yaml --algorithm ppo --device cuda:0

# 从检查点恢复
python main.py --config config.yaml --algorithm ppo --resume_from_checkpoint ./output/ppo/checkpoints/step-1000
```

## 🧪 测试和评估

### 模型评估

```python
from src.trainers import PPOTrainer
from src.utils import compute_training_metrics

# 加载训练好的模型
trainer = PPOTrainer.from_pretrained("./output/ppo/final_model")

# 评估模型
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 生成响应测试
prompts = [
    "请介绍一下人工智能的发展历史",
    "如何学习机器学习？"
]

responses = trainer.generate_responses(
    prompts=prompts,
    max_new_tokens=256,
    temperature=0.7
)

for prompt, response in zip(prompts, responses):
    print(f"问题: {prompt}")
    print(f"回答: {response}")
    print("-" * 50)
```

### 偏好评估

```python
# DPO模型偏好评估
dpo_trainer = DPOTrainer.from_pretrained("./output/dpo/final_model")

# 计算偏好指标
preference_metrics = dpo_trainer.compute_preference_metrics(
    prompts=test_prompts,
    chosen_responses=chosen_responses,
    rejected_responses=rejected_responses
)

print(f"偏好准确率: {preference_metrics['preference_accuracy']:.4f}")
print(f"奖励差异: {preference_metrics['reward_diff']:.4f}")
```

### 批量测试

```python
import json
from src.utils import MetricsTracker

# 加载测试数据
with open("./data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 批量生成和评估
metrics_tracker = MetricsTracker()

for item in test_data:
    prompt = item["prompt"]
    expected = item["expected"]
    
    # 生成回答
    generated = trainer.generate_responses([prompt])[0]
    
    # 计算指标（这里需要根据具体任务定义评估指标）
    metrics = {
        "length": len(generated.split()),
        "prompt_length": len(prompt.split())
    }
    
    metrics_tracker.update(metrics)

# 获取平均指标
avg_metrics = metrics_tracker.get_average_metrics()
print(f"平均生成长度: {avg_metrics['length']:.2f}")
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

### 常见问题

1. **内存不足**
   - 减少批大小 (`per_device_train_batch_size`)
   - 增加梯度累积步数 (`gradient_accumulation_steps`)
   - 使用混合精度训练 (`fp16: true`)

2. **训练不收敛**
   - 调整学习率
   - 检查数据质量
   - 调整算法超参数

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