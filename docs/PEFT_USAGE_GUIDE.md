# PEFT模型使用完整指南

本指南详细说明如何在本项目中使用PEFT（Parameter-Efficient Fine-Tuning）技术，包括LoRA和QLoRA的配置、训练、保存和合并。

## 📋 目录

1. [PEFT简介](#peft简介)
2. [配置PEFT](#配置peft)
3. [训练PEFT模型](#训练peft模型)
4. [保存PEFT适配器](#保存peft适配器)
5. [合并PEFT模型](#合并peft模型)
6. [使用合并后的模型](#使用合并后的模型)
7. [常见问题](#常见问题)
8. [最佳实践](#最佳实践)

## 🎯 PEFT简介

### 什么是PEFT？

PEFT（Parameter-Efficient Fine-Tuning）是一种参数高效的微调技术，只训练模型的一小部分参数，而不是整个模型。

### 支持的PEFT方法

| 方法 | 显存需求 | 训练速度 | 效果 | 适用场景 |
|------|----------|----------|------|----------|
| **LoRA** | 中等 (8-16GB) | 快 | 优秀 | 平衡性能和资源 |
| **QLoRA** | 低 (4-8GB) | 中等 | 良好 | 显存受限环境 |
| **全量微调** | 高 (16-32GB) | 慢 | 最佳 | 资源充足时 |

## ⚙️ 配置PEFT

### 1. LoRA配置

在 `config.yaml` 中启用LoRA：

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  use_peft: true
  peft_config:
    r: 16                    # LoRA rank，控制适配器大小
    lora_alpha: 32          # LoRA缩放参数，通常是r的2倍
    lora_dropout: 0.1       # Dropout率
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]  # 目标模块
    bias: "none"            # 偏置处理: "none", "all", "lora_only"
  # 不设置quantization_config
```

### 2. QLoRA配置

在 `config.yaml` 中启用QLoRA（4bit量化 + LoRA）：

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  use_peft: true
  peft_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    bias: "none"
  
  # 添加量化配置启用QLoRA
  quantization_config:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
```

### 3. 全量微调配置

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  use_peft: false  # 关闭PEFT
  # 不需要peft_config和quantization_config
```

### 参数说明

#### LoRA参数

- **r (rank)**: LoRA适配器的秩
  - 值越大，适配器参数越多，效果可能更好但显存占用更高
  - 推荐值: 8, 16, 32, 64
  - 一般从16开始尝试

- **lora_alpha**: LoRA缩放参数
  - 控制LoRA适配器的学习率缩放
  - 通常设为r的2倍
  - 如果r=16，则lora_alpha=32

- **lora_dropout**: Dropout率
  - 防止过拟合
  - 推荐值: 0.05-0.1

- **target_modules**: 应用LoRA的模块
  - 注意力层: `["q_proj", "k_proj", "v_proj", "o_proj"]`
  - 所有线性层: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

## 🚀 训练PEFT模型

### 1. SFT训练（LoRA）

```bash
# 使用LoRA进行SFT训练
python main.py --config config.yaml --mode train --algorithm sft --experiment_name "sft_qwen_lora_r16"
```

### 2. Reward模型训练（QLoRA）

```bash
# 使用QLoRA训练Reward模型
python main.py --config config_qlora.yaml --mode train --algorithm reward --experiment_name "reward_qwen_qlora"
```

### 3. RLHF训练（LoRA）

```bash
# 使用LoRA进行PPO训练
python main.py --config config.yaml --mode train --algorithm ppo --experiment_name "ppo_qwen_lora"
```

## 💾 保存PEFT适配器

### 训练过程中的自动保存

训练过程中，PEFT适配器会自动保存到输出目录：

```
outputs/
└── sft_qwen_lora_r16/
    ├── checkpoint-500/
    │   ├── adapter_config.json    # 适配器配置
    │   ├── adapter_model.bin      # 适配器权重
    │   └── ...
    ├── checkpoint-1000/
    └── final_model/
```

### 手动保存适配器

使用我们提供的脚本：

```bash
# 保存PEFT适配器
python scripts/peft_model_manager.py --action save --model_path ./outputs/sft_lora/final_model --output_path ./saved_adapter
```

## 🔄 合并PEFT模型

### 为什么需要合并？

- **PEFT适配器**: 只包含少量参数，需要与基础模型一起使用
- **合并后的模型**: 包含完整参数，可以独立使用，便于部署

### 使用简化脚本合并

我们提供了简化的合并脚本 `merge_peft_model.py`：

```bash
# 1. 检查适配器信息
python merge_peft_model.py --check_adapter ./outputs/sft_lora/checkpoint-1000

# 2. 合并LoRA适配器
python merge_peft_model.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path ./outputs/sft_lora/checkpoint-1000 \
    --output_path ./merged_model

# 3. 验证合并后的模型
python merge_peft_model.py --validate ./merged_model
```

### 使用高级脚本合并

```bash
# 使用高级脚本进行合并
python scripts/peft_model_manager.py \
    --action merge \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path ./outputs/sft_lora/checkpoint-1000 \
    --output_path ./merged_model \
    --torch_dtype float16
```

### 合并过程说明

1. **加载基础模型**: 从HuggingFace或本地路径加载原始模型
2. **加载适配器**: 加载训练好的LoRA/QLoRA适配器
3. **合并权重**: 将适配器权重合并到基础模型中
4. **保存完整模型**: 保存为标准的HuggingFace模型格式

## 🎮 使用合并后的模型

### 1. 在代码中使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载合并后的模型
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# 生成文本
input_text = "你好，请介绍一下自己。"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 2. 使用vLLM部署

```bash
# 安装vLLM
pip install vllm

# 启动API服务
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model \
    --served-model-name my-model \
    --host 0.0.0.0 \
    --port 8000
```

### 3. 上传到HuggingFace Hub

```python
from huggingface_hub import HfApi

# 上传模型
api = HfApi()
api.upload_folder(
    folder_path="./merged_model",
    repo_id="your-username/your-model-name",
    repo_type="model"
)
```

## ❓ 常见问题

### Q1: LoRA和QLoRA的区别？

**A**: 
- **LoRA**: 只使用低秩适配器，不进行量化
- **QLoRA**: LoRA + 4bit量化，显存需求更低

### Q2: 如何选择LoRA的rank值？

**A**: 
- 小模型(7B以下): r=8-16
- 中等模型(7B-13B): r=16-32
- 大模型(13B以上): r=32-64
- 从16开始尝试，根据效果调整

### Q3: 训练后只有适配器文件，如何使用？

**A**: 有两种方式：
1. **直接使用**: 加载基础模型 + 适配器
2. **合并后使用**: 使用本指南的合并脚本

### Q4: 合并后的模型大小？

**A**: 合并后的模型大小与原始基础模型相同，因为LoRA参数被合并到原始权重中。

### Q5: 可以在不同的基础模型之间转移适配器吗？

**A**: 不建议。适配器是针对特定基础模型训练的，在不同模型间转移可能效果不佳。

### Q6: 如何验证PEFT训练是否成功？

**A**: 
1. 检查训练日志中的LoRA信息
2. 确认输出目录中有 `adapter_config.json` 和 `adapter_model.bin`
3. 使用验证脚本测试模型

## 🏆 最佳实践

### 1. 选择合适的PEFT方法

- **显存充足(16GB+)**: 优先选择LoRA
- **显存受限(8GB-)**: 选择QLoRA
- **追求最佳效果**: 全量微调（如果资源允许）

### 2. 参数调优建议

```yaml
# 保守配置（稳定但可能效果一般）
peft_config:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1

# 平衡配置（推荐）
peft_config:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1

# 激进配置（效果可能更好但需要更多资源）
peft_config:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.05
```

### 3. 目标模块选择

```yaml
# 最小配置（只对注意力层应用LoRA）
target_modules: ["q_proj", "v_proj"]

# 标准配置（推荐）
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# 完整配置（包含FFN层）
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 4. 实验管理

```bash
# 使用有意义的实验名称
python main.py --experiment_name "sft_qwen7b_lora_r16_alpha32_$(date +%Y%m%d)"

# 保存配置文件
cp config.yaml outputs/sft_qwen7b_lora_r16_alpha32_20241201/config_used.yaml
```

### 5. 模型版本管理

```
models/
├── base_models/           # 基础模型
│   └── Qwen2.5-7B-Instruct/
├── adapters/             # PEFT适配器
│   ├── sft_lora_v1/
│   ├── reward_qlora_v1/
│   └── ppo_lora_v1/
└── merged_models/        # 合并后的模型
    ├── sft_merged_v1/
    └── ppo_merged_v1/
```

### 6. 性能监控

训练时关注以下指标：
- **可训练参数比例**: 应该远小于100%
- **显存使用**: QLoRA < LoRA < 全量微调
- **训练速度**: LoRA > QLoRA > 全量微调
- **收敛情况**: 损失是否正常下降

## 📚 相关资源

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [PEFT库文档](https://huggingface.co/docs/peft)
- [Transformers文档](https://huggingface.co/docs/transformers)

## 🔧 故障排除

### 常见错误及解决方案

1. **ImportError: No module named 'peft'**
   ```bash
   pip install peft>=0.4.0
   ```

2. **CUDA out of memory**
   - 减小batch_size
   - 使用QLoRA而不是LoRA
   - 减小LoRA的rank值

3. **适配器加载失败**
   - 检查adapter_config.json是否存在
   - 确认基础模型路径正确
   - 验证适配器与基础模型的兼容性

4. **合并后模型效果差**
   - 检查训练是否充分收敛
   - 验证适配器配置是否合理
   - 尝试调整LoRA参数

---

如果您在使用过程中遇到问题，请查看项目的其他文档或提交Issue。