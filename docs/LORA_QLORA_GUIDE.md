# LoRA/QLoRA 切换指南

本指南详细说明如何在 **LoRA** 和 **QLoRA** 之间切换，以及如何为不同的模型组件（Policy Model、Reward Model）配置PEFT。

## 🔄 LoRA 和 QLoRA 切换方法

### 方法1: 修改 config.yaml

#### 切换到 LoRA (推荐)
```yaml
model:
  use_peft: true
  peft_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias: "none"
  # 注释掉或删除 quantization_config 部分
  # quantization_config:
  #   load_in_4bit: true
  #   ...
```

#### 切换到 QLoRA (最节省显存)
```yaml
model:
  use_peft: true
  peft_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias: "none"
  quantization_config:  # 添加这个部分启用QLoRA
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_use_double_quant: true
```

#### 切换到全量微调
```yaml
model:
  use_peft: false  # 关闭PEFT
  # 删除 peft_config 和 quantization_config 部分
```

### 方法2: 使用不同的配置文件

创建多个配置文件，快速切换：

```bash
# LoRA配置
cp config.yaml config_lora.yaml
# 编辑 config_lora.yaml，移除 quantization_config

# QLoRA配置
cp config.yaml config_qlora.yaml
# 编辑 config_qlora.yaml，保留 quantization_config

# 全量微调配置
cp config.yaml config_full.yaml
# 编辑 config_full.yaml，设置 use_peft: false
```

使用时指定不同配置：
```bash
# 使用LoRA
python main.py --algorithm sft --config config_lora.yaml

# 使用QLoRA
python main.py --algorithm sft --config config_qlora.yaml

# 使用全量微调
python main.py --algorithm sft --config config_full.yaml
```

## 🎯 支持 PEFT 的模型组件

### ✅ Policy Model (策略模型)
- **支持**: LoRA, QLoRA, 全量微调
- **用途**: SFT, DPO, PPO, GRPO 训练
- **配置**: 通过 `config.yaml` 中的 `model` 部分

### ✅ Reward Model (奖励模型) 
- **支持**: LoRA, QLoRA, 全量微调
- **用途**: Reward Model 训练, PPO/GRPO 中的奖励评估
- **配置**: 使用相同的 `model` 配置
- **特点**: 使用 `task_type="FEATURE_EXTRACTION"` 的LoRA配置

### ❌ Value Model (价值模型)
- **当前状态**: 暂不支持PEFT
- **计划**: 后续版本将添加支持

## 📊 不同模式的资源消耗对比

| 模式 | Policy Model 显存 | Reward Model 显存 | 总显存需求 | 训练速度 |
|------|------------------|------------------|------------|----------|
| 全量微调 | 高 (>16GB) | 高 (>8GB) | 很高 (>24GB) | 慢 |
| LoRA | 中 (8-12GB) | 中 (4-6GB) | 中等 (12-18GB) | 快 |
| QLoRA | 低 (4-6GB) | 低 (2-3GB) | 低 (6-9GB) | 中等 |

## ⚙️ 高级配置选项

### 针对不同模型的 target_modules

```yaml
# Qwen2.5 系列 (推荐)
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# LLaMA 系列
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# 只对注意力层应用LoRA (更少参数)
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# 对所有线性层应用LoRA (更多参数)
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
```

### 不同任务的推荐配置

#### 快速实验 (最少资源)
```yaml
peft_config:
  r: 8                    # 小rank
  lora_alpha: 16         # 对应调整
  target_modules: ["q_proj", "v_proj"]  # 只对关键层应用
quantization_config:     # 启用QLoRA
  load_in_4bit: true
```

#### 平衡配置 (推荐)
```yaml
peft_config:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
# 不使用量化，使用常规LoRA
```

#### 高质量训练 (更多资源)
```yaml
peft_config:
  r: 64                   # 大rank
  lora_alpha: 128        # 对应调整
  lora_dropout: 0.05     # 更小dropout
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## 🔧 实际使用示例

### 示例1: SFT训练使用QLoRA
```bash
# 1. 修改config.yaml启用QLoRA
# 2. 运行SFT训练
python main.py --algorithm sft --config config.yaml

# 训练输出会显示:
# QLoRA enabled for policy model: {'load_in_4bit': True, ...}
# LoRA applied to policy model:
#   - Rank (r): 16
#   - Alpha: 32
#   - Trainable parameters: 4,194,304 (0.25%)
```

### 示例2: Reward Model训练使用LoRA
```bash
# 1. 在config.yaml中设置use_peft: true，但不设置quantization_config
# 2. 运行Reward训练
python main.py --algorithm reward --config config.yaml

# 训练输出会显示:
# LoRA applied to reward model:
#   - Rank (r): 16
#   - Alpha: 32
#   - Target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
#   - Trainable parameters: 2,097,152 (0.15%)
```

### 示例3: PPO训练同时使用PEFT
```bash
# PPO会同时创建Policy Model和Reward Model，都会应用相同的PEFT配置
python main.py --algorithm ppo --config config.yaml

# 两个模型都会显示LoRA信息
```

## 🚨 注意事项

### 1. 依赖要求
确保安装了必要依赖：
```bash
pip install peft>=0.4.0
pip install bitsandbytes>=0.41.0  # QLoRA需要
```

### 2. 显存不足时的解决方案
```yaml
# 方案1: 使用QLoRA
quantization_config:
  load_in_4bit: true

# 方案2: 减小LoRA rank
peft_config:
  r: 8  # 从16减小到8

# 方案3: 减少target_modules
peft_config:
  target_modules: ["q_proj", "v_proj"]  # 只对关键层应用

# 方案4: 减小batch_size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

### 3. 模型保存和加载
- LoRA适配器会自动保存到输出目录
- 可以单独保存适配器权重，节省存储空间
- 推理时需要加载基础模型 + LoRA适配器

### 4. 性能优化建议
- QLoRA适合显存受限的环境
- LoRA适合平衡性能和资源的场景
- 全量微调适合追求最佳效果的生产环境
- 可以先用QLoRA快速实验，再用LoRA精调

## 📚 相关资源

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [PEFT库文档](https://huggingface.co/docs/peft)
- [BitsAndBytes文档](https://github.com/TimDettmers/bitsandbytes)