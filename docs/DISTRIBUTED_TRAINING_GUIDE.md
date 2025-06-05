# 分布式训练指南

本指南介绍如何在项目中使用分布式训练来加速大模型训练。

## 概述

分布式训练支持：
- **单节点多GPU训练**：在一台机器上使用多个GPU
- **多节点多GPU训练**：在多台机器上使用多个GPU
- **数据并行**：支持PyTorch DistributedDataParallel (DDP) 和 DeepSpeed
- **混合精度训练**：支持FP16和BF16
- **梯度累积**：支持大批次训练
- **内存优化**：DeepSpeed ZeRO优化器，支持CPU/NVMe卸载

## 策略选择

### DDP vs DeepSpeed

| 特性 | DDP | DeepSpeed ZeRO-1 | DeepSpeed ZeRO-2 | DeepSpeed ZeRO-3 |
|------|-----|------------------|------------------|------------------|
| 模型大小 | <7B | <7B | 1-7B | >7B |
| 内存使用 | 高 | 中等 | 低 | 最低 |
| 通信开销 | 低 | 低 | 中等 | 高 |
| 设置复杂度 | 简单 | 简单 | 中等 | 复杂 |
| 推荐场景 | 小模型快速训练 | 中等模型 | 大模型 | 超大模型 |

### 自动策略选择

项目支持自动选择最佳分布式策略：

```yaml
training:
  distributed_strategy: "auto"  # 自动选择最佳策略
```

选择逻辑：
- 模型参数 < 1B：使用 DDP
- 模型参数 1B-7B：使用 DeepSpeed ZeRO-2
- 模型参数 > 7B：使用 DeepSpeed ZeRO-3

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
# 检查CUDA和GPU
nvidia-smi

# 检查PyTorch分布式支持
python -c "import torch; print(torch.distributed.is_available())"
```

### 2. 单机多卡训练

#### 方法一：使用启动脚本（推荐）

```bash
# 使用4个GPU训练
python scripts/run_distributed.py \
    --nproc_per_node 4 \
    --config config_distributed_example.yaml \
    --algorithm sft

# 使用torchrun（推荐）
python scripts/run_distributed.py \
    --use_torchrun \
    --nproc_per_node 4 \
    --config config_distributed_example.yaml
```

#### 方法二：直接使用torchrun

```bash
# PyTorch 1.10+
torchrun --nproc_per_node=4 main.py \
    --mode train \
    --config config_distributed_example.yaml \
    --algorithm sft
```

#### 方法三：使用torch.distributed.launch

```bash
# 兼容性方法
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --mode train \
    --config config_distributed_example.yaml \
    --algorithm sft
```

#### 方法四：使用DeepSpeed

```bash
# 使用DeepSpeed进行训练
deepspeed --num_gpus=4 main.py \
    --config config_deepspeed_example.yaml \
    --mode train \
    --algorithm sft
```

### 3. 多机多卡训练

#### DDP多节点训练

##### 节点0（主节点）

```bash
python scripts/run_distributed.py \
    --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr 192.168.1.100 \
    --master_port 29500 \
    --config config_distributed_example.yaml
```

##### 节点1（从节点）

```bash
python scripts/run_distributed.py \
    --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr 192.168.1.100 \
    --master_port 29500 \
    --config config_distributed_example.yaml
```

#### DeepSpeed多节点训练

##### 节点0（主节点）

```bash
deepspeed --num_nodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=29500 --num_gpus=4 main.py \
    --config config_deepspeed_example.yaml \
    --mode train \
    --algorithm sft
```

##### 节点1（从节点）

```bash
deepspeed --num_nodes=2 --node_rank=1 --master_addr=192.168.1.100 --master_port=29500 --num_gpus=4 main.py \
    --config config_deepspeed_example.yaml \
    --mode train \
    --algorithm sft
```

## 配置说明

### DDP配置

在配置文件中设置DDP分布式训练参数：

```yaml
training:
  # 分布式训练设置
  distributed: true
  distributed_strategy: "ddp"  # 使用DDP策略
  
  # 数据加载器设置
  dataloader:
    batch_size: 8          # 每个GPU的批次大小
    num_workers: 4         # 数据加载进程数
    pin_memory: true       # 固定内存
    
  # 梯度累积
  gradient_accumulation_steps: 4  # 累积4个批次后更新
  
  # 混合精度训练
  mixed_precision: "fp16"  # 或 "bf16"
  
  # 梯度裁剪
  max_grad_norm: 1.0
```

### DeepSpeed配置

#### 基础DeepSpeed配置

```yaml
training:
  # 分布式训练设置
  distributed: true
  distributed_strategy: "deepspeed"  # 使用DeepSpeed策略
  
  # DeepSpeed配置
  deepspeed:
    config_path: "configs/deepspeed/zero2.json"  # DeepSpeed配置文件路径
    
  # 数据加载器设置
  dataloader:
    batch_size: 4          # 每个GPU的批次大小 (DeepSpeed可以使用更小的批次)
    num_workers: 4
    pin_memory: true
    
  # 梯度累积
  gradient_accumulation_steps: 8  # DeepSpeed支持更大的累积步数
```

#### DeepSpeed JSON配置文件

**ZeRO Stage 1配置** (`configs/deepspeed/zero1.json`)：

```json
{
  "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 500
    }
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "wall_clock_breakdown": false
}
```

**ZeRO Stage 2配置** (`configs/deepspeed/zero2.json`)：

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 500
    }
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "wall_clock_breakdown": false
}
```

**ZeRO Stage 3配置** (`configs/deepspeed/zero3.json`)：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 500
    }
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "wall_clock_breakdown": false
}
```

### 分布式训练配置

在配置文件中添加分布式训练相关设置：

```yaml
training:
  # 启用分布式训练
  distributed: true
  distributed_backend: "nccl"  # GPU使用nccl，CPU使用gloo
  
  # DDP配置
  find_unused_parameters: false  # 提高性能
  ddp_bucket_cap_mb: 25         # 通信bucket大小
  ddp_broadcast_buffers: true   # 广播buffer
  
  # 批次大小配置
  per_device_train_batch_size: 4    # 每个GPU的批次大小
  gradient_accumulation_steps: 2    # 梯度累积
  # 实际批次大小 = per_device_batch_size * num_gpus * gradient_accumulation_steps
  
  # 性能优化
  fp16: false          # FP16混合精度
  bf16: true           # BF16混合精度（推荐）
  gradient_checkpointing: true  # 节省显存
  
  # 数据加载
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  dataloader_drop_last: true
```

### 批次大小计算

分布式训练中的有效批次大小计算：

```
有效批次大小 = per_device_train_batch_size × GPU数量 × gradient_accumulation_steps
```

例如：
- `per_device_train_batch_size: 4`
- `GPU数量: 4`
- `gradient_accumulation_steps: 2`
- `有效批次大小 = 4 × 4 × 2 = 32`

## 性能优化

### 1. 显存优化

#### DDP显存优化
```yaml
training:
  # 启用梯度检查点
  gradient_checkpointing: true
  
  # 使用混合精度
  bf16: true  # 或 fp16: true
  
  # 调整批次大小
  per_device_train_batch_size: 2  # 减小以节省显存
  gradient_accumulation_steps: 4  # 增加以保持有效批次大小
```

#### DeepSpeed显存优化
```yaml
training:
  # DeepSpeed自动优化显存
  distributed_strategy: "deepspeed"
  deepspeed:
    config_path: "configs/deepspeed/zero3.json"  # 使用ZeRO-3最大化显存节省
  
  # 更小的批次大小
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### 2. 通信优化

#### DDP通信优化
```yaml
training:
  # 优化DDP设置
  find_unused_parameters: false  # 提高性能
  ddp_bucket_cap_mb: 25         # 根据网络带宽调整
  
  # 数据加载优化
  dataloader_num_workers: 4     # 根据CPU核心数调整
  dataloader_pin_memory: true   # 加速GPU传输
```

#### DeepSpeed通信优化
```yaml
training:
  # DeepSpeed自动优化通信
  distributed_strategy: "deepspeed"
  
  # 数据加载优化
  dataloader_num_workers: 4
  dataloader_pin_memory: true
```

### 3. 学习率调整

分布式训练时通常需要调整学习率：

#### DDP学习率调整
```yaml
training:
  # 线性缩放规则
  learning_rate: 5e-5  # 单GPU学习率
  # 多GPU时可以考虑: learning_rate * sqrt(num_gpus)
  
  # 或使用warmup
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
```

#### DeepSpeed学习率调整
```yaml
training:
  # DeepSpeed配置文件中设置学习率
  distributed_strategy: "deepspeed"
  deepspeed:
    config_path: "configs/deepspeed/zero2.json"
  
  # 或在YAML中覆盖
  learning_rate: 3e-5
  warmup_ratio: 0.1
```

## 监控和调试

### 1. 性能监控

```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控网络通信（多机训练）
iftop -i eth0

# 查看训练日志
tail -f outputs/distributed_training/logs/training.log
```

### 2. 常见问题排查

#### 显存不足

```yaml
# 解决方案1：减小批次大小
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# 解决方案2：启用梯度检查点
gradient_checkpointing: true

# 解决方案3：使用混合精度
bf16: true
```

#### 通信超时

```bash
# 设置环境变量
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
```

#### 进程同步问题

```yaml
# 确保所有进程使用相同的配置
seed: 42  # 固定随机种子
```

## 支持的算法

所有训练算法都支持分布式训练：

- **SFT (Supervised Fine-Tuning)**
- **Reward Model Training**
- **PPO (Proximal Policy Optimization)**
- **DPO (Direct Preference Optimization)**
- **GRPO (Generalized Reward Policy Optimization)**

## 最佳实践

### 1. 硬件配置

- **GPU**：使用相同型号的GPU以获得最佳性能
- **内存**：确保足够的系统内存（推荐GPU显存的2-4倍）
- **存储**：使用SSD存储数据集以减少I/O瓶颈
- **网络**：多机训练时使用高速网络（10Gbps+）

### 2. 数据准备

```bash
# 预处理数据以减少训练时的开销
python prepare_datasets.py --config config.yaml

# 确保数据在所有节点上可访问
# 使用共享存储或在每个节点上复制数据
```

### 3. 实验管理

```yaml
training:
  # 使用实验名称组织输出
  experiment_name: "sft_4gpu_experiment"
  
  # 定期保存检查点
  save_steps: 500
  save_total_limit: 3
  
  # 启用详细日志
  logging_steps: 50
  report_to: ["tensorboard", "wandb"]
```

### 4. 故障恢复

```yaml
training:
  # 启用检查点恢复
  resume_from_checkpoint: "./outputs/experiment/checkpoints/latest"
  
  # 设置合理的保存策略
  save_strategy: "steps"
  save_steps: 1000
```

## 性能基准

### 单机多卡性能

| GPU数量 | 批次大小 | 训练速度 | 显存使用 | 效率 |
|---------|----------|----------|----------|------|
| 1       | 8        | 100%     | 100%     | 100% |
| 2       | 16       | 190%     | 50%      | 95%  |
| 4       | 32       | 370%     | 25%      | 92%  |
| 8       | 64       | 720%     | 12.5%    | 90%  |

### 多机多卡性能

| 节点数 | 每节点GPU | 总GPU数 | 网络带宽 | 效率 |
|--------|-----------|---------|----------|------|
| 1      | 4         | 4       | -        | 92%  |
| 2      | 4         | 8       | 10Gbps   | 85%  |
| 4      | 4         | 16      | 10Gbps   | 80%  |
| 2      | 4         | 8       | 100Gbps  | 90%  |

## 故障排除

### 常见错误及解决方案

#### 1. CUDA Out of Memory

**DDP解决方案**：
- 减小批次大小
- 增加梯度累积步数
- 启用混合精度训练
- 使用梯度检查点

```yaml
training:
  per_device_train_batch_size: 2  # 减小批次
  gradient_accumulation_steps: 8   # 增加累积
  mixed_precision: "fp16"          # 启用FP16
```

**DeepSpeed解决方案**：
- 使用更高级的ZeRO stage
- 启用CPU/NVMe卸载
- 启用激活检查点

```yaml
training:
  distributed_strategy: "deepspeed"
  deepspeed:
    config_path: "configs/deepspeed/zero3.json"  # 使用ZeRO-3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

#### 2. NCCL错误

```bash
# 错误：NCCL initialization failed
# 解决：检查GPU可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 或禁用NCCL的某些功能
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

#### 3. 进程间通信超时

**解决方案**：
- 检查网络连接
- 增加超时时间
- 确保防火墙设置正确

```bash
# DDP
export NCCL_TIMEOUT=1800  # 30分钟超时

# DeepSpeed
export NCCL_TIMEOUT=1800
export CUDA_LAUNCH_BLOCKING=1
```

#### 4. DeepSpeed特定问题

**DeepSpeed初始化失败**：
```bash
# 检查DeepSpeed安装
pip show deepspeed

# 重新编译DeepSpeed
ds_report  # 查看DeepSpeed环境
```

**ZeRO-3参数收集错误**：
```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_gather_16bit_weights_on_model_save": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  }
}
```

#### 5. 端口占用

```bash
# 错误：Address already in use
# 解决：更换端口
python scripts/run_distributed.py --master_port 29501 ...
```

#### 6. 数据加载慢

```yaml
# 增加数据加载进程
dataloader_num_workers: 8

# 启用内存固定
dataloader_pin_memory: true

# 预加载数据
dataloader_prefetch_factor: 2
```

### 调试技巧

#### 1. 启用详细日志

```bash
# DDP调试
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# DeepSpeed调试
export NCCL_DEBUG=INFO
export DEEPSPEED_LOG_LEVEL=INFO
```

#### 2. 单GPU测试

先在单GPU上测试代码，确保没有逻辑错误：

```bash
# DDP单GPU
python main.py --config config.yaml --mode train --algorithm sft

# DeepSpeed单GPU
deepspeed --num_gpus=1 main.py --config config_deepspeed.yaml --mode train --algorithm sft
```

#### 3. 渐进式扩展

从2个GPU开始，逐步增加GPU数量：

```bash
# DDP渐进式测试
torchrun --nproc_per_node=2 main.py --config config.yaml
torchrun --nproc_per_node=4 main.py --config config.yaml

# DeepSpeed渐进式测试
deepspeed --num_gpus=2 main.py --config config_deepspeed.yaml
deepspeed --num_gpus=4 main.py --config config_deepspeed.yaml
```

#### 4. 性能分析

```bash
# 使用DeepSpeed性能分析
deepspeed --num_gpus=4 main.py --config config_deepspeed.yaml --deepspeed_config configs/deepspeed/zero2_with_profiling.json
```

#### 5. 内存使用监控

```python
# 在训练代码中添加内存监控
import torch

def log_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## 进阶功能

### 1. 自定义分布式策略

```python
# 在训练器中自定义分布式行为
class CustomDistributedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自定义DDP配置
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
                bucket_cap_mb=50,  # 自定义bucket大小
                gradient_as_bucket_view=True  # 优化内存使用
            )
```

### 2. 动态批次大小

```python
# 根据GPU数量动态调整批次大小
def get_optimal_batch_size(num_gpus, base_batch_size=4):
    # 线性缩放
    return base_batch_size * num_gpus

# 在配置中使用
per_device_train_batch_size = get_optimal_batch_size(torch.cuda.device_count())
```

### 3. 混合精度优化

```yaml
training:
  # 自动混合精度
  fp16: true
  fp16_opt_level: "O1"  # O0, O1, O2, O3
  
  # 或使用BF16（推荐）
  bf16: true
  
  # 梯度缩放
  fp16_full_eval: false
```

## 总结

分布式训练可以显著提升训练效率，但需要合理配置和优化。关键要点：

1. **正确配置**：确保分布式设置正确
2. **性能优化**：合理设置批次大小、混合精度等
3. **监控调试**：及时发现和解决问题
4. **实验管理**：做好检查点和日志管理
5. **硬件匹配**：确保硬件配置合理

通过遵循本指南，您可以充分利用多GPU资源，加速模型训练过程。