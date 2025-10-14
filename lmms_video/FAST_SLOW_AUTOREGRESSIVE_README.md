# 快慢帧自回归训练指南

LLaVA-NeXT 风格的快慢帧自回归视频重建，集成到 lmms-engine 框架。

## 🎯 核心特性

### 1. 快慢帧机制

- **慢帧 (Slow Frames)**: 使用 `stride=4` 的池化，保留更多 patches，捕捉细节
- **快帧 (Fast Frames)**: 使用 `stride=8` 的池化，patches 更少，更大感受野
- **穿插排列**: 快慢帧交替出现，例如 `S F S F S F...`

### 2. 自回归生成

- 帧级别因果 mask: 每一帧只能看到之前的帧
- 考虑不同帧的 patches 数量差异
- 支持并行处理多个时间步

### 3. 集成到 lmms-engine

- 无需重写训练循环
- 直接使用 YAML 配置
- 支持多 GPU 训练
- 自动日志和 checkpoint

## 🚀 快速开始

### 1. 环境准备

```bash
cd /home/aiscuser/lmms-engine-mini/lmms_video

# 安装依赖
pip install -e .

# 验证安装
python -c "from lmms_engine.models.fast_slow_autoregressive import FastSlowAutoregressiveModule; print('✅ 快慢帧模块已安装')"
```

### 2. 准备数据

**选项 A: 从 HuggingFace 加载（推荐）**
```bash
# 数据会自动下载
DATASET_PATH="lmms-lab/LLaVA-Video-178K"
```

**选项 B: 使用本地数据**
```bash
# 下载到本地
git clone https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K /path/to/data

# 或者直接使用 parquet 文件目录
DATASET_PATH="/path/to/LLaVA-Video-178K"
```

### 3. 修改配置

编辑 `configs/fast_slow_autoregressive.yaml`:

```yaml
# 数据路径
data_path: "lmms-lab/LLaVA-Video-178K"  # 或本地路径

# 快慢帧配置
autoregressive_config:
  mm_spatial_pool_stride: 4              # 慢帧池化步长
  frame_sampling_strategy: "interleave"   # 快慢帧采样策略
  slow_frame_ratio: 0.5                   # 慢帧占比
  loss_weight: 0.15                       # 损失权重
```

### 4. 启动训练

```bash
cd /home/aiscuser/lmms-engine-mini/lmms_video

# 使用提供的脚本
bash scripts/train_fast_slow.sh
```

或者手动启动：

```bash
# 单卡
python -m lmms_engine.train.runner \
    --config configs/fast_slow_autoregressive.yaml

# 多卡 (8卡)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 \
    -m lmms_engine.train.runner \
    --config configs/fast_slow_autoregressive.yaml
```

## 📝 配置说明

### 快慢帧核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mm_spatial_pool_stride` | 4 | 慢帧池化步长 |
| `frame_sampling_strategy` | `interleave` | 帧采样策略 |
| `slow_frame_ratio` | 0.5 | 慢帧占比 |
| `loss_weight` | 0.15 | 自回归损失权重 |

### 帧采样策略

1. **`interleave`** (交替): `S F S F S F S F ...`
   - 快慢帧均匀分布
   - 适合需要平衡细节和全局的场景

2. **`first_slow`** (前慢后快): `S S S S ... F F F F`
   - 前面用慢帧捕捉初始细节
   - 后面用快帧快速过渡

3. **`uniform`** (均匀分布): 根据 `slow_frame_ratio` 均匀分布慢帧
   - 灵活控制慢帧位置
   - 适合实验不同配置

### 训练参数建议

```yaml
# 对于 Qwen2-VL-7B
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-5
max_frames_num: 16

# 对于更大模型（72B）
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1e-5
max_frames_num: 8  # 减少帧数以节省内存
```

## 🔍 验证训练

### 1. 查看日志

```bash
# 训练日志
tail -f output/fast_slow_autoregressive_*/train.log

# 查找自回归损失
grep "autoregressive_loss" output/fast_slow_autoregressive_*/train.log
```

### 2. TensorBoard

```bash
tensorboard --logdir output/fast_slow_autoregressive_*/logs
```

### 3. 检查 Checkpoint

```bash
ls -lh output/fast_slow_autoregressive_*/checkpoint-*
```

## 🐛 常见问题

### 1. 内存不足 (OOM)

**解决方案**:
```yaml
# 减少batch size
per_device_train_batch_size: 1

# 减少帧数
max_frames_num: 8

# 增加快帧比例（快帧patches少，内存占用小）
slow_frame_ratio: 0.3

# 启用梯度检查点
gradient_checkpointing: true
```

### 2. 找不到模块

```bash
# 确保在正确目录
cd /home/aiscuser/lmms-engine-mini/lmms_video

# 重新安装
pip install -e .

# 验证
python -c "from lmms_engine.models.fast_slow_autoregressive import FastSlowAutoregressiveModule"
```

### 3. 数据加载失败

```bash
# 检查数据集路径
ls -la /path/to/LLaVA-Video-178K

# 使用 HuggingFace（自动处理）
data_path: "lmms-lab/LLaVA-Video-178K"

# 测试数据加载
python -c "from datasets import load_dataset; ds = load_dataset('lmms-lab/LLaVA-Video-178K', split='train', streaming=True); print(next(iter(ds)))"
```

### 4. 损失不下降

**检查配置**:
```yaml
# 确保启用输出hidden states
output_hidden_states: true

# 确保损失权重合适
loss_weight: 0.15  # 不要太小，也不要太大

# 检查学习率
learning_rate: 2e-5  # 可以尝试 1e-5 到 5e-5
```

## 📊 性能优化

### 1. 数据加载优化

```yaml
dataloader_num_workers: 4
dataloader_pin_memory: true
dataloader_persistent_workers: true
```

### 2. 混合精度训练

```yaml
bf16: true  # 推荐
tf32: true
fp16: false # 不推荐，精度损失较大
```

### 3. 梯度累积

```yaml
# 有效 batch size = per_device_batch_size * num_gpus * gradient_accumulation_steps
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
# 8卡: 有效batch size = 2 * 8 * 8 = 128
```

## 📈 实验建议

### 1. 消融实验

测试不同配置的影响：

```yaml
# 实验1: 只用慢帧（baseline）
slow_frame_ratio: 1.0

# 实验2: 只用快帧
slow_frame_ratio: 0.0

# 实验3: 50-50 混合（推荐）
slow_frame_ratio: 0.5

# 实验4: 70% 慢帧
slow_frame_ratio: 0.7
```

### 2. 采样策略对比

```yaml
# 测试三种策略
frame_sampling_strategy: "interleave"  # vs "first_slow" vs "uniform"
```

### 3. 损失权重调优

```yaml
# 从小到大测试
loss_weight: 0.05  # 轻量级
loss_weight: 0.10  # 平衡
loss_weight: 0.15  # 推荐
loss_weight: 0.20  # 激进
```

## 🔗 相关文件

- **核心模块**: `src/lmms_engine/models/fast_slow_autoregressive.py`
- **Trainer**: `src/lmms_engine/train/autoregressive_trainer.py`
- **配置文件**: `configs/fast_slow_autoregressive.yaml`
- **启动脚本**: `scripts/train_fast_slow.sh`
- **示例**: `examples/train_with_autoregressive.py`

## 📚 参考

- LLaVA-NeXT: https://github.com/LLaVA-VL/LLaVA-NeXT
- LLaVA-Video-178K: https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K
- lmms-engine: https://github.com/lmms-lab/lmms-engine

---

如有问题，请查看日志文件或提交 Issue。