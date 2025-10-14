# 自回归视频重建训练指南

本文档介绍如何在lmms-engine-mini中使用自回归视频重建功能，该功能参考了LLaVA-NeXT的实现。

## 功能概述

自回归视频重建模块为视频理解模型添加了一个辅助训练目标：预测视频序列中的下一帧特征。这个辅助损失可以帮助模型更好地理解视频的时序动态。

### 核心特性

- **冻结视觉编码器副本**: 创建视觉编码器的深拷贝用于生成目标特征，不影响原始模型训练
- **并行预测**: 支持并行处理多个时间步的预测
- **多模态融合**: 融合LLM特征和视觉特征进行预测
- **可配置损失权重**: 灵活控制自回归损失在总损失中的比重

## 架构说明

### 1. FrozenVisionEmbeddingProcessor

使用冻结的视觉编码器副本处理视频嵌入：
- 对视觉编码器进行深拷贝并冻结参数
- 编码视频帧序列: `[B, T, 3, H, W] -> [B, T, num_patches, embedding_dim]`
- 准备自回归目标: 输入为前T-1帧，目标为后T-1帧

### 2. ParallelAutoregressivePredictor

并行自回归预测器：
- 视觉特征投影到LLM空间
- 多模态注意力融合
- Transformer解码器预测
- 输出投影回视觉特征空间

### 3. AutoregressiveReconstructionModule

集成模块：
- 管理冻结编码器和预测器
- 计算MSE重建损失
- 应用可配置的损失权重

## 使用方法

### 1. 模型集成

在你的模型中添加自回归重建模块：

```python
from lmms_engine.models.autoregressive_reconstruction import (
    create_autoregressive_reconstruction_module
)

class YourVideoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... 初始化其他组件

        # 添加自回归重建模块
        if config.enable_autoregressive:
            self.autoregressive_module = create_autoregressive_reconstruction_module(
                vision_tower=self.vision_tower,
                hidden_size=config.hidden_size,
                config={
                    "embedding_dim": 1152,
                    "num_hist": 3,
                    "loss_weight": 0.1,
                    "num_heads": 8,
                    "num_layers": 3,
                }
            )

    def forward(self, ..., video_frames=None):
        # 标准前向传播
        outputs = super().forward(...)

        # 注意: 不需要在这里手动计算自回归loss
        # AutoregressiveTrainer会自动处理
        return outputs
```

### 2. 使用AutoregressiveTrainer

使用提供的AutoregressiveTrainer进行训练：

```python
from lmms_engine.train import AutoregressiveTrainer

trainer = AutoregressiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    enable_autoregressive=True,  # 启用自回归训练
)

trainer.train()
```

### 3. 配置文件示例

参见 `examples/autoregressive_video_config.yaml`:

```yaml
- type: trainer
  config:
    trainer_type: autoregressive_trainer
    enable_autoregressive: true

    dataset_config:
      type: vision
      # ... 数据集配置

    model_config:
      # ... 模型配置
      autoregressive_config:
        embedding_dim: 1152
        loss_weight: 0.1
```

### 4. 数据格式

视频数据应包含在dataset的返回字典中：

```python
{
    "input_ids": tensor,
    "labels": tensor,
    "pixel_values": tensor,
    "video_frames": tensor,  # [B, T, 3, H, W] - 关键！
}
```

## 配置参数

### 自回归模块配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_dim` | 1152 | 视觉嵌入维度 |
| `num_hist` | 3 | 历史帧数量 |
| `loss_weight` | 0.1 | 自回归损失权重（0-1） |
| `num_heads` | 8 | 注意力头数 |
| `num_layers` | 3 | Transformer解码器层数 |

### 损失权重调优建议

- **0.05-0.1**: 适合大多数场景，不会过度影响主任务
- **0.1-0.2**: 如果希望模型更关注时序建模
- **0.01-0.05**: 如果主任务已经表现很好，只需轻微辅助

## 训练示例

### 单GPU训练

```bash
python -m lmms_engine.launch.cli \
    --config examples/autoregressive_video_config.yaml
```

### 多GPU训练 (torchrun)

```bash
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="8000" \
    -m lmms_engine.launch.cli \
    --config examples/autoregressive_video_config.yaml
```

## 监控训练

训练时会记录以下指标：

- `loss`: 总损失（主损失 + 自回归损失）
- `autoregressive_loss`: 自回归重建损失
- `main_loss`: 主任务损失

在TensorBoard或日志中查看：

```bash
tensorboard --logdir output/autoregressive_video/runs
```

## 注意事项

### 1. 内存使用

自回归模块会创建视觉编码器的副本，会增加内存使用：
- 副本是冻结的，不需要存储梯度
- 如果内存紧张，可以减少 `num_layers` 或 `num_heads`

### 2. 计算开销

额外的计算开销主要来自：
- 视频帧的编码（通过冻结副本）
- 预测器的前向传播

建议：
- 从较小的视频帧数开始（如4-8帧）
- 根据GPU内存调整batch size

### 3. 与原始训练的兼容性

- 原始vision_tower保持可训练状态
- 主任务的训练不受影响
- 可以随时通过 `enable_autoregressive=False` 关闭

## 实现参考

本实现参考了以下源码：
- LLaVA-NeXT: `/home/v-zimowen/LLaVA-NeXT-main/llava/model/continuous_autoregressive_reconstruction.py`
- 核心理念：使用冻结的视觉编码器副本作为teacher，训练预测器学习视频时序

## 故障排查

### 问题：找不到 `autoregressive_module`

确保模型初始化时创建了该模块：
```python
if config.enable_autoregressive:
    self.autoregressive_module = create_autoregressive_reconstruction_module(...)
```

### 问题：显存不足

尝试：
- 减少 `num_layers` (如2层)
- 减少视频帧数
- 减小batch size
- 启用 `gradient_checkpointing`

### 问题：损失不收敛

调整：
- 降低 `loss_weight` (如0.05)
- 检查视频数据是否正确加载
- 验证 `hidden_states` 是否正确传递

## 扩展建议

可以考虑的扩展方向：

1. **条件预测**: 基于文本prompt条件化视频预测
2. **长时序建模**: 支持更长的视频序列
3. **多尺度预测**: 预测多个时间尺度的特征
4. **对比学习**: 结合对比损失增强表征学习

## 联系与反馈

如有问题或建议，请在GitHub仓库提issue。
