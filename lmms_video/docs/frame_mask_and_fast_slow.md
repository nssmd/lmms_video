# 帧级别Mask和快慢帧处理机制

本文档详细介绍自回归视频重建模块中的两个关键特性：**帧级别因果mask**和**快慢帧处理机制**。

## 🎯 核心概念

### 1. 帧级别因果Mask (Frame-Level Causal Mask)

**为什么需要帧级别mask？**

在视频自回归预测中，我们希望预测第t帧时，只能看到前面的帧(0到t-1)，而不能看到未来的帧。传统的token级别因果mask不够精确，因为：

```
问题：Token级别mask
Frame 1: [patch1, patch2, patch3]
Frame 2: [patch4, patch5, patch6]
Frame 3: [patch7, patch8, patch9]

如果只用token级别mask:
- patch4 可以看到 patch1, patch2, patch3 ✅
- patch4 也能看到 patch5 ❌ (同一帧内不应该有先后顺序)

解决：帧级别mask
- Frame 2的所有patch都能看到 Frame 1的所有patch ✅
- Frame 2的所有patch之间可以互相看到 ✅
- Frame 2的所有patch都不能看到 Frame 3的任何patch ✅
```

**实现原理：**

```python
# 创建帧级别因果mask
mask = create_causal_frame_mask(num_frames=3, num_patches_per_frame=3)

# 可视化 (1表示可以attend，0表示mask掉):
# 行=query, 列=key
     F1_P1 F1_P2 F1_P3 | F2_P1 F2_P2 F2_P3 | F3_P1 F3_P2 F3_P3
F1_P1  1     1     1   |  0     0     0   |  0     0     0
F1_P2  1     1     1   |  0     0     0   |  0     0     0
F1_P3  1     1     1   |  0     0     0   |  0     0     0
-------------------------------------------------------------
F2_P1  1     1     1   |  1     1     1   |  0     0     0
F2_P2  1     1     1   |  1     1     1   |  0     0     0
F2_P3  1     1     1   |  1     1     1   |  0     0     0
-------------------------------------------------------------
F3_P1  1     1     1   |  1     1     1   |  1     1     1
F3_P2  1     1     1   |  1     1     1   |  1     1     1
F3_P3  1     1     1   |  1     1     1   |  1     1     1
```

**关键特点：**
- ✅ 帧内patch可以互相看到（全1块）
- ✅ 只能看到历史帧（下三角结构）
- ✅ 不能看到未来帧（上三角全0）

### 2. 快慢帧处理机制 (Fast-Slow Frame Sampling)

**为什么需要快慢帧？**

视频理解需要在两个时间尺度上建模：
- **快帧（Fast）**: 捕捉细节运动和快速变化
- **慢帧（Slow）**: 理解长时序上下文和整体结构

参考SlowFast Networks和VideoMAE的设计思想。

**采样策略：**

```python
原始视频: [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15]
                                                                    └─ 当前时刻

# 快帧采样 (stride=1, num_fast=8)
# 从最近的帧开始高密度采样
Fast frames: [F8, F9, F10, F11, F12, F13, F14, F15]
              └────────────── 捕捉细节运动 ──────────┘

# 慢帧采样 (stride=4, num_slow=8)
# 从整个视频均匀采样
Slow frames: [F0, F2, F4, F6, F8, F10, F12, F14]
              └───────── 理解长时序上下文 ────────┘

# 合并策略：concat
Merged: [F0, F2, F4, F6, F8, F10, F12, F14] + [F8, F9, F10, F11, F12, F13, F14, F15]
        └──────── 慢帧 ─────────┘             └──────── 快帧 ─────────┘
```

**优势：**
- 🎯 快帧：高分辨率时序信息（用于动作识别、细节理解）
- 🌍 慢帧：全局上下文信息（用于场景理解、长期依赖）
- 💾 效率：相比全采样，减少冗余帧，节省计算

## 📚 API使用

### 1. 帧级别Mask生成

```python
from lmms_engine.models.video_frame_utils import FrameLevelMaskGenerator

mask_gen = FrameLevelMaskGenerator()

# 1. 标准因果mask
causal_mask = mask_gen.create_causal_frame_mask(
    num_frames=8,
    num_patches_per_frame=256,
    device=torch.device("cuda")
)
# 返回: [8*256, 8*256] 的bool tensor

# 2. 滑动窗口mask（每帧只看前window_size帧）
window_mask = mask_gen.create_sliding_window_frame_mask(
    num_frames=8,
    num_patches_per_frame=256,
    window_size=3,  # 每帧只看前3帧
    device=torch.device("cuda")
)

# 3. 快慢帧专用mask
fast_slow_mask = mask_gen.create_fast_slow_frame_mask(
    num_fast_frames=8,
    num_slow_frames=8,
    num_patches_per_frame=256,
    device=torch.device("cuda")
)
# 快帧可以attend所有慢帧 + 之前的快帧
# 慢帧之间可以互相attend

# 4. 可视化mask（调试用）
mask_gen.visualize_mask(causal_mask, title="Causal Frame Mask")
```

### 2. 快慢帧采样

```python
from lmms_engine.models.video_frame_utils import VideoFrameSampler

sampler = VideoFrameSampler()

# 视频: [B, T, C, H, W]
video = torch.randn(2, 16, 3, 224, 224)

# 采样快慢帧
fast_frames, slow_frames = sampler.sample_fast_slow_frames(
    video_frames=video,
    fast_stride=1,   # 快帧步长
    slow_stride=4,   # 慢帧步长
    num_fast=8,      # 快帧数量
    num_slow=8,      # 慢帧数量
)

print(f"快帧: {fast_frames.shape}")  # [2, 8, 3, 224, 224]
print(f"慢帧: {slow_frames.shape}")  # [2, 8, 3, 224, 224]

# 合并策略1: 拼接
merged_concat = sampler.merge_fast_slow_frames(
    fast_frames, slow_frames,
    merge_strategy="concat"
)
# [2, 16, 3, 224, 224] = [慢帧8] + [快帧8]

# 合并策略2: 交错
merged_interleave = sampler.merge_fast_slow_frames(
    fast_frames, slow_frames,
    merge_strategy="interleave"
)
# [2, 16, 3, 224, 224] = 慢帧和快帧交错排列
```

### 3. 在自回归模块中使用

```python
from lmms_engine.models.autoregressive_reconstruction import (
    create_autoregressive_reconstruction_module
)

# 配置
config = {
    "embedding_dim": 1152,
    "loss_weight": 0.1,

    # 启用帧级别因果mask
    "use_frame_causal_mask": True,

    # 启用快慢帧
    "use_fast_slow_frames": True,
    "fast_stride": 1,
    "slow_stride": 4,
    "num_fast": 8,
    "num_slow": 8,
}

# 创建模块
module = create_autoregressive_reconstruction_module(
    vision_tower=model.vision_tower,
    hidden_size=model.config.hidden_size,
    config=config
)

# 使用
loss = module.compute_autoregressive_loss(
    hidden_states=llm_hidden_states,  # [B, seq_len, D]
    video_frames=video_frames,         # [B, T, 3, H, W]
)
```

## 🔬 技术细节

### Mask的PyTorch实现细节

**问题：TransformerDecoder的mask格式**

PyTorch的`TransformerDecoder`要求：
- `True` = mask掉（不能attend）
- `False` = 可以attend

但直观上我们希望：
- `True` = 可以attend
- `False` = mask掉

**解决方案：**

```python
# 我们的mask_generator返回: True=可以attend
mask = mask_generator.create_causal_frame_mask(...)

# 传给TransformerDecoder前取反
mask_for_transformer = ~mask  # True=mask掉

# 使用
predictor(tgt=x, memory=x, tgt_mask=mask_for_transformer)
```

### 帧位置编码

```python
def create_frame_position_ids(num_frames, num_patches_per_frame):
    """
    创建帧级别位置ID

    示例: num_frames=3, num_patches_per_frame=2
    返回: [0, 0, 1, 1, 2, 2]
         └─F0─┘ └─F1─┘ └─F2─┘
    """
    frame_ids = torch.arange(num_frames)
    return frame_ids.repeat_interleave(num_patches_per_frame)
```

同一帧内的patch共享相同的帧位置ID，帮助模型理解帧的边界。

## 📊 实验建议

### 消融实验

建议进行以下消融实验来验证效果：

| 实验 | Frame Mask | Fast-Slow | 说明 |
|------|-----------|-----------|------|
| Baseline | ❌ | ❌ | 标准token级别mask，全帧采样 |
| +Frame Mask | ✅ | ❌ | 添加帧级别因果mask |
| +Fast-Slow | ❌ | ✅ | 添加快慢帧采样 |
| Full | ✅ | ✅ | 两者都启用 |

### 超参数调优

**帧级别mask:**
- `use_frame_causal_mask=True`: 推荐始终启用
- 如果内存受限，可以尝试`sliding_window_mask`，设置`window_size=4-8`

**快慢帧:**
```yaml
# 短视频（<30帧）
use_fast_slow_frames: false  # 直接用全部帧

# 中等视频（30-60帧）
use_fast_slow_frames: true
num_fast: 8
num_slow: 8
fast_stride: 1
slow_stride: 4

# 长视频（>60帧）
use_fast_slow_frames: true
num_fast: 16
num_slow: 8
fast_stride: 2
slow_stride: 8
```

## 🐛 调试工具

### 可视化Mask

```python
from lmms_engine.models.video_frame_utils import FrameLevelMaskGenerator

mask_gen = FrameLevelMaskGenerator()
mask = mask_gen.create_causal_frame_mask(4, 3)

# 保存为图片
mask_gen.visualize_mask(mask, title="My Frame Mask")
# 输出: my_frame_mask.png
```

### 验证快慢帧采样

```python
import torch
from lmms_engine.models.video_frame_utils import VideoFrameSampler

# 创建有标记的测试视频
video = torch.arange(16).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
video = video.expand(1, 16, 3, 224, 224).float()
# video[0, t, 0, 0, 0] = t (帧号)

sampler = VideoFrameSampler()
fast, slow = sampler.sample_fast_slow_frames(video, num_fast=4, num_slow=4)

print("快帧索引:", fast[0, :, 0, 0, 0])
print("慢帧索引:", slow[0, :, 0, 0, 0])
# 验证采样是否正确
```

## 📖 参考文献

1. **SlowFast Networks for Video Recognition** (Feichtenhofer et al., 2019)
   - 快慢帧的设计灵感来源

2. **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training** (Tong et al., 2022)
   - 视频自回归预测的应用

3. **Attention Is All You Need** (Vaswani et al., 2017)
   - Causal mask的理论基础

## 🔗 相关文件

- 核心实现: `src/lmms_engine/models/video_frame_utils.py`
- 集成模块: `src/lmms_engine/models/autoregressive_reconstruction.py`
- 配置示例: `examples/autoregressive_video_config.yaml`

---

**更新时间**: 2025-10-08
**作者**: lmms-engine-mini团队
