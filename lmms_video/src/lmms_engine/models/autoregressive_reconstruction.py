"""
自回归视频重建模块 for LLaVA-OneVision
基于 LLaVA-NeXT 的真正实现逻辑

核心思想：
1. 视频特征经过池化后展平拼接成一个长序列（快帧用大 stride，慢帧用小 stride）
2. 使用阶梯型 causal mask 确保自回归预测（第 i 帧只能看到前 i-1 帧）
3. 在统一的 Transformer 中处理，通过 mask 控制信息流
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def get_2d_pool(features, height, width, stride=2, mode='average'):
    """
    2D 空间池化 (LLaVA-NeXT 实现)

    Args:
        features: [num_patches, D]
        height, width: 特征图空间尺寸
        stride: 池化步长
        mode: 池化模式

    Returns:
        pooled: [num_patches_pooled, D]
    """
    D = features.shape[-1]

    # [H, W, D]
    features = features.view(height, width, D)
    # [D, H, W]
    features = features.permute(2, 0, 1).contiguous().unsqueeze(0)

    if mode == "average":
        pooled = F.avg_pool2d(features, kernel_size=stride, stride=stride)
    elif mode == "max":
        pooled = F.max_pool2d(features, kernel_size=stride, stride=stride)
    elif mode == "bilinear":
        new_h, new_w = height // stride, width // stride
        pooled = F.interpolate(features, size=(new_h, new_w), mode='bilinear', align_corners=False)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")

    # [1, D, H', W'] -> [H'*W', D]
    pooled = pooled.squeeze(0).permute(1, 2, 0).contiguous()
    pooled = pooled.view(-1, D)

    return pooled


def create_frame_types(num_frames: int, strategy: str = "interleave", slow_ratio: float = 0.25):
    """
    创建帧类型标签 (0=慢帧, 1=快帧)

    Args:
        num_frames: 总帧数
        strategy: 'interleave', 'periodic', 'uniform'
        slow_ratio: 慢帧比例

    Returns:
        frame_types: [num_frames], 0=慢帧, 1=快帧
    """
    frame_types = torch.ones(num_frames, dtype=torch.long)

    if strategy == "interleave":
        # 交错: 慢,快,快,快,慢,快,快,快...
        period = int(1.0 / slow_ratio)
        for i in range(0, num_frames, period):
            frame_types[i] = 0
    elif strategy == "periodic":
        num_slow = max(1, int(num_frames * slow_ratio))
        frame_types[:num_slow] = 0
    elif strategy == "uniform":
        num_slow = max(1, int(num_frames * slow_ratio))
        indices = torch.linspace(0, num_frames - 1, num_slow).long()
        frame_types[indices] = 0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return frame_types


def create_causal_frame_mask(frame_token_counts: List[int], device='cuda'):
    """
    创建阶梯型 causal mask

    Args:
        frame_token_counts: 每帧的 token 数量（慢帧多，快帧少）

    Returns:
        mask: [total_tokens, total_tokens] 布尔 mask
              True = 被 mask (不能 attend)

    Example:
        frame_token_counts = [64, 16, 16, 64]  # 慢,快,快,慢

        Frame 0: [能看自己的64个token]
        Frame 1: [能看Frame0的64个token][能看自己的16个token]
        Frame 2: [能看Frame0+1的80个token][能看自己的16个token]
        Frame 3: [能看Frame0+1+2的96个token][能看自己的64个token]
    """
    total_tokens = sum(frame_token_counts)
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

    start_idx = 0
    for i, count in enumerate(frame_token_counts):
        end_idx = start_idx + count

        # 当前帧不能看到后续帧
        if end_idx < total_tokens:
            mask[start_idx:end_idx, end_idx:] = True

        start_idx = end_idx

    return mask


class AutoregressiveReconstructionModule(nn.Module):
    """
    自回归视频重建模块

    流程：
    1. 视频帧 -> 冻结 vision tower -> 视觉特征 [B, T, P, D]
    2. 根据帧类型池化 (慢帧 stride=4, 快帧 stride=8)
    3. 展平拼接成长序列 [B, total_tokens, D]
    4. LLM hidden states 投影到视觉空间 [B, total_tokens, D]
    5. Transformer + causal mask -> 预测下一帧
    """

    def __init__(self, vision_tower, hidden_size, config):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = config.get('embedding_dim', 1152)
        self.loss_weight = config.get('loss_weight', 0.15)

        # 快慢帧配置
        self.add_faster_video = config.get('add_faster_video', True)
        self.mm_spatial_pool_stride = config.get('mm_spatial_pool_stride', 4)
        self.mm_spatial_pool_mode = config.get('mm_spatial_pool_mode', 'average')
        self.frame_sampling_strategy = config.get('frame_sampling_strategy', 'interleave')
        self.slow_frame_ratio = config.get('slow_frame_ratio', 0.25)

        # 冻结 vision tower 副本
        self.frozen_vision_tower = copy.deepcopy(vision_tower)
        for param in self.frozen_vision_tower.parameters():
            param.requires_grad = False
        self.frozen_vision_tower.eval()

        # Projection layers
        self.llm_to_vision = nn.Linear(hidden_size, self.embedding_dim)

        # Transformer for autoregressive modeling
        num_heads = config.get('num_heads', 8)
        num_layers = config.get('num_layers', 3)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=self.embedding_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )

        print(f"✅ 自回归模块初始化:")
        print(f"   - LLM hidden_size: {hidden_size}")
        print(f"   - Vision embedding_dim: {self.embedding_dim}")
        print(f"   - 慢帧 stride: {self.mm_spatial_pool_stride}")
        print(f"   - 快帧 stride: {self.mm_spatial_pool_stride * 2 if self.add_faster_video else 'N/A'}")
        print(f"   - 帧采样策略: {self.frame_sampling_strategy}, 慢帧比例: {self.slow_frame_ratio}")
        print(f"   - 损失权重: {self.loss_weight}")

    @torch.no_grad()
    def encode_video_frames(self, video_frames):
        """
        使用冻结 vision tower 编码视频

        Args:
            video_frames: [B, T, C, H, W]

        Returns:
            embeddings: [B, T, num_patches, D]
        """
        B, T, C, H, W = video_frames.shape

        # 展平时间维度
        frames_flat = video_frames.view(B * T, C, H, W)

        # 通过冻结 vision tower
        embeddings_flat = self.frozen_vision_tower(frames_flat)  # [B*T, P, D]

        # 恢复时间维度
        num_patches = embeddings_flat.shape[1]
        embeddings = embeddings_flat.view(B, T, num_patches, self.embedding_dim)

        return embeddings

    def pool_and_flatten_features(self, video_embeddings, frame_types):
        """
        根据帧类型池化并展平视频特征

        Args:
            video_embeddings: [B, T, num_patches, D]
            frame_types: [T] - 0=慢帧, 1=快帧

        Returns:
            flattened: [B, total_tokens, D]
            frame_token_counts: List[int]
        """
        B, T, num_patches, D = video_embeddings.shape
        H = W = int(math.sqrt(num_patches))

        flattened_list = []
        frame_token_counts = []

        for t in range(T):
            frame_feat = video_embeddings[:, t, :, :]  # [B, num_patches, D]
            frame_type = frame_types[t].item()

            # 选择池化 stride
            if frame_type == 0:  # 慢帧
                stride = self.mm_spatial_pool_stride
            else:  # 快帧
                stride = self.mm_spatial_pool_stride * 2 if self.add_faster_video else self.mm_spatial_pool_stride

            # 池化每个 batch
            pooled_list = []
            for b in range(B):
                pooled = get_2d_pool(
                    frame_feat[b],
                    height=H,
                    width=W,
                    stride=stride,
                    mode=self.mm_spatial_pool_mode
                )  # [num_pooled, D]
                pooled_list.append(pooled)

            pooled_batch = torch.stack(pooled_list, dim=0)  # [B, num_pooled, D]
            flattened_list.append(pooled_batch)
            frame_token_counts.append(pooled_batch.shape[1])

        # 拼接所有帧
        flattened = torch.cat(flattened_list, dim=1)  # [B, total_tokens, D]

        return flattened, frame_token_counts

    def compute_autoregressive_loss(self, llm_hidden_states, video_frames):
        """
        计算自回归重建损失

        Args:
            llm_hidden_states: [B, seq_len, hidden_size]
            video_frames: [B, T, C, H, W]

        Returns:
            loss: 标量
        """
        B, T = video_frames.shape[0], video_frames.shape[1]
        device = video_frames.device

        # 1. 生成帧类型
        frame_types = create_frame_types(
            T,
            strategy=self.frame_sampling_strategy,
            slow_ratio=self.slow_frame_ratio
        ).to(device)

        # 2. 编码视频 (冻结 vision tower)
        video_embeddings = self.encode_video_frames(video_frames)  # [B, T, P, D]

        # 3. 池化并展平
        target_flat, frame_token_counts = self.pool_and_flatten_features(
            video_embeddings, frame_types
        )  # [B, total_tokens, D]

        total_tokens = target_flat.shape[1]

        # 4. LLM 特征投影到 vision 空间
        if llm_hidden_states.shape[1] < total_tokens:
            # 自适应池化
            llm_features = F.adaptive_avg_pool1d(
                llm_hidden_states.transpose(1, 2),
                total_tokens
            ).transpose(1, 2)
        else:
            llm_features = llm_hidden_states[:, :total_tokens, :]

        llm_projected = self.llm_to_vision(llm_features)  # [B, total_tokens, D]

        # 5. 创建 causal mask
        causal_mask = create_causal_frame_mask(frame_token_counts, device=device)

        # 6. Transformer 自回归建模
        transformer_out = self.transformer(
            llm_projected,
            mask=causal_mask,
            is_causal=False  # 我们自己提供 mask
        )  # [B, total_tokens, D]

        # 7. 预测下一帧
        predictions = self.prediction_head(transformer_out)  # [B, total_tokens, D]

        # 8. 计算损失: 每帧预测下一帧
        loss = 0.0
        count = 0

        start_idx = 0
        for i in range(len(frame_token_counts) - 1):
            current_count = frame_token_counts[i]
            next_start = start_idx + current_count
            next_count = frame_token_counts[i + 1]
            next_end = next_start + next_count

            # 当前帧的表示预测下一帧
            current_repr = predictions[:, start_idx:next_start, :].mean(dim=1)  # [B, D]
            target_repr = target_flat[:, next_start:next_end, :].mean(dim=1)  # [B, D]

            loss += F.mse_loss(current_repr, target_repr)
            count += 1

            start_idx = next_start

        if count > 0:
            loss = loss / count

        # 加权
        weighted_loss = self.loss_weight * loss

        return weighted_loss


def create_autoregressive_reconstruction_module(vision_tower, hidden_size, config):
    """
    创建自回归重建模块

    Args:
        vision_tower: 视觉编码器
        hidden_size: LLM hidden size
        config: 配置字典

    Returns:
        模块实例
    """
    return AutoregressiveReconstructionModule(vision_tower, hidden_size, config)
