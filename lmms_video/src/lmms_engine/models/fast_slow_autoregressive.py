"""
LLaVA-NeXT 风格的快慢帧自回归重建
关键设计：
1. 每个视频中，某些帧标记为慢帧(stride=4池化)，某些帧标记为快帧(stride=8池化)
2. 慢帧保留更多patch（更多细节），快帧patch更少（更大感受野）
3. 快慢帧穿插排列，使用专门的mask实现帧级别自回归生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import copy

from .video_frame_utils import MultiScaleVideoFrameLoader


class FastSlowFrameSampler:
    """
    快慢帧采样器：根据策略决定哪些帧是快帧，哪些帧是慢帧
    """

    @staticmethod
    def create_frame_types(
        num_frames: int,
        strategy: str = "interleave",
        slow_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        创建帧类型标记

        Args:
            num_frames: 总帧数
            strategy: 采样策略
                - "interleave": 交错排列，例如 [S, F, S, F, ...]
                - "first_slow": 前面慢帧，后面快帧
                - "uniform": 均匀分布慢帧
            slow_ratio: 慢帧占比

        Returns:
            frame_types: [num_frames] - 0表示慢帧，1表示快帧
        """
        frame_types = torch.zeros(num_frames, dtype=torch.long)

        if strategy == "interleave":
            # 交错：每隔一个帧切换类型
            # 例如：S F S F S F S F
            for i in range(num_frames):
                if i % 2 == 1:  # 奇数位置是快帧
                    frame_types[i] = 1

        elif strategy == "first_slow":
            # 前面慢帧，后面快帧
            num_slow = int(num_frames * slow_ratio)
            frame_types[num_slow:] = 1

        elif strategy == "uniform":
            # 均匀分布慢帧
            num_slow = int(num_frames * slow_ratio)
            slow_indices = torch.linspace(0, num_frames - 1, num_slow, dtype=torch.long)
            # 默认全是快帧
            frame_types[:] = 1
            # 标记慢帧
            frame_types[slow_indices] = 0

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return frame_types


class FastSlowAutoregressiveModule(nn.Module):
    """
    快慢帧自回归重建模块（LLaVA-NeXT 风格）

    核心逻辑：
    1. 根据frame_types对每个帧应用不同池化
       - 慢帧: stride=mm_spatial_pool_stride (例如4) -> 更多patches
       - 快帧: stride=mm_spatial_pool_stride*2 (例如8) -> 更少patches
    2. 构建穿插的序列：[slow_patches, fast_patches, slow_patches, ...]
    3. 使用快慢帧专用mask实现自回归
    """

    def __init__(
        self,
        vision_tower,
        hidden_size: int,
        embedding_dim: int = 1152,
        loss_weight: float = 0.15,
        num_heads: int = 8,
        num_layers: int = 3,
        # 快慢帧参数
        mm_spatial_pool_stride: int = 4,
        mm_spatial_pool_mode: str = "average",
        frame_sampling_strategy: str = "interleave",
        slow_frame_ratio: float = 0.5,
    ):
        """
        Args:
            vision_tower: 视觉编码器
            hidden_size: LLM隐藏层大小
            embedding_dim: 视觉嵌入维度
            loss_weight: 损失权重
            num_heads: 注意力头数
            num_layers: Transformer层数
            mm_spatial_pool_stride: 慢帧池化步长
            mm_spatial_pool_mode: 池化模式
            frame_sampling_strategy: 快慢帧采样策略
            slow_frame_ratio: 慢帧占比
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.loss_weight = loss_weight

        # 快慢帧参数
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.fast_stride = mm_spatial_pool_stride * 2  # 快帧stride是慢帧的2倍
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.frame_sampling_strategy = frame_sampling_strategy
        self.slow_frame_ratio = slow_frame_ratio

        # 冻结的视觉编码器副本
        self.frozen_vision_tower = copy.deepcopy(vision_tower)
        for param in self.frozen_vision_tower.parameters():
            param.requires_grad = False
        self.frozen_vision_tower.eval()

        # 多尺度加载器
        self.multiscale_loader = MultiScaleVideoFrameLoader()

        # 快慢帧采样器
        self.frame_sampler = FastSlowFrameSampler()

        # 特征投影
        self.slow_projection = nn.Linear(embedding_dim, hidden_size)
        self.fast_projection = nn.Linear(embedding_dim, hidden_size)

        # Transformer预测器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.predictor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_projection = nn.Linear(hidden_size, embedding_dim)

        # 帧类型嵌入
        self.slow_type_embedding = nn.Parameter(torch.randn(hidden_size))
        self.fast_type_embedding = nn.Parameter(torch.randn(hidden_size))

        self.reconstruction_criterion = nn.MSELoss()

        print(f"✅ 快慢帧自回归模块初始化完成")
        print(f"   - 慢帧 stride: {self.mm_spatial_pool_stride}")
        print(f"   - 快帧 stride: {self.fast_stride}")
        print(f"   - 采样策略: {frame_sampling_strategy}")
        print(f"   - 慢帧占比: {slow_frame_ratio}")

    def create_fast_slow_mask(
        self,
        frame_types: torch.Tensor,
        slow_patches_per_frame: int,
        fast_patches_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        创建快慢帧专用的因果mask

        关键逻辑：
        - 每个位置只能attend到之前的所有帧的patches
        - 慢帧有更多patches，快帧有更少patches
        - 需要考虑不同帧的patches数量不同

        Args:
            frame_types: [num_frames] - 0=慢帧, 1=快帧
            slow_patches_per_frame: 慢帧的patches数量
            fast_patches_per_frame: 快帧的patches数量
            device: 设备

        Returns:
            mask: [total_patches, total_patches] - True表示mask掉
        """
        num_frames = len(frame_types)

        # 计算每个帧的起始位置
        frame_start_positions = []
        current_pos = 0
        for frame_idx in range(num_frames):
            frame_start_positions.append(current_pos)
            if frame_types[frame_idx] == 0:  # 慢帧
                current_pos += slow_patches_per_frame
            else:  # 快帧
                current_pos += fast_patches_per_frame

        total_patches = current_pos
        frame_start_positions.append(total_patches)  # 添加结束位置

        # 创建因果mask
        mask = torch.ones(total_patches, total_patches, dtype=torch.bool, device=device)

        # 对每个帧，允许attend到之前的所有帧
        for frame_idx in range(num_frames):
            start = frame_start_positions[frame_idx]
            end = frame_start_positions[frame_idx + 1]

            # 当前帧的所有patches可以attend到之前所有帧的所有patches（包括自己）
            mask[start:end, :end] = False

        return mask

    def compute_autoregressive_loss(
        self,
        hidden_states: torch.Tensor,
        video_frames: torch.Tensor,
        frame_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算快慢帧自回归损失

        Args:
            hidden_states: [B, seq_len, hidden_size] - LLM隐藏状态
            video_frames: [B, T, 3, H, W] - 视频帧
            frame_types: [B, T] - 帧类型，0=慢帧，1=快帧（可选）

        Returns:
            loss: 自回归重建损失
        """
        B, T, C, H, W = video_frames.shape
        device = video_frames.device

        # 1. 如果没有提供frame_types，自动生成
        if frame_types is None:
            frame_types = self.frame_sampler.create_frame_types(
                num_frames=T,
                strategy=self.frame_sampling_strategy,
                slow_ratio=self.slow_frame_ratio,
            )
            frame_types = frame_types.unsqueeze(0).expand(B, -1).to(device)  # [B, T]

        # 2. 通过冻结的视觉编码器获取特征
        with torch.no_grad():
            frames_flat = video_frames.view(B * T, C, H, W)
            video_embeddings_flat = self.frozen_vision_tower(frames_flat)
            # video_embeddings_flat: [(B*T), num_patches, embedding_dim]

            num_patches_original = video_embeddings_flat.shape[1]
            video_embeddings = video_embeddings_flat.view(
                B, T, num_patches_original, self.embedding_dim
            )

        # 3. 对每个帧根据类型应用不同池化
        all_frame_features = []  # 存储所有帧的特征（不同尺寸）
        all_frame_targets = []  # 存储目标特征

        for batch_idx in range(B):
            batch_features = []
            batch_targets = []

            for frame_idx in range(T):
                frame_feature = video_embeddings[batch_idx, frame_idx]  # [P, D]
                frame_type = frame_types[batch_idx, frame_idx].item()

                if frame_type == 0:  # 慢帧
                    # 应用慢帧池化
                    pooled_feature = self.multiscale_loader.apply_spatial_pooling(
                        frame_feature.unsqueeze(0),  # [1, P, D]
                        stride=self.mm_spatial_pool_stride,
                        mode=self.mm_spatial_pool_mode,
                    ).squeeze(0)  # [P_slow, D]

                    # 投影并添加类型嵌入
                    projected = self.slow_projection(pooled_feature)
                    projected = projected + self.slow_type_embedding

                else:  # 快帧
                    # 应用快帧池化
                    pooled_feature = self.multiscale_loader.apply_spatial_pooling(
                        frame_feature.unsqueeze(0),  # [1, P, D]
                        stride=self.fast_stride,
                        mode=self.mm_spatial_pool_mode,
                    ).squeeze(0)  # [P_fast, D]

                    # 投影并添加类型嵌入
                    projected = self.fast_projection(pooled_feature)
                    projected = projected + self.fast_type_embedding

                batch_features.append(projected)
                batch_targets.append(pooled_feature)  # 目标是池化后的原始特征

            all_frame_features.append(batch_features)
            all_frame_targets.append(batch_targets)

        # 4. 构建因果mask（考虑不同帧的patches数量）
        # 以第一个batch为例（假设所有batch的frame_types相同）
        slow_patches = all_frame_features[0][0].shape[0] if frame_types[0, 0] == 0 else all_frame_features[0][1].shape[0]
        fast_patches = all_frame_features[0][1].shape[0] if frame_types[0, 1] == 1 else all_frame_features[0][0].shape[0]

        causal_mask = self.create_fast_slow_mask(
            frame_types[0],  # 使用第一个batch的frame_types
            slow_patches_per_frame=slow_patches,
            fast_patches_per_frame=fast_patches,
            device=device,
        )

        # 5. 对每个batch进行预测
        total_loss = 0.0
        for batch_idx in range(B):
            # 拼接所有帧的features
            frame_features_cat = torch.cat(all_frame_features[batch_idx], dim=0)  # [total_patches, hidden_size]
            frame_targets_cat = torch.cat(all_frame_targets[batch_idx], dim=0)  # [total_patches, embedding_dim]

            # Transformer预测（使用因果mask）
            predicted = self.predictor(
                tgt=frame_features_cat.unsqueeze(0),  # [1, total_patches, hidden_size]
                memory=frame_features_cat.unsqueeze(0),
                tgt_mask=causal_mask,
            ).squeeze(0)  # [total_patches, hidden_size]

            # 投影回嵌入空间
            predicted_embeddings = self.output_projection(predicted)  # [total_patches, embedding_dim]

            # 计算重建损失（跳过第一帧，因为没有历史信息）
            first_frame_patches = all_frame_features[batch_idx][0].shape[0]
            loss = self.reconstruction_criterion(
                predicted_embeddings[first_frame_patches:],
                frame_targets_cat[first_frame_patches:],
            )
            total_loss += loss

        return (total_loss / B) * self.loss_weight
