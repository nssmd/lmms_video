"""
连续自回归视频重建损失模块
参考LLaVA-NeXT的实现，适配lmms-engine-mini框架
支持快慢帧处理和帧级别因果mask
"""
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .video_frame_utils import (
    FrameLevelMaskGenerator,
    VideoFrameSampler,
    MultiScaleVideoFrameLoader,
    create_frame_position_ids,
)


class FrozenVisionEmbeddingProcessor:
    """使用冻结的视觉编码器副本处理视频嵌入"""

    def __init__(self, vision_tower, embedding_dim: int = 1152):
        """
        Args:
            vision_tower: 原始视觉编码器（保持可训练）
            embedding_dim: 视觉嵌入维度
        """
        # 创建vision_tower的深拷贝，不影响原始模型
        self.frozen_vision_tower = copy.deepcopy(vision_tower)
        self.embedding_dim = embedding_dim

        # 只冻结副本，原始vision_tower继续训练
        for param in self.frozen_vision_tower.parameters():
            param.requires_grad = False
        self.frozen_vision_tower.eval()

        print("✅ 创建了冻结的vision_tower副本用于自回归损失")

    def encode_video_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        编码视频帧序列

        Args:
            video_frames: [B, T, 3, H, W] - 视频帧序列

        Returns:
            video_embeddings: [B, T, num_patches, embedding_dim] - 视频嵌入
        """
        with torch.no_grad():  # 冻结编码器副本，不参与梯度计算
            B, T, C, H, W = video_frames.shape
            # 将时间维度展平进行批处理
            frames_flat = rearrange(video_frames, "b t c h w -> (b t) c h w")

            # 通过冻结的视觉编码器副本
            embeddings_flat = self.frozen_vision_tower(
                frames_flat
            )  # [(B*T), num_patches, embedding_dim]

            # 恢复时间维度
            embeddings = rearrange(embeddings_flat, "(b t) p d -> b t p d", b=B, t=T)

        return embeddings

    def prepare_autoregressive_targets(
        self, video_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备自回归预测的目标序列

        Args:
            video_embeddings: [B, T, num_patches, embedding_dim]

        Returns:
            input_embeddings: [B, T-1, num_patches, embedding_dim] - 输入序列
            target_embeddings: [B, T-1, num_patches, embedding_dim] - 目标序列
        """
        # 输入: 前T-1帧
        input_embeddings = video_embeddings[:, :-1, :, :]
        # 目标: 后T-1帧 (预测下一帧)
        target_embeddings = video_embeddings[:, 1:, :, :]

        return input_embeddings, target_embeddings


class ParallelAutoregressivePredictor(nn.Module):
    """
    并行自回归预测器，支持并行处理多个时间步的预测
    支持帧级别的因果mask和快慢帧处理
    """

    def __init__(
        self,
        hidden_size: int,
        embedding_dim: int = 1152,
        num_hist: int = 3,
        num_heads: int = 8,
        num_layers: int = 3,
        use_frame_causal_mask: bool = True,
        use_fast_slow_frames: bool = False,
    ):
        """
        Args:
            hidden_size: LLM隐藏层大小
            embedding_dim: 视觉嵌入维度
            num_hist: 历史帧数量
            num_heads: 注意力头数
            num_layers: Transformer解码器层数
            use_frame_causal_mask: 是否使用帧级别因果mask
            use_fast_slow_frames: 是否使用快慢帧机制
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_hist = num_hist
        self.use_frame_causal_mask = use_frame_causal_mask
        self.use_fast_slow_frames = use_fast_slow_frames

        # 视觉特征投影
        self.visual_projection = nn.Linear(embedding_dim, hidden_size)

        # 多模态融合层 (LLM特征 + 视觉特征)
        self.multimodal_fusion = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

        # 预测器网络 - 使用自定义TransformerDecoder以支持帧级别mask
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.predictor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出投影回视觉特征空间
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, embedding_dim),
        )

        # 帧位置编码
        self.frame_position_embedding = nn.Embedding(512, hidden_size)  # 支持最多512帧

        # mask生成器
        self.mask_generator = FrameLevelMaskGenerator()

    def forward(
        self,
        llm_features: torch.Tensor,
        visual_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        并行自回归预测（支持帧级别因果mask）

        Args:
            llm_features: [B, seq_len, hidden_size] - LLM特征
            visual_embeddings: [B, T, num_patches, embedding_dim] - 视觉嵌入
            attention_mask: [T*num_patches, T*num_patches] - 可选的自定义mask

        Returns:
            predictions: [B, T-1, num_patches, embedding_dim] - 预测的下一帧特征
        """
        B, T, num_patches, _ = visual_embeddings.shape
        device = visual_embeddings.device

        # 1. 投影视觉特征到LLM空间
        visual_proj = self.visual_projection(
            visual_embeddings
        )  # [B, T, num_patches, hidden_size]

        # 2. 添加帧位置编码
        frame_pos_ids = create_frame_position_ids(T, num_patches, device)
        frame_pos_emb = self.frame_position_embedding(frame_pos_ids)  # [T*P, D]
        frame_pos_emb = frame_pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, T*P, D]

        # 3. 重塑为序列并添加位置编码
        visual_seq = rearrange(
            visual_proj, "b t p d -> b (t p) d"
        )  # [B, T*num_patches, hidden_size]
        visual_seq = visual_seq + frame_pos_emb

        # 4. 从LLM特征中提取视觉相关信息
        if llm_features.shape[1] >= num_patches:
            llm_visual = llm_features[:, :num_patches, :]  # [B, num_patches, hidden_size]
        else:
            # 如果LLM特征不够，进行自适应池化
            llm_visual = F.adaptive_avg_pool1d(
                llm_features.transpose(1, 2), num_patches
            ).transpose(1, 2)

        # 5. 多模态融合
        fused_features, _ = self.multimodal_fusion(
            query=visual_seq,  # 视觉序列作为query
            key=llm_visual,  # LLM特征作为key
            value=llm_visual,  # LLM特征作为value
        )

        # 6. 创建帧级别因果mask（关键！）
        if self.use_frame_causal_mask and attention_mask is None:
            # 创建帧级别的因果mask
            attention_mask = self.mask_generator.create_causal_frame_mask(
                num_frames=T, num_patches_per_frame=num_patches, device=device
            )
            # TransformerDecoder需要的mask格式：True表示mask掉（不能attend）
            # 我们的mask_generator返回的是：True表示可以attend
            # 所以需要取反
            attention_mask = ~attention_mask

        # 7. 并行预测下一帧（使用帧级别mask）
        predictions_seq = self.predictor(
            tgt=fused_features,  # [B, T*num_patches, hidden_size]
            memory=fused_features,  # [B, T*num_patches, hidden_size]
            tgt_mask=attention_mask,  # [T*P, T*P] - 帧级别因果mask
        )

        # 8. 投影回视觉特征空间
        predictions_proj = self.output_projection(
            predictions_seq
        )  # [B, T*num_patches, embedding_dim]

        # 9. 重塑回原始维度并取前T-1帧作为预测
        predictions = rearrange(
            predictions_proj, "b (t p) d -> b t p d", t=T, p=num_patches
        )
        predictions = predictions[:, :-1, :, :]  # [B, T-1, num_patches, embedding_dim]

        return predictions


class AutoregressiveReconstructionModule(nn.Module):
    """
    集成连续自回归视频重建的模块
    支持快慢帧处理、帧级别因果mask和多尺度特征处理
    集成了 LLaVA-NeXT 的高级自回归损失策略
    """

    def __init__(
        self,
        vision_tower,
        hidden_size: int,
        embedding_dim: int = 1152,
        num_hist: int = 3,
        loss_weight: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 3,
        use_frame_causal_mask: bool = True,
        use_fast_slow_frames: bool = False,
        fast_stride: int = 1,
        slow_stride: int = 4,
        num_fast: int = 8,
        num_slow: int = 8,
        # 新增：多尺度参数（LLaVA-NeXT 特性）
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_mode: str = 'average',
        add_faster_video: bool = False,
        enable_multiscale_pooling: bool = False,
    ):
        """
        Args:
            vision_tower: 视觉编码器
            hidden_size: LLM隐藏层大小
            embedding_dim: 视觉嵌入维度
            num_hist: 历史帧数量
            loss_weight: 自回归损失权重
            num_heads: 注意力头数
            num_layers: Transformer层数
            use_frame_causal_mask: 是否使用帧级别因果mask
            use_fast_slow_frames: 是否使用快慢帧机制
            fast_stride: 快帧采样步长
            slow_stride: 慢帧采样步长
            num_fast: 快帧数量
            num_slow: 慢帧数量
            mm_spatial_pool_stride: 空间池化步长（LLaVA-NeXT）
            mm_spatial_pool_mode: 池化模式（LLaVA-NeXT）
            add_faster_video: 是否添加更快的视频流（LLaVA-NeXT）
            enable_multiscale_pooling: 是否启用多尺度池化
        """
        super().__init__()

        self.use_fast_slow_frames = use_fast_slow_frames
        self.fast_stride = fast_stride
        self.slow_stride = slow_stride
        self.num_fast = num_fast
        self.num_slow = num_slow

        # LLaVA-NeXT 多尺度参数
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.add_faster_video = add_faster_video
        self.enable_multiscale_pooling = enable_multiscale_pooling

        # 冻结视觉编码器处理器
        self.frozen_embedding_processor = FrozenVisionEmbeddingProcessor(
            vision_tower=vision_tower, embedding_dim=embedding_dim
        )

        # 并行自回归预测器
        self.parallel_predictor = ParallelAutoregressivePredictor(
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            num_hist=num_hist,
            num_heads=num_heads,
            num_layers=num_layers,
            use_frame_causal_mask=use_frame_causal_mask,
            use_fast_slow_frames=use_fast_slow_frames,
        )

        # 损失权重
        self.loss_weight = loss_weight
        self.reconstruction_criterion = nn.MSELoss()

        # 快慢帧采样器
        if use_fast_slow_frames:
            self.frame_sampler = VideoFrameSampler()
            print(f"✅ 启用快慢帧机制: 快帧{num_fast}帧(步长{fast_stride}), 慢帧{num_slow}帧(步长{slow_stride})")

        # 多尺度帧加载器（LLaVA-NeXT）
        self.multiscale_loader = MultiScaleVideoFrameLoader()

        # 多尺度特征投影（LLaVA-NeXT）
        if enable_multiscale_pooling:
            self.slow_frame_projector = nn.Linear(embedding_dim, hidden_size)
            if add_faster_video:
                self.fast_frame_projector = nn.Linear(embedding_dim, hidden_size)
            print(f"✅ 启用多尺度池化: 步长={mm_spatial_pool_stride}, 模式={mm_spatial_pool_mode}, 快速流={add_faster_video}")

    def compute_autoregressive_loss(
        self, hidden_states: torch.Tensor, video_frames: torch.Tensor
    ) -> torch.Tensor:
        """
        计算自回归视频重建损失（支持快慢帧、帧级别mask和多尺度池化）
        集成了 LLaVA-NeXT 的多尺度特征处理

        Args:
            hidden_states: [B, seq_len, hidden_size] - LLM隐藏状态
            video_frames: [B, T, 3, H, W] - 视频帧序列

        Returns:
            loss: 自回归重建损失
        """
        # 0. 如果使用快慢帧，先进行采样，并记录快慢帧的数量
        fast_slow_mask = None
        if self.use_fast_slow_frames:
            fast_frames, slow_frames = self.frame_sampler.sample_fast_slow_frames(
                video_frames,
                fast_stride=self.fast_stride,
                slow_stride=self.slow_stride,
                num_fast=self.num_fast,
                num_slow=self.num_slow,
            )
            # 合并快慢帧
            video_frames = self.frame_sampler.merge_fast_slow_frames(
                fast_frames, slow_frames, merge_strategy="concat"
            )
            # 记录快慢帧信息，用于生成专用mask
            num_fast_actual = fast_frames.shape[1]
            num_slow_actual = slow_frames.shape[1]

        # 1. 通过冻结的视觉编码器副本获取目标特征
        video_embeddings = self.frozen_embedding_processor.encode_video_frames(
            video_frames
        )
        # video_embeddings: [B, T, num_patches, embedding_dim]

        # 1.1 如果使用快慢帧，创建快慢帧专用mask
        if self.use_fast_slow_frames:
            B, T, num_patches, D = video_embeddings.shape
            device = video_embeddings.device
            # 创建快慢帧mask（慢帧在前，快帧在后，因为merge_strategy="concat"）
            fast_slow_mask = self.parallel_predictor.mask_generator.create_fast_slow_frame_mask(
                num_fast_frames=num_fast_actual,
                num_slow_frames=num_slow_actual,
                num_patches_per_frame=num_patches,
                device=device,
            )
            # 转换为 TransformerDecoder 需要的格式（取反）
            fast_slow_mask = ~fast_slow_mask

        # 1.5 如果启用多尺度池化（LLaVA-NeXT 特性），应用空间池化
        if self.enable_multiscale_pooling and self.mm_spatial_pool_stride > 1:
            # 对视频特征应用空间池化（慢帧流）
            B, T, P, D = video_embeddings.shape
            video_embeddings_list = []

            for i in range(B):
                # 对每个batch应用池化
                slow_feat = self.multiscale_loader.apply_spatial_pooling(
                    video_embeddings[i],  # [T, P, D]
                    stride=self.mm_spatial_pool_stride,
                    mode=self.mm_spatial_pool_mode,
                )
                video_embeddings_list.append(slow_feat)

            video_embeddings_slow = torch.stack(video_embeddings_list, dim=0)

            # 如果启用快速视频流
            if self.add_faster_video:
                fast_stride = self.mm_spatial_pool_stride * 2
                video_embeddings_fast_list = []
                for i in range(B):
                    fast_feat = self.multiscale_loader.apply_spatial_pooling(
                        video_embeddings[i],
                        stride=fast_stride,
                        mode=self.mm_spatial_pool_mode,
                    )
                    video_embeddings_fast_list.append(fast_feat)
                video_embeddings_fast = torch.stack(video_embeddings_fast_list, dim=0)

                # 合并慢帧和快帧特征
                # 这里采用concatenation策略
                video_embeddings = torch.cat(
                    [video_embeddings_slow, video_embeddings_fast], dim=2
                )  # 在patch维度拼接
            else:
                video_embeddings = video_embeddings_slow

        # 2. 准备自回归目标
        (
            input_embeddings,
            target_embeddings,
        ) = self.frozen_embedding_processor.prepare_autoregressive_targets(
            video_embeddings
        )
        # input_embeddings: [B, T-1, num_patches, embedding_dim]
        # target_embeddings: [B, T-1, num_patches, embedding_dim]

        # 3. 训练自回归预测器（内部会使用帧级别因果mask）
        predicted_embeddings = self.parallel_predictor(
            llm_features=hidden_states,
            visual_embeddings=video_embeddings,
            attention_mask=None,  # 让predictor自动生成因果mask
        )
        # predicted_embeddings: [B, T-1, num_patches, embedding_dim]

        # 4. 计算重建损失
        reconstruction_loss = self.reconstruction_criterion(
            predicted_embeddings, target_embeddings
        )

        return reconstruction_loss * self.loss_weight


def create_autoregressive_reconstruction_module(
    vision_tower,
    hidden_size: int,
    config: Optional[dict] = None,
) -> AutoregressiveReconstructionModule:
    """
    创建自回归重建模块（增强版，集成 LLaVA-NeXT 特性）

    Args:
        vision_tower: 视觉编码器
        hidden_size: LLM隐藏层大小
        config: 配置字典，包含以下可选参数:
            - embedding_dim: 视觉嵌入维度 (默认: 1152)
            - num_hist: 历史帧数量 (默认: 3)
            - loss_weight: 自回归损失权重 (默认: 0.1)
            - num_heads: 注意力头数 (默认: 8)
            - num_layers: Transformer层数 (默认: 3)
            - use_frame_causal_mask: 是否使用帧级别因果mask (默认: True)
            - use_fast_slow_frames: 是否使用快慢帧机制 (默认: False)
            - fast_stride: 快帧采样步长 (默认: 1)
            - slow_stride: 慢帧采样步长 (默认: 4)
            - num_fast: 快帧数量 (默认: 8)
            - num_slow: 慢帧数量 (默认: 8)
            - mm_spatial_pool_stride: 空间池化步长 (默认: 2) [LLaVA-NeXT]
            - mm_spatial_pool_mode: 池化模式 (默认: 'average') [LLaVA-NeXT]
            - add_faster_video: 是否添加更快的视频流 (默认: False) [LLaVA-NeXT]
            - enable_multiscale_pooling: 是否启用多尺度池化 (默认: False) [LLaVA-NeXT]

    Returns:
        module: 自回归重建模块
    """
    config = config or {}

    return AutoregressiveReconstructionModule(
        vision_tower=vision_tower,
        hidden_size=hidden_size,
        embedding_dim=config.get("embedding_dim", 1152),
        num_hist=config.get("num_hist", 3),
        loss_weight=config.get("loss_weight", 0.1),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 3),
        use_frame_causal_mask=config.get("use_frame_causal_mask", True),
        use_fast_slow_frames=config.get("use_fast_slow_frames", False),
        fast_stride=config.get("fast_stride", 1),
        slow_stride=config.get("slow_stride", 4),
        num_fast=config.get("num_fast", 8),
        num_slow=config.get("num_slow", 8),
        # LLaVA-NeXT 特性
        mm_spatial_pool_stride=config.get("mm_spatial_pool_stride", 2),
        mm_spatial_pool_mode=config.get("mm_spatial_pool_mode", "average"),
        add_faster_video=config.get("add_faster_video", False),
        enable_multiscale_pooling=config.get("enable_multiscale_pooling", False),
    )
