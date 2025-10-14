"""
视频帧处理工具
支持快慢帧采样、帧级别mask生成和多尺度帧加载
集成了 LLaVA-NeXT 的多尺度视频处理逻辑
"""
import math
import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleVideoFrameLoader:
    """
    多尺度视频帧加载器
    参考 LLaVA-NeXT 的视频处理逻辑，支持不同分辨率的帧采样
    """

    @staticmethod
    def load_video_frames(
        video_path: Union[str, Path],
        max_frames: int = 8,
        sample_rate: int = 1,
        target_size: Tuple[int, int] = (224, 224),
        enable_multiscale: bool = False,
        scale_factors: List[float] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        从视频文件加载帧序列，支持多尺度采样

        Args:
            video_path: 视频文件路径
            max_frames: 最大帧数
            sample_rate: 帧采样率
            target_size: 目标分辨率 (H, W)
            enable_multiscale: 是否启用多尺度加载
            scale_factors: 多尺度因子列表，例如 [1.0, 0.75, 0.5]

        Returns:
            如果 enable_multiscale=False: [T, 3, H, W] - 单尺度视频帧
            如果 enable_multiscale=True: List[[T, 3, H_i, W_i]] - 多尺度视频帧列表
        """
        if not Path(video_path).exists():
            print(f"⚠️ 视频文件不存在: {video_path}")
            # 返回dummy frames
            if enable_multiscale:
                scale_factors = scale_factors or [1.0, 0.75, 0.5]
                return [
                    torch.randn(max_frames, 3, int(target_size[0] * s), int(target_size[1] * s))
                    for s in scale_factors
                ]
            return torch.randn(max_frames, 3, target_size[0], target_size[1])

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 均匀采样帧索引
        if frame_count > max_frames * sample_rate:
            indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        else:
            indices = list(range(0, frame_count, sample_rate))[:max_frames]

        # 读取帧
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        # 如果帧数不足，重复最后一帧
        while len(frames) < max_frames:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))

        frames = frames[:max_frames]

        # 单尺度处理
        if not enable_multiscale:
            processed_frames = []
            for frame in frames:
                frame_resized = cv2.resize(frame, (target_size[1], target_size[0]))
                frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                processed_frames.append(frame_tensor)
            return torch.stack(processed_frames)  # [T, 3, H, W]

        # 多尺度处理
        scale_factors = scale_factors or [1.0, 0.75, 0.5]
        multiscale_frames = []

        for scale in scale_factors:
            scaled_size = (int(target_size[0] * scale), int(target_size[1] * scale))
            processed_frames = []
            for frame in frames:
                frame_resized = cv2.resize(frame, (scaled_size[1], scaled_size[0]))
                frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                processed_frames.append(frame_tensor)
            multiscale_frames.append(torch.stack(processed_frames))  # [T, 3, H_s, W_s]

        return multiscale_frames

    @staticmethod
    def apply_spatial_pooling(
        video_features: torch.Tensor,
        stride: int = 2,
        mode: str = 'average',
    ) -> torch.Tensor:
        """
        对视频特征应用空间池化（参考 LLaVA-NeXT 的 get_2dPool）

        Args:
            video_features: [num_frames, num_patches, hidden_dim] - 视频特征
            stride: 池化步长
            mode: 池化模式 ('average', 'max', 'bilinear')

        Returns:
            pooled_features: [num_frames, num_patches_pooled, hidden_dim]
        """
        num_frames, num_patches, hidden_dim = video_features.shape

        # 计算空间尺寸 (假设是方形)
        height = width = int(math.sqrt(num_patches))

        # 重塑为空间形状
        video_features = video_features.view(num_frames, height, width, hidden_dim)

        # 调整维度顺序以适配池化操作
        video_features = video_features.permute(0, 3, 1, 2).contiguous()  # [T, D, H, W]

        # 应用池化
        if mode == "average":
            pooled_features = F.avg_pool2d(video_features, stride)
        elif mode == "max":
            pooled_features = F.max_pool2d(video_features, stride)
        elif mode == "bilinear":
            height, width = video_features.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            pooled_features = F.interpolate(
                video_features, size=scaled_shape, mode='bilinear', align_corners=False
            )
        else:
            raise ValueError(f"不支持的池化模式: {mode}")

        # 恢复原始维度顺序
        pooled_features = pooled_features.permute(0, 2, 3, 1)  # [T, H', W', D]

        # 重塑回序列形状
        pooled_features = pooled_features.view(num_frames, -1, hidden_dim)

        return pooled_features


class VideoFrameSampler:
    """
    视频帧采样器，支持快慢帧处理

    快帧：高帧率采样，用于捕捉细节运动
    慢帧：低帧率采样，用于理解长时序上下文
    """

    @staticmethod
    def sample_fast_slow_frames(
        video_frames: torch.Tensor,
        fast_stride: int = 1,
        slow_stride: int = 4,
        num_fast: int = 8,
        num_slow: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从视频中采样快慢帧

        Args:
            video_frames: [B, T, C, H, W] - 原始视频帧
            fast_stride: 快帧采样步长
            slow_stride: 慢帧采样步长
            num_fast: 快帧数量
            num_slow: 慢帧数量

        Returns:
            fast_frames: [B, num_fast, C, H, W] - 快帧
            slow_frames: [B, num_slow, C, H, W] - 慢帧
        """
        B, T, C, H, W = video_frames.shape

        # 快帧采样：从最近的帧开始，高密度采样
        fast_indices = torch.arange(
            T - num_fast * fast_stride, T, fast_stride, device=video_frames.device
        )
        fast_indices = torch.clamp(fast_indices, 0, T - 1)
        fast_frames = video_frames[:, fast_indices[-num_fast:], :, :, :]

        # 慢帧采样：从整个视频均匀采样
        slow_indices = torch.linspace(0, T - 1, num_slow, device=video_frames.device).long()
        slow_frames = video_frames[:, slow_indices, :, :, :]

        return fast_frames, slow_frames

    @staticmethod
    def merge_fast_slow_frames(
        fast_frames: torch.Tensor,
        slow_frames: torch.Tensor,
        merge_strategy: str = "interleave",
    ) -> torch.Tensor:
        """
        合并快慢帧

        Args:
            fast_frames: [B, T_fast, C, H, W]
            slow_frames: [B, T_slow, C, H, W]
            merge_strategy: 合并策略
                - "concat": 直接拼接 [slow_frames, fast_frames]
                - "interleave": 交错排列

        Returns:
            merged_frames: [B, T_total, C, H, W]
        """
        if merge_strategy == "concat":
            # [慢帧序列] + [快帧序列]
            return torch.cat([slow_frames, fast_frames], dim=1)

        elif merge_strategy == "interleave":
            # 交错排列慢帧和快帧
            B, T_slow, C, H, W = slow_frames.shape
            T_fast = fast_frames.shape[1]

            # 计算交错比例
            if T_slow >= T_fast:
                # 慢帧更多，在慢帧中插入快帧
                result = []
                slow_step = T_slow // T_fast
                for i in range(T_fast):
                    result.append(slow_frames[:, i * slow_step : (i + 1) * slow_step, :, :, :])
                    result.append(fast_frames[:, i : i + 1, :, :, :])
                return torch.cat(result, dim=1)
            else:
                # 快帧更多，在快帧中插入慢帧
                result = []
                fast_step = T_fast // T_slow
                for i in range(T_slow):
                    result.append(fast_frames[:, i * fast_step : (i + 1) * fast_step, :, :, :])
                    result.append(slow_frames[:, i : i + 1, :, :, :])
                return torch.cat(result, dim=1)

        else:
            raise ValueError(f"Unknown merge_strategy: {merge_strategy}")


class FrameLevelMaskGenerator:
    """
    帧级别mask生成器
    用于自回归视频预测的因果mask
    """

    @staticmethod
    def create_causal_frame_mask(
        num_frames: int,
        num_patches_per_frame: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        创建帧级别的因果mask

        每一帧只能看到它之前的所有帧（包括自己）

        Args:
            num_frames: 帧数量 T
            num_patches_per_frame: 每帧的patch数量 P
            device: 设备

        Returns:
            mask: [T*P, T*P] - 因果attention mask
                True表示可以attend，False表示mask掉
        """
        # 创建帧级别的因果mask [T, T]
        frame_mask = torch.tril(torch.ones(num_frames, num_frames, device=device))

        # 扩展到patch级别 [T*P, T*P]
        # 每个帧内的patch可以互相看到，但不能看到未来帧的patch
        mask = frame_mask.repeat_interleave(num_patches_per_frame, dim=0)
        mask = mask.repeat_interleave(num_patches_per_frame, dim=1)

        return mask.bool()

    @staticmethod
    def create_sliding_window_frame_mask(
        num_frames: int,
        num_patches_per_frame: int,
        window_size: int = 3,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        创建滑动窗口帧mask

        每一帧只能看到前window_size个帧

        Args:
            num_frames: 帧数量
            num_patches_per_frame: 每帧的patch数量
            window_size: 滑动窗口大小
            device: 设备

        Returns:
            mask: [T*P, T*P] - 滑动窗口mask
        """
        # 创建帧级别的滑动窗口mask [T, T]
        frame_mask = torch.zeros(num_frames, num_frames, device=device)
        for i in range(num_frames):
            start = max(0, i - window_size + 1)
            frame_mask[i, start : i + 1] = 1

        # 扩展到patch级别
        mask = frame_mask.repeat_interleave(num_patches_per_frame, dim=0)
        mask = mask.repeat_interleave(num_patches_per_frame, dim=1)

        return mask.bool()

    @staticmethod
    def create_fast_slow_frame_mask(
        num_fast_frames: int,
        num_slow_frames: int,
        num_patches_per_frame: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        创建快慢帧的attention mask

        快帧可以attend到所有慢帧 + 之前的快帧
        慢帧之间可以互相attend

        Args:
            num_fast_frames: 快帧数量
            num_slow_frames: 慢帧数量
            num_patches_per_frame: 每帧的patch数量
            device: 设备

        Returns:
            mask: [(T_slow + T_fast)*P, (T_slow + T_fast)*P]
        """
        total_frames = num_slow_frames + num_fast_frames
        frame_mask = torch.zeros(total_frames, total_frames, device=device)

        # 慢帧之间可以互相attend
        frame_mask[:num_slow_frames, :num_slow_frames] = 1

        # 快帧可以attend到所有慢帧
        frame_mask[num_slow_frames:, :num_slow_frames] = 1

        # 快帧之间的因果mask
        fast_causal = torch.tril(
            torch.ones(num_fast_frames, num_fast_frames, device=device)
        )
        frame_mask[num_slow_frames:, num_slow_frames:] = fast_causal

        # 扩展到patch级别
        mask = frame_mask.repeat_interleave(num_patches_per_frame, dim=0)
        mask = mask.repeat_interleave(num_patches_per_frame, dim=1)

        return mask.bool()

    @staticmethod
    def visualize_mask(mask: torch.Tensor, title: str = "Frame Mask"):
        """
        可视化mask（调试用）

        Args:
            mask: [N, N] - mask矩阵
            title: 标题
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(mask.cpu().float().numpy(), cmap="Greys", interpolation="nearest")
            plt.title(title)
            plt.xlabel("Key Frame Patches")
            plt.ylabel("Query Frame Patches")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{title.replace(' ', '_').lower()}.png")
            plt.close()
            print(f"✅ Mask可视化保存到: {title.replace(' ', '_').lower()}.png")
        except ImportError:
            print("⚠️ matplotlib未安装，跳过可视化")


def create_frame_position_ids(
    num_frames: int,
    num_patches_per_frame: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    创建帧级别的位置ID

    Args:
        num_frames: 帧数量
        num_patches_per_frame: 每帧的patch数量
        device: 设备

    Returns:
        position_ids: [num_frames * num_patches_per_frame]
            每个patch的位置ID，同一帧内的patch有相同的帧位置ID
    """
    # 每一帧内的patch共享同一个帧位置ID
    frame_ids = torch.arange(num_frames, device=device)
    position_ids = frame_ids.repeat_interleave(num_patches_per_frame)
    return position_ids


# 示例使用
if __name__ == "__main__":
    # 测试帧级别因果mask
    print("测试帧级别因果mask...")
    num_frames = 4
    num_patches = 3

    mask_gen = FrameLevelMaskGenerator()

    # 1. 因果mask
    causal_mask = mask_gen.create_causal_frame_mask(num_frames, num_patches)
    print(f"因果mask形状: {causal_mask.shape}")
    print(f"因果mask示例:\n{causal_mask[:6, :6].int()}")

    # 2. 滑动窗口mask
    window_mask = mask_gen.create_sliding_window_frame_mask(
        num_frames, num_patches, window_size=2
    )
    print(f"\n滑动窗口mask示例:\n{window_mask[:6, :6].int()}")

    # 3. 快慢帧mask
    fast_slow_mask = mask_gen.create_fast_slow_frame_mask(
        num_fast_frames=2, num_slow_frames=2, num_patches_per_frame=num_patches
    )
    print(f"\n快慢帧mask形状: {fast_slow_mask.shape}")
    print(f"快慢帧mask示例:\n{fast_slow_mask[:6, :6].int()}")

    # 4. 测试快慢帧采样
    print("\n测试快慢帧采样...")
    sampler = VideoFrameSampler()
    video = torch.randn(2, 16, 3, 224, 224)  # [B, T, C, H, W]
    fast, slow = sampler.sample_fast_slow_frames(video, num_fast=4, num_slow=4)
    print(f"快帧形状: {fast.shape}")
    print(f"慢帧形状: {slow.shape}")

    merged = sampler.merge_fast_slow_frames(fast, slow, merge_strategy="concat")
    print(f"合并后形状: {merged.shape}")
