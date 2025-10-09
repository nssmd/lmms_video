"""
视频帧处理工具
支持快慢帧采样和帧级别mask生成
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


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
