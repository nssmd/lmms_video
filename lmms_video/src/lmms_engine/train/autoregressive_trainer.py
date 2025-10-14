"""
支持自回归视频重建的Trainer
扩展lmms_engine.train.Trainer，添加自回归loss支持
"""
from typing import Dict, Optional, Union

import torch
from transformers.modeling_outputs import ModelOutput

from .trainer import Trainer


class AutoregressiveTrainer(Trainer):
    """支持自回归视频重建损失的训练器"""

    def __init__(self, *args, enable_autoregressive: bool = False, **kwargs):
        """
        Args:
            enable_autoregressive: 是否启用自回归视频重建
            *args, **kwargs: 传递给父类Trainer的参数
        """
        super().__init__(*args, **kwargs)
        self.enable_autoregressive = enable_autoregressive

        if self.enable_autoregressive:
            print("🔧 启用自回归视频重建训练模式")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        重写损失计算，添加自回归视频重建损失

        Args:
            model: 模型
            inputs: 输入数据，可能包含 'video_frames' 字段
            return_outputs: 是否返回模型输出
            num_items_in_batch: batch中的样本数

        Returns:
            loss or (loss, outputs)
        """
        # 提取视频帧（如果存在）
        video_frames = inputs.pop("video_frames", None)

        # 调用父类计算主损失
        if return_outputs:
            loss, outputs = super().compute_loss(
                model=model,
                inputs=inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            loss = super().compute_loss(
                model=model,
                inputs=inputs,
                return_outputs=False,
                num_items_in_batch=num_items_in_batch,
            )
            outputs = None

        # 如果启用了自回归重建且有视频帧数据
        if (
            self.enable_autoregressive
            and video_frames is not None
            and self.model.training
        ):
            # 检查模型是否有自回归重建模块
            if hasattr(model, "autoregressive_module"):
                autoregressive_module = model.autoregressive_module
            elif hasattr(model, "module") and hasattr(
                model.module, "autoregressive_module"
            ):
                # 处理DDP/FSDP wrapped模型
                autoregressive_module = model.module.autoregressive_module
            else:
                print(
                    "⚠️ 模型没有autoregressive_module，跳过自回归loss计算"
                )
                return (loss, outputs) if return_outputs else loss

            # 获取hidden states
            if outputs is not None and hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states
                if hidden_states is not None and len(hidden_states) > 0:
                    # 使用最后一层的hidden states
                    last_hidden_state = hidden_states[-1]

                    # 计算自回归重建损失
                    try:
                        autoregressive_loss = (
                            autoregressive_module.compute_autoregressive_loss(
                                last_hidden_state, video_frames
                            )
                        )

                        # 将自回归损失添加到主损失
                        loss = loss + autoregressive_loss

                        # 记录日志
                        if self.state.global_step % 10 == 0:
                            self.log(
                                {
                                    "autoregressive_loss": autoregressive_loss.item(),
                                    "main_loss": (loss - autoregressive_loss).item(),
                                }
                            )
                    except Exception as e:
                        print(f"⚠️ 计算自回归loss时出错: {e}")
                        # 如果计算失败，继续使用主损失

        return (loss, outputs) if return_outputs else loss
