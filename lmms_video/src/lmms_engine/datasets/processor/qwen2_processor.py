from typing import List, Optional

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoProcessor

from lmms_engine.mapping_func import register_processor

from .base_qwen2_5_vl_processor import BaseQwen2_5_DataProcessor


@register_processor("qwen2")
class Qwen2DataProcessor(BaseQwen2_5_DataProcessor):
    def _build_processor(self):
        processor = AutoProcessor.from_pretrained(self.config.processor_name)
        return processor

    def process(
        self,
        images: List[Image],
        hf_messages,
        audios: Optional[List[np.ndarray]] = None,
        sampling_rate: Optional[int] = None,
        videos=None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,  # Whether add a generation prompt at the end
        **kwargs,
    ):
        assert audios is None, "Qwen2DataProcessor does not support audio"
        assert videos is None, "Qwen2DataProcessor does not support video"
        return super().process(
            images,
            hf_messages,
            audios,
            sampling_rate,
            videos,
            system_message,
            add_system_prompt,
            add_generation_prompt,
            **kwargs,
        )

    @property
    def audio_token_id(self):
        return None

    @property
    def tokenizer(self):
        return self.processor
    
    def _convert_vision_tokens_to_ids(self, content):
        """
        将已经tokenized的视觉token字符串转换为token IDs
        例如: "<|chunk_start|>vid_499vid_20848<|chunk_end|>" -> [token_ids]
        """
        import re
        
        # 如果不包含vision token，直接使用普通tokenize
        if not ('<|chunk_start|>' in content and 'vid_' in content):
            return self.processor.encode(content, add_special_tokens=False)
        
        print(f"[Vision Token Processing] 检测到vision token数据，长度: {len(content)}")
        print(f"[Vision Token Processing] 内容预览: {content[:100]}...")  # 只显示前100个字符
        
        # 处理vision token
        result_ids = []
        
        # 提取特殊标记和vision token
        special_pattern = r'<\|[^|]+\|>'
        vid_pattern = r'vid_\d+'
        
        # 分割内容，保留分隔符
        parts = re.split(r'(<\|[^|]+\|>)', content)
        
        for part in parts:
            if not part:
                continue
                
            if re.match(special_pattern, part):
                # 特殊标记，需要确保在词汇表中
                if part in ['<|chunk_start|>', '<|chunk_end|>']:
                    # 这些应该是新添加的vision token，映射到扩展词汇表
                    if part == '<|chunk_start|>':
                        result_ids.append(32000)  # LLaMA基础词汇表后的第一个token
                    elif part == '<|chunk_end|>':
                        result_ids.append(32001)  # LLaMA基础词汇表后的第二个token
                else:
                    # 其他特殊token，正常encode
                    result_ids.extend(self.processor.encode(part, add_special_tokens=False))
            else:
                # 处理包含vid_xxx的部分
                vid_matches = re.findall(vid_pattern, part)
                if vid_matches:
                    for vid_token in vid_matches:
                        # vid_xxx -> token_id (映射到扩展词汇表)
                        vid_num = int(vid_token.split('_')[1])
                        # 假设vision token从32002开始映射
                        token_id = 32002 + vid_num
                        result_ids.append(token_id)
                else:
                    # 普通文本部分
                    if part.strip():
                        result_ids.extend(self.processor.encode(part, add_special_tokens=False))
        
        # 统计信息
        import re
        vid_count = len(re.findall(r'vid_\d+', content))
        special_count = len(re.findall(r'<\|[^|]+\|>', content))
        print(f"[Vision Token Processing] 转换完成: {len(result_ids)} token IDs")
        print(f"[Vision Token Processing] 包含: {vid_count} 个vision tokens, {special_count} 个特殊标记")
        
        return result_ids

    def get_qwen_template_labels(
        self,
        hf_messages,
        num_image_tokens: List[int],
        num_audio_tokens: List[int],
        num_video_tokens: List[int],
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        # print(f"[Debug] Qwen2DataProcessor.get_qwen_template_labels 被调用！消息数量: {len(hf_messages)}")
        special_tokens = self.processor.additional_special_tokens
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_tokens_idx = [
            self.processor.convert_tokens_to_ids(t) for t in special_tokens
        ]
        input_id, target = [], []
        if add_system_prompt and hf_messages[0]["role"] != "system":
            input_id += self.processor.apply_chat_template(
                [{"role": "system", "content": system_message}],
            )
            target += [-100] * len(input_id)
        for message in hf_messages:
            role = message["role"]
            content = message.get("content", "")
            
            # print(f"[Debug] 处理消息: role={role}, content长度={len(content)}, 内容预览={content[:50]}...")
            
            # 检查是否包含vision token，如果是则特殊处理
            # 注意：转换后的数据格式是 user content为空，assistant content包含vision tokens
            if role == "assistant" and ('<|chunk_start|>' in content or 'vid_' in content):
                # print(f"[VIDTOK DEBUG] ✅ 发现视频token数据!")
                # print(f"[VIDTOK DEBUG] 角色: {role}, 内容长度: {len(content)}")
                # print(f"[VIDTOK DEBUG] 内容预览: {content[:200]}...")
                print(f"[Vision Data Processing] 使用特殊vision token处理逻辑")
                # 对于vision token数据，不使用apply_chat_template，而是手动构建
                # 添加角色开始标记
                role_start_tokens = self.processor.encode("<|im_start|>assistant\n", add_special_tokens=False)
                input_id += role_start_tokens
                target += [-100] * len(role_start_tokens)
                
                # 处理vision token内容
                content_tokens = self._convert_vision_tokens_to_ids(content)
                input_id += content_tokens
                target += content_tokens  # vision token需要学习
                
                # 添加角色结束标记
                role_end_tokens = self.processor.encode("<|im_end|>\n", add_special_tokens=False)
                input_id += role_end_tokens
                target += [-100] * len(role_end_tokens)
            else:
                # 普通文本使用原来的逻辑
                encode_id = self.processor.apply_chat_template([message], tokenize=True)
                input_id += encode_id
                if role in ["user", "system"]:
                    target += [-100] * len(encode_id)
                else:
                    # Adopted from llava-ov that mask out the assistant
                    encode_id[:3] = [-100] * 3
                    target += encode_id

        if add_generation_prompt:
            generation_tokens = self.processor.encode("<|im_start|>assistant\n")
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )
