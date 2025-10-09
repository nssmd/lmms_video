from typing import Dict
import torch
from transformers import DataCollatorForLanguageModeling
from lmms_engine.mapping_func import register_dataset
from .base_dataset import BaseDataset


@register_dataset("text")
class TextDataset(BaseDataset):
    """Text dataset for loading text-only data with support for packing, shuffle, etc."""
    
    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        """Load and process text data from JSON format.
        
        BaseDataset already handles loading, packing, shuffle, etc.
        We just need to process the individual data items.
        """
        # Expect data to have either 'text' field or 'messages' field
        if 'text' in data:
            # Simple text format - 直接使用文本，避免chat template
            text_content = data['text']
            # 直接tokenize，跳过chat template（性能优化）
            model_inputs = self.processor(
                text_content,
                truncation=True,
                padding=False,
                max_length=self.config.packing_length if hasattr(self.config, 'packing_length') else 2048,
                return_tensors="pt"
            )
        elif 'messages' in data:
            # Already in message format - 使用chat template
            messages = data['messages']
            text_data = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            model_inputs = self.processor(
                text_data,
                truncation=True,
                padding=False,
                max_length=self.config.packing_length if hasattr(self.config, 'packing_length') else 2048,
                return_tensors="pt"
            )
        else:
            raise ValueError("Data must contain either 'text' or 'messages' field")
        
        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": model_inputs["input_ids"].squeeze(0).clone()
        }
    
    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        """Load text data from HuggingFace dataset format."""
        return self.load_from_json(data)
    
    def get_collator(self):
        """Get the data collator for causal language modeling."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.processor.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8,
            return_tensors="pt"
        )