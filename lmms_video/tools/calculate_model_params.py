#!/usr/bin/env python3
"""
直接统计自定义 Qwen2.5-VL 模型的实际参数数量
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_parameters(model):
    """统计模型的参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    return total_params / 1e6

def create_qwen2_model(config):
    """
    根据配置创建 Qwen2 模型并统计参数
    
    Args:
        config: 模型配置字典
    
    Returns:
        总参数数量 (单位: M)
    """
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        
        print(f"=== 模型配置 ===")
        print(f"模型类型: Qwen2 (文本模型)")
        print(f"vocab_size: {config['vocab_size']}")
        print(f"hidden_size: {config['hidden_size']}, layers: {config['num_hidden_layers']}")
        print()
        
        # 创建配置
        model_config = Qwen2Config(**config)
        
        # 创建模型 (不加载预训练权重)
        print("正在创建模型...")
        model = Qwen2ForCausalLM(model_config)
        
        print("模型创建成功，开始统计参数...")
        
        # 分模块统计参数
        print(f"\n=== 分模块参数统计 ===")
        
        # 1. 词嵌入层
        if hasattr(model.model, 'embed_tokens'):
            embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
            print(f"词嵌入层: {embed_params:,} ({embed_params/1e6:.2f}M)")
        
        # 2. Transformer层
        if hasattr(model.model, 'layers'):
            layer_params = sum(p.numel() for p in model.model.layers.parameters())
            print(f"Transformer层: {layer_params:,} ({layer_params/1e6:.2f}M)")
        
        # 3. 输出层
        if hasattr(model, 'lm_head'):
            lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
            print(f"输出层(lm_head): {lm_head_params:,} ({lm_head_params/1e6:.2f}M)")
        
        # 4. 其他模块
        other_params = 0
        for name, module in model.named_children():
            if name not in ['model', 'lm_head']:
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    print(f"{name}: {module_params:,} ({module_params/1e6:.2f}M)")
                    other_params += module_params
        
        print(f"\n=== 总参数统计 ===")
        total_params = count_parameters(model)
        
        return total_params
        


# 更新后的三种配置 (使用LLaMA tokenizer的Qwen2模型)
configs = {
    "60M": {
        'vocab_size': 32000,  # LLaMA tokenizer
        'max_position_embeddings': 4096,
        'rope_theta': 10000.0,
        'hidden_size': 512,
        'intermediate_size': 1376,
        'num_hidden_layers': 8,
        'num_attention_heads': 8,
        'num_key_value_heads': 8,
        'tie_word_embeddings': False,
        'use_cache': True,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
        'initializer_range': 0.02
    },
    "130M": {
        'vocab_size': 32000,  # LLaMA tokenizer
        'max_position_embeddings': 4096,
        'rope_theta': 10000.0,
        'hidden_size': 768,
        'intermediate_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'num_key_value_heads': 2,  # GQA
        'tie_word_embeddings': False,
        'use_cache': True,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
        'initializer_range': 0.02
    },
    "350M": {
        'vocab_size': 32000,  # LLaMA tokenizer
        'max_position_embeddings': 4096,
        'rope_theta': 10000.0,
        'hidden_size': 1024,
        'intermediate_size': 2736,
        'num_hidden_layers': 24,
        'num_attention_heads': 16,
        'num_key_value_heads': 2,  # GQA
        'tie_word_embeddings': False,
        'use_cache': True,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
        'initializer_range': 0.02
    }
}

if __name__ == "__main__":
    print("统计 Qwen2 模型参数 (使用LLaMA tokenizer)")
    print("首先尝试创建真实模型，如果失败则使用简化模型估算")
    print()
    
    for model_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"统计 {model_name} 模型参数")
        print(f"{'='*60}")
        
        # 首先尝试创建真实模型
        total_params = create_qwen2_model(config)
        
        if total_params is not None:
            print(f"\n🎯 {model_name} 模型总参数: {total_params:.2f}M")
        else:
            print(f"\n❌ {model_name} 模型参数统计失败")
        print()
