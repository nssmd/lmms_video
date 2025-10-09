#!/usr/bin/env python3
"""
统计 vision token 数量，为 experiment_configs.yaml 提供正确的 vocab_size
不需要下载任何模型，只统计数据
"""

import json
import glob


def extract_vision_tokens_from_data(data_pattern, max_files=100000, max_samples_per_file=1000):
    """
    从 vidtok 数据中提取所有 vision token 并统计数量
    """
    print("开始提取 vision tokens...")
    
    files = glob.glob(data_pattern)
    print(f"找到 {len(files)} 个数据文件")
    
    all_tokens = set()
    special_tokens = set()
    
    for i, file in enumerate(files[:max_files]):
        print(f"处理文件 {i+1}/{min(len(files), max_files)}: {file}")
        
        with open(file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= max_samples_per_file:
                    break
                    
                try:
                    data = json.loads(line)
                    vidtok = data.get('vidtok', '')
                    
                    # 直接从连续字符串中提取 vid_xxx token
                    import re
                    
                    # 提取特殊标记
                    special_pattern = r'<\|[^|]+\|>'
                    special_matches = re.findall(special_pattern, vidtok)
                    for match in special_matches:
                        special_tokens.add(match)
                    
                    # 提取vid_xxx token (数字后面跟着vid_表示下一个token开始)
                    vid_pattern = r'vid_\d+'
                    vid_matches = re.findall(vid_pattern, vidtok)
                    for match in vid_matches:
                        all_tokens.add(match)
                            
                except Exception as e:
                    continue
    
    print(f"提取的特殊 token: {len(special_tokens)} 个")
    print(f"提取的 vision token: {len(all_tokens)} 个")
    
    # 显示一些示例
    if special_tokens:
        print(f"特殊 token 示例: {list(special_tokens)[:5]}")
    if all_tokens:
        print(f"vision token 示例: {list(all_tokens)[:10]}")
    
    total_vision_tokens = len(special_tokens) + len(all_tokens)
    print(f"\n总 vision token 数量: {total_vision_tokens}")
    
    return total_vision_tokens, list(special_tokens), list(all_tokens)


def calculate_vocab_size(base_vocab_size, vision_token_count):
    """
    计算扩展后的词汇表大小
    """
    # 基础词汇表 + vision tokens
    total_vocab_size = base_vocab_size + vision_token_count
    
    # 向上取整到最近的1024的倍数 (通常模型设计习惯)
    padded_vocab_size = ((total_vocab_size + 1023) // 1024) * 1024
    
    print(f"\n词汇表大小计算:")
    print(f"基础词汇表: {base_vocab_size}")
    print(f"Vision tokens: {vision_token_count}")
    print(f"理论总大小: {total_vocab_size}")
    print(f"建议词汇表大小: {padded_vocab_size} (向上取整到1024倍数)")
    
    return padded_vocab_size


def main():
    """
    主函数 - 统计 vision token 并提供配置建议
    """
    # 1. 提取 vision tokens
    data_pattern = "/blob_new/output_laion20M/laion_consolidate/*.jsonl"
    vision_token_count, special_tokens, vision_tokens = extract_vision_tokens_from_data(
        data_pattern, max_files=98, max_samples_per_file=1000  # 处理所有98个文件
    )
    
    # 2. 计算词汇表大小 (假设使用 LLaMA 基础词汇表)
    base_vocab_size = 32000  # LLaMA tokenizer 默认大小
    suggested_vocab_size = calculate_vocab_size(base_vocab_size, vision_token_count)
    
    # 3. 输出配置建议
    print(f"\n" + "="*60)
    print("配置建议:")
    print("="*60)
    print(f"在 experiment_configs.yaml 中设置:")
    print(f"  vocab_size: {suggested_vocab_size}")
    print(f"  processor_name: 'meta-llama/Llama-2-7b-hf' (或其他兼容的tokenizer)")
    print()
    print("这样 lmms-engine-mini 框架会:")
    print("1. 使用指定的 tokenizer")
    print("2. 创建对应大小的 embedding 层")
    print("3. 自动处理 vision token 映射")
    print("="*60)


if __name__ == "__main__":
    main()
