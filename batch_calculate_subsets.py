#!/usr/bin/env python3
"""
批量计算多个规模的数据子集
利用累积特性一次性计算 1B, 2B, 6.4B, 10B token的子集
"""

import glob
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import re

def estimate_tokens_from_json(file_path: str, max_samples: int = None) -> Tuple[int, int]:
    """估算JSON/JSONL文件的token数量"""
    try:
        total_tokens = 0
        total_samples = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_samples and line_num >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    total_samples += 1
                    
                    # 估算tokens
                    if "messages" in data:
                        tokens = estimate_openai_format_tokens(data["messages"])
                    elif "text" in data:
                        tokens = len(data["text"].split()) * 1.5
                    elif "vidtok" in data:
                        tokens = estimate_vidtok_tokens(data["vidtok"])
                    else:
                        text_content = str(data)
                        tokens = len(text_content.split()) * 1.5
                    
                    total_tokens += tokens
                    
                except json.JSONDecodeError:
                    continue
                    
        return total_tokens, total_samples
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return 0, 0

def estimate_openai_format_tokens(messages: List[Dict]) -> int:
    """估算OpenAI格式消息的token数"""
    total_tokens = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            # 检查是否是vidtok内容
            if "vid_" in content and "<|chunk_start|>" in content:
                total_tokens += estimate_vidtok_tokens(content)
            else:
                total_tokens += len(content.split()) * 1.5
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    total_tokens += len(item.get("text", "").split()) * 1.5
    return total_tokens

def estimate_vidtok_tokens(vidtok: str) -> int:
    """估算vidtok字符串的token数量"""
    # 计算视频token数量
    vid_tokens = len(re.findall(r'vid_\d+', vidtok))
    
    # 计算文本token数量
    text_content = re.sub(r'vid_\d+|<\|[^|]+\|>', ' ', vidtok)
    text_tokens = len(text_content.split()) * 1.5
    
    return int(vid_tokens + text_tokens)

def calculate_cumulative_subsets(data_pattern: str, target_tokens_list: List[int], data_type: str):
    """
    计算累积子集
    一次遍历文件，计算出所有目标token数对应的文件列表
    """
    print(f"\n处理 {data_type} 数据: {data_pattern}")
    
    # 获取所有文件并按大小排序
    files = glob.glob(data_pattern)
    files.sort(key=lambda x: os.path.getsize(x))
    
    print(f"找到 {len(files)} 个文件")
    
    # 累积计算
    cumulative_tokens = 0
    cumulative_files = []
    subsets = {}
    
    # 初始化所有目标
    sorted_targets = sorted(target_tokens_list)
    current_target_idx = 0
    
    for file_path in tqdm(files, desc=f"计算{data_type}子集"):
        # 估算文件token数
        file_tokens, file_samples = estimate_tokens_from_json(file_path)
        
        # 添加到累积列表
        cumulative_files.append(file_path)
        cumulative_tokens += file_tokens
        
        # 检查是否达到当前目标
        while (current_target_idx < len(sorted_targets) and 
               cumulative_tokens >= sorted_targets[current_target_idx]):
            
            target = sorted_targets[current_target_idx]
            subsets[target] = {
                'files': cumulative_files.copy(),
                'actual_tokens': cumulative_tokens,
                'file_count': len(cumulative_files)
            }
            
            print(f"  达到 {target/1e9:.1f}B tokens: {len(cumulative_files)} 文件, 实际 {cumulative_tokens/1e9:.2f}B tokens")
            current_target_idx += 1
    
    # 处理未达到的目标（使用所有文件）
    while current_target_idx < len(sorted_targets):
        target = sorted_targets[current_target_idx]
        subsets[target] = {
            'files': cumulative_files.copy(),
            'actual_tokens': cumulative_tokens,
            'file_count': len(cumulative_files)
        }
        print(f"  目标 {target/1e9:.1f}B tokens: 使用所有文件 ({len(cumulative_files)}), 实际 {cumulative_tokens/1e9:.2f}B tokens")
        current_target_idx += 1
    
    return subsets

def save_subset_paths(subsets: Dict, data_type: str, output_dir: str = "."):
    """保存子集文件路径"""
    os.makedirs(output_dir, exist_ok=True)
    
    for target_tokens, subset_data in subsets.items():
        size_label = f"{target_tokens/1e9:.1f}B"
        output_file = os.path.join(output_dir, f"{data_type}_{size_label.replace('.', '_')}_subset_paths.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {data_type.upper()} {size_label} tokens subset\n")
            f.write(f"# 实际tokens: {subset_data['actual_tokens']/1e9:.2f}B\n")
            f.write(f"# 文件数量: {subset_data['file_count']}\n\n")
            
            for file_path in subset_data['files']:
                f.write(f"{file_path}\n")
        
        print(f"保存 {data_type} {size_label} 子集路径到: {output_file}")

def main():
    """主函数"""
    print("开始批量计算数据子集...")
    
    # 配置
    vision_pattern = "/blob_new/output_laion20M/converted_openai_format/*.jsonl"
    text_pattern = "/blob/c4/en/*train*.json"
    
    # 目标token数量 (1B, 2B, 6.4B, 10B)
    target_tokens = [
        1_000_000_000,    # 1B
        2_000_000_000,    # 2B
        6_400_000_000,    # 6.4B
        10_000_000_000,   # 10B
    ]
    
    print(f"目标token数量: {[f'{t/1e9:.1f}B' for t in target_tokens]}")
    
    # 计算vision子集
    vision_subsets = calculate_cumulative_subsets(vision_pattern, target_tokens, "vision")
    save_subset_paths(vision_subsets, "vision")
    
    # 计算text子集
    text_subsets = calculate_cumulative_subsets(text_pattern, target_tokens, "text")
    save_subset_paths(text_subsets, "text")
    
    # 输出总结
    print("\n" + "="*60)
    print("批量计算完成！")
    print("="*60)
    
    for target in target_tokens:
        size_label = f"{target/1e9:.1f}B"
        print(f"\n{size_label} tokens:")
        
        if target in vision_subsets:
            v_data = vision_subsets[target]
            print(f"  Vision: {v_data['file_count']} 文件, {v_data['actual_tokens']/1e9:.2f}B tokens")
        
        if target in text_subsets:
            t_data = text_subsets[target]
            print(f"  Text:   {t_data['file_count']} 文件, {t_data['actual_tokens']/1e9:.2f}B tokens")

if __name__ == "__main__":
    main()
