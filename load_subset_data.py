#!/usr/bin/env python3
"""
从txt文件读取路径列表，然后加载对应的JSON数据
"""

import json
import glob
import os
from typing import List, Dict
from datasets import Dataset

def load_subset_from_txt(txt_file_path: str) -> Dataset:
    """
    从txt文件读取路径列表，加载所有JSON数据并合并为Dataset
    
    Args:
        txt_file_path: 包含文件路径列表的txt文件
        
    Returns:
        Dataset: 合并后的数据集
    """
    print(f"加载子集数据: {txt_file_path}")
    
    # 读取文件路径列表
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤掉注释行，只保留文件路径
    file_paths = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            file_paths.append(line)
    
    print(f"找到 {len(file_paths)} 个文件")
    
    # 加载所有JSON数据
    all_data = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue
            
        print(f"加载文件: {os.path.basename(file_path)}")
        
        try:
            if file_path.endswith('.jsonl'):
                # JSONL格式 - 每行一个JSON对象
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"  警告: 第{line_num}行JSON解析错误: {e}")
                            continue
            else:
                # 普通JSON格式 - 整个文件是一个JSON数组
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                        
        except Exception as e:
            print(f"  错误: 加载文件失败 {file_path}: {e}")
            continue
    
    print(f"总共加载了 {len(all_data)} 条数据")
    
    # 转换为Hugging Face Dataset
    dataset = Dataset.from_list(all_data)
    return dataset

def main():
    """测试函数"""
    # 测试加载1B tokens的text子集
    text_1b_file = "/home/v-zimowen/lmms-engine-mini/text_1_0B_subset_paths.txt"
    if os.path.exists(text_1b_file):
        dataset = load_subset_from_txt(text_1b_file)
        print(f"Text 1B子集大小: {len(dataset)}")
        print(f"第一条数据: {dataset[0]}")
    
    # 测试加载1B tokens的vision子集
    vision_1b_file = "/home/v-zimowen/lmms-engine-mini/vision_1_0B_subset_paths.txt"
    if os.path.exists(vision_1b_file):
        dataset = load_subset_from_txt(vision_1b_file)
        print(f"Vision 1B子集大小: {len(dataset)}")
        print(f"第一条数据: {dataset[0]}")

if __name__ == "__main__":
    main()


