#!/usr/bin/env python3
"""
批量更新所有txt文件中的路径为Parquet格式
"""

import os
import re

def update_file_paths(file_path: str):
    """更新单个文件中的路径"""
    print(f"Updating {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # # 替换.jsonl为.parquet
    # content = content.replace('.parquet', '.jsonl')
    # # 替换.json为.parquet  
    content = content.replace('.jsonl', '.parquet')
    
    # 更新注释
    if 'vision' in file_path:
        content = re.sub(
            r'# VISION (\d+\.?\d*B) tokens subset',
            r'# VISION \1 tokens subset (Parquet format)',
            content
        )
    elif 'text' in file_path:
        content = re.sub(
            r'# TEXT (\d+\.?\d*B) tokens subset',
            r'# TEXT \1 tokens subset (Parquet format)',
            content
        )
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

def main():
    """主函数"""
    files_to_update = [
        'vision_1_0B_subset_paths.txt',
        'vision_2_0B_subset_paths.txt',
        'vision_6_4B_subset_paths.txt', 
        'vision_10_0B_subset_paths.txt',
        'text_1_0B_subset_paths.txt',
        'text_2_0B_subset_paths.txt',
        'text_6_4B_subset_paths.txt',
        'text_10_0B_subset_paths.txt'
    ]
    
    for file_name in files_to_update:
        if os.path.exists(file_name):
            update_file_paths(file_name)
        else:
            print(f"File {file_name} not found, skipping")
    
    print("All files updated!")

if __name__ == "__main__":
    main()
