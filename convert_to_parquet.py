#!/usr/bin/env python3
"""
将JSONL和JSON文件转换为Parquet格式，提高数据加载性能
"""

import json
import jsonlines
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import argparse
from tqdm import tqdm
import os

def convert_jsonl_to_parquet(input_path: str, output_path: str, batch_size: int = 10000):
    """将JSONL文件转换为Parquet格式"""
    print(f"Converting {input_path} to {output_path}")
    
    # 读取JSONL文件
    data_list = []
    with jsonlines.open(input_path, "r") as f:
        for data in f:
            data_list.append(data)
    
    print(f"Loaded {len(data_list)} records")
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list)
    
    # 保存为Parquet
    df.to_parquet(output_path, index=False, compression='snappy')
    print(f"Saved to {output_path}")

def convert_json_to_parquet(input_path: str, output_path: str):
    """将JSON文件转换为Parquet格式（支持标准JSON和JSONL格式）"""
    print(f"Converting {input_path} to {output_path}")
    
    data_list = []
    
    try:
        # 首先尝试作为标准JSON文件读取
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            data_list = data
        else:
            data_list = [data]
            
    except json.JSONDecodeError:
        # 如果标准JSON解析失败，尝试作为JSONL格式读取
        print(f"  Standard JSON failed, trying JSONL format...")
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        print(f"  Line {line_num} JSON parse error: {e}")
                        continue
    
    if not data_list:
        print(f"  No valid data found in {input_path}")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list)
    print(f"Loaded {len(df)} records")
    
    # 保存为Parquet
    df.to_parquet(output_path, index=False, compression='snappy')
    print(f"Saved to {output_path}")

def convert_file_list_to_parquet(file_list_path: str, output_dir: str, data_type: str = "jsonl", skip_existing: bool = True):
    """转换文件列表中的所有文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(file_list_path, 'r') as f:
        lines = f.readlines()
    
    # 过滤掉注释行和空行
    file_paths = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    print(f"Found {len(file_paths)} files to convert")
    
    for i, file_path in enumerate(tqdm(file_paths, desc="Converting files")):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping")
            continue
            
        # 生成输出文件名 - 直接替换后缀
        if file_path.endswith('.jsonl'):
            output_file = file_path.replace('.jsonl', '.parquet')
        elif file_path.endswith('.json'):
            output_file = file_path.replace('.json', '.parquet')
        else:
            print(f"Unknown file extension: {file_path}")
            continue
        
        # 检查输出文件是否已存在
        if skip_existing and os.path.exists(output_file):
            print(f"Skipping {Path(file_path).name} (already exists)")
            continue
        
        try:
            if data_type == "jsonl":
                convert_jsonl_to_parquet(file_path, output_file)
            elif data_type == "json":
                convert_json_to_parquet(file_path, output_file)
            else:
                print(f"Unknown data type: {data_type}")
                continue
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Convert JSON/JSONL files to Parquet format")
    parser.add_argument("--vision_list", type=str, 
                       default="/home/v-zimowen/lmms-engine-mini/vision_1_0B_subset_paths.txt",
                       help="Path to vision file list")
    parser.add_argument("--text_list", type=str,
                       default="/home/v-zimowen/lmms-engine-mini/text_1_0B_subset_paths.txt", 
                       help="Path to text file list")
    
    args = parser.parse_args()
    
    print("Converting vision files...")
    convert_file_list_to_parquet(args.vision_list, "", "jsonl")
    
    print("\nConverting text files...")
    convert_file_list_to_parquet(args.text_list, "", "json")
    
    print(f"\nConversion complete! Parquet files saved to same locations as original files")

if __name__ == "__main__":
    main()
