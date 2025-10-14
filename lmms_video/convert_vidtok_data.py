#!/usr/bin/env python3
"""
将 vidtok 数据转换为 lmms-engine-mini 期望的 OpenAI 对话格式
不改变 vidtok 内容，只是包装成对话格式
"""

import json
import glob
import os
from tqdm import tqdm

def convert_vidtok_to_openai_format(input_pattern, output_dir):
    """
    将 vidtok 数据转换为 OpenAI 对话格式
    
    输入格式:
    {
        "instruction": "描述文本",
        "vidtok": "<|chunk_start|>vid_499vid_20848...<|chunk_end|>",
        "video_shapes": [...],
        ...
    }
    
    输出格式:
    {
        "messages": [
            {"role": "user", "content": "描述文本"},
            {"role": "assistant", "content": "<|chunk_start|>vid_499vid_20848...<|chunk_end|>"}
        ]
    }
    """
    
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob.glob(input_pattern)
    
    print(f"找到 {len(input_files)} 个输入文件")
    
    total_converted = 0
    
    for input_file in tqdm(input_files, desc="转换文件"):
        # 生成输出文件名
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"converted_{base_name}")
        
        converted_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 检查必要字段
                    if 'instruction' not in data or 'vidtok' not in data:
                        print(f"警告: 第{line_num}行缺少必要字段，跳过")
                        continue
                    
                    # 转换为 OpenAI 格式 - vision token 自回归训练
                    # 训练目标: 从 vision token 预测下一个 vision token
                    converted = {
                        "messages": [
                            {
                                "role": "user", 
                                "content": ""  # 空输入，纯 vision token 自回归
                            },
                            {
                                "role": "assistant", 
                                "content": data["vidtok"]  # 模型学习生成整个 vision token 序列
                            }
                        ]
                    }
                    
                    # 保留原始元数据（可选）
                    if "video_shapes" in data:
                        converted["video_shapes"] = data["video_shapes"]
                    if "video_id" in data:
                        converted["video_id"] = data["video_id"]
                    
                    f_out.write(json.dumps(converted, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 第{line_num}行处理错误: {e}")
                    continue
        
        print(f"文件 {base_name}: 转换了 {converted_count} 条记录")
        total_converted += converted_count
    
    print(f"\n转换完成！总共转换了 {total_converted} 条记录")
    print(f"输出目录: {output_dir}")

def main():
    # 配置
    input_pattern = "/blob_new/output_laion20M/laion_consolidate/*.jsonl"
    output_dir = "/blob_new/output_laion20M/converted_openai_format"
    
    print("开始转换 vidtok 数据为 OpenAI 格式...")
    print(f"输入模式: {input_pattern}")
    print(f"输出目录: {output_dir}")
    print()
    
    convert_vidtok_to_openai_format(input_pattern, output_dir)

if __name__ == "__main__":
    main()
