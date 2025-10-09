#!/usr/bin/env python3
"""
测试txt格式数据加载功能
"""

import sys
import os
sys.path.append('/home/v-zimowen/lmms-engine-mini/src')

from lmms_engine.utils.data_utils import DataUtilities

def test_txt_loading():
    print("测试txt格式数据加载...")
    
    # 测试1B tokens的文本数据
    txt_path = "/home/v-zimowen/lmms-engine-mini/text_1_0B_subset_paths.txt"
    print(f"\n加载: {txt_path}")
    
    try:
        dataset = DataUtilities.load_subset_from_txt(txt_path)
        print(f"成功加载数据集，共 {len(dataset)} 条样本")
        
        # 显示前几个样本
        print("\n前3个样本:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"样本 {i+1}:")
            if 'text' in sample:
                print(f"  text: {sample['text'][:100]}...")
            elif 'messages' in sample:
                print(f"  messages: {len(sample['messages'])} 条消息")
                print(f"  第一条消息: {str(sample['messages'][0])[:100]}...")
            else:
                print(f"  键: {list(sample.keys())}")
            print()
            
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_txt_loading()
