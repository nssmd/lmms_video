"""
Simple script to occupy GPU memory.
用于占用GPU显存的简单脚本
"""

import torch
import time
import argparse


def occupy_gpu(gpu_ids=None, memory_fraction=0.9):
    """
    占用指定GPU的显存
    
    Args:
        gpu_ids: GPU ID列表，如 [0, 1, 2]。如果为None则占用所有可用GPU
        memory_fraction: 占用显存的比例，默认0.9（90%）
    """
    if gpu_ids is None:
        # 占用所有可用GPU
        gpu_ids = list(range(torch.cuda.device_count()))
    
    if not gpu_ids:
        print("No GPU available!")
        return
    
    print(f"Occupying GPU(s): {gpu_ids}")
    print(f"Memory fraction: {memory_fraction * 100}%")
    
    tensors = []
    
    for gpu_id in gpu_ids:
        device = torch.device(f'cuda:{gpu_id}')
        
        # 获取GPU总显存
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        target_memory = int(total_memory * memory_fraction)
        
        # 分配显存 (以GB为单位显示)
        print(f"\nGPU {gpu_id}:")
        print(f"  Total memory: {total_memory / 1e9:.2f} GB")
        print(f"  Occupying: {target_memory / 1e9:.2f} GB")
        
        # 创建大tensor占用显存
        # 每个float32占4字节
        num_elements = target_memory // 4
        tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
        tensors.append(tensor)
        
        print(f"  Successfully occupied GPU {gpu_id}")
    
    print("\n✓ GPU occupation complete!")
    print("Press Ctrl+C to release and exit...")
    
    try:
        # 保持运行
        while True:
            time.sleep(60)
            # 定期打印状态
            for i, gpu_id in enumerate(gpu_ids):
                allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                print(f"GPU {gpu_id}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    except KeyboardInterrupt:
        print("\n\nReleasing GPU resources...")
        del tensors
        torch.cuda.empty_cache()
        print("✓ GPU resources released. Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupy GPU memory")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to occupy, e.g., '0,1,2'. Default: all GPUs"
    )
    parser.add_argument(
        "--memory",
        type=float,
        default=0.9,
        help="Fraction of memory to occupy (0.0-1.0). Default: 0.9"
    )
    
    args = parser.parse_args()
    
    # 解析GPU IDs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        gpu_ids = None
    
    occupy_gpu(gpu_ids=gpu_ids, memory_fraction=args.memory)

