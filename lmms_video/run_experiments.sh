#!/bin/bash

# =============================================================================
# 六组实验启动脚本
# 60M/130M/350M模型 × (视觉+文本混合 vs 纯文本) = 6个实验
# =============================================================================

# 基础配置
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1  # 单卡训练
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

# HuggingFace设置
export HF_HOME="/tmp/hf_cache"
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

# Wandb设置
export WANDB_PROJECT="qwen2_multimodal_experiments"

# GPU内存优化设置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 帮助函数
run_experiment() {
    local exp_num=$1
    local config_file=$2
    local description=$3
    
    echo "=========================================="
    echo "开始实验 $exp_num: $description"
    echo "配置文件: $config_file"
    echo "时间: $(date)"
    echo "=========================================="
    
    # 创建临时配置文件 (从主配置文件中提取对应实验)
    local temp_config="temp_exp${exp_num}_config.yaml"
    sed -n "/# ========== 实验${exp_num}:/,/^---$/p" examples/experiment_configs.yaml | sed '/^---$/d' > $temp_config
    
    # 启动训练
    export WANDB_RUN_NAME="exp${exp_num}_$(date +%Y%m%d_%H%M%S)"
    
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m lmms_engine.launch.cli \
        --config $temp_config
    
    local exit_code=$?
    
    # 清理临时文件
    rm -f $temp_config
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 实验 $exp_num 完成成功"
    else
        echo "❌ 实验 $exp_num 失败 (退出码: $exit_code)"
    fi
    
    echo "实验 $exp_num 结束时间: $(date)"
    echo
}

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 [实验编号]"
    echo "实验列表:"
    echo "  1: 60M模型 - 视觉+文本混合数据 (1B tokens)"
    echo "  2: 60M模型 - 纯文本数据 (1B tokens)"
    echo "  3: 130M模型 - 视觉+文本混合数据 (2B tokens)"
    echo "  4: 130M模型 - 纯文本数据 (2B tokens)"
    echo "  5: 350M模型 - 视觉+文本混合数据 (6.4B tokens)"
    echo "  6: 350M模型 - 纯文本数据 (6.4B tokens)"
    echo "  all: 运行所有实验"
    exit 1
fi

case $1 in
    1)
        run_experiment 1 "exp1_config.yaml" "60M模型 - 视觉+文本混合数据 (1B tokens)"
        ;;
    2)
        run_experiment 2 "exp2_config.yaml" "60M模型 - 纯文本数据 (1B tokens)"
        ;;
    3)
        run_experiment 3 "exp3_config.yaml" "130M模型 - 视觉+文本混合数据 (2B tokens)"
        ;;
    4)
        run_experiment 4 "exp4_config.yaml" "130M模型 - 纯文本数据 (2B tokens)"
        ;;
    5)
        run_experiment 5 "exp5_config.yaml" "350M模型 - 视觉+文本混合数据 (6.4B tokens)"
        ;;
    6)
        run_experiment 6 "exp6_config.yaml" "350M模型 - 纯文本数据 (6.4B tokens)"
        ;;
    all)
        echo "开始运行所有6个实验..."
        for i in {1..6}; do
            case $i in
                1) desc="60M模型 - 视觉+文本混合数据 (1B tokens)" ;;
                2) desc="60M模型 - 纯文本数据 (1B tokens)" ;;
                3) desc="130M模型 - 视觉+文本混合数据 (2B tokens)" ;;
                4) desc="130M模型 - 纯文本数据 (2B tokens)" ;;
                5) desc="350M模型 - 视觉+文本混合数据 (6.4B tokens)" ;;
                6) desc="350M模型 - 纯文本数据 (6.4B tokens)" ;;
            esac
            run_experiment $i "exp${i}_config.yaml" "$desc"
            
            # 实验间休息
            if [ $i -lt 6 ]; then
                echo "等待 30 秒后开始下一个实验..."
                sleep 30
            fi
        done
        echo "所有实验完成！"
        ;;
    *)
        echo "错误: 无效的实验编号 '$1'"
        echo "请使用 1-6 或 'all'"
        exit 1
        ;;
esac
