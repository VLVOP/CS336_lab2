#!/bin/bash

# 创建保存分析文件的目录
mkdir -p profiles

echo "=========================================================="
echo "阶段 1: 运行端到端基准测试 (对应 1.1.3 a, b, c)"
echo "=========================================================="
uv run python cs336_systems/run_full_benchmark.py

echo -e "
"
echo "=========================================================="
echo "阶段 2: 运行 Nsight Systems Profiling (对应 1.1.4)"
echo "注意: 如果显存不足(OOM)，脚本会自动跳过该配置并继续"
echo "=========================================================="

# 定义 1.1.4 要求的所有参数
MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")
CONTEXT_LENGTHS=(128 256 512 1024)

# 为了减小分析文件的大小，我们将 active_steps 设为 2
# warmup_steps 设为 2，确保内核已就绪
for size in "${MODEL_SIZES[@]}"; do
    for len in "${CONTEXT_LENGTHS[@]}"; do
        echo "----------------------------------------------------------"
        echo "正在分析模型: $size | 上下文长度: $len"
        OUTPUT_FILE="profiles/profile_${size}_${len}"
        
        # 使用 nsys 进行 profile
        # --force-overwrite=true: 如果文件存在则覆盖
        # --pytorch=true: 自动标注 PyTorch 操作
        uv run nsys profile -o "$OUTPUT_FILE" --force-overwrite true --pytorch true 
            python cs336_systems/profile_model.py 
            --model_size "$size" 
            --context_length "$len" 
            --active_steps 2 
            --warmup_steps 2
            
        if [ $? -ne 0 ]; then
            echo "警告: $size 模型在长度 $len 下运行失败 (可能显存溢出)，跳过..."
        fi
    done
done

echo -e "
=========================================================="
echo "实验完成！"
echo "1. 请查看 benchmark.log 获取 1.1.3 的数据表格。"
echo "2. 请在 profiles/ 目录下找到 .nsys-rep 文件，并用 Nsight Systems 软件打开以回答 1.1.4 的问题。"
echo "=========================================================="
