#!/bin/bash

# --- 配置 ---

# 1. 定义要为其生成数据的任务列表
operators=("string_reverse" "char_shift")

# 2. 指定 Python 脚本的名称
python_script="dataset/create_string_dataset.py"
# 3. 指定所有数据集的根输出目录
output_dir="data/newtask/eval/"

# 4. 为每个生成的 .txt 文件设置样本数
limit=500

# --- 脚本执行 ---

# 创建根输出目录，如果它不存在的话
mkdir -p "$output_dir"

echo "开始为固定长度生成评估数据集..."
echo "========================================="

# 遍历 operators 数组中的每个任务
for op in "${operators[@]}"; do
    echo
    echo "--> 正在处理任务: $op"
    
    # 为当前任务创建一个子目录
    mkdir -p "$output_dir/$op"
    
    # 对于每个任务，对长度从 1 到 100 生成数据集
    for length in $(seq 1 100); do
        # 定义最终的输出文件路径，例如: data/string_evaluation/string_reverse/string_reverse_1.txt
        output_file="${output_dir}/$op/${op}_${length}.txt"
        
        echo "  生成长度为 ${length} 的数据集，保存到 ${output_file}"
        
        # 调用 Python 脚本生成数据
        # --min_string_length 和 --max_string_length 都设置为当前的循环变量 "length"
        # 以确保文件中的所有字符串都有完全相同的长度。
        python "$python_script" \
            --operator "$op" \
            --limit "$limit" \
            --min_string_length "$length" \
            --max_string_length "$length" \
            --output_file "$output_file"
    done
done

echo
echo "========================================="
echo "所有固定长度的数据集生成完成。"
echo "文件保存在目录: $output_dir"