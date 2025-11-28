#!/bin/bash

# --- 配置 ---

# 1. 定义要为其生成训练数据的任务列表
operators=("string_reverse" "char_shift")

# 2. 指定 Python 脚本的名称
python_script="dataset/create_string_dataset.py"

# 3. 指定训练数据集的根输出目录
output_dir="data/newtask"

# 4. 为每个任务的训练集设置总样本数
limit=1000

# 5. 设置训练数据中字符串的最小和最大长度
min_len=11
max_len=50


# --- 脚本执行 ---

# 创建根输出目录，如果它不存在的话
mkdir -p "$output_dir"

echo "开始生成大型训练数据集..."
echo "========================================="

# 遍历 operators 数组中的每个任务
for op in "${operators[@]}"; do
    echo
    echo "--> 正在为任务 '$op' 生成训练集..."
    
    # 定义最终的输出文件路径，例如: data/string_training/string_reverse_train.txt
    output_file="${output_dir}/${op}_test_50.txt"
    
    echo "  样本总数: $limit"
    echo "  字符串长度范围: [${min_len}-${max_len}]"
    echo "  输出文件: ${output_file}"
    
    # 调用 Python 脚本生成数据
    # - 我们没有设置 --is_test 参数，因此脚本会使用默认的“递增权重”
    #   来采样长度，这对训练模型更有利。
    python "$python_script" \
        --operator "$op" \
        --limit "$limit" \
        --min_string_length "$min_len" \
        --max_string_length "$max_len" \
        --output_file "$output_file" --is_test
done

echo
echo "========================================="
echo "所有训练数据集生成完成。"
echo "文件保存在目录: $output_dir"