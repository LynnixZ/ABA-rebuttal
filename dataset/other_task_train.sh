#!/bin/bash

# 定义任务列表
operators=( "copy" "reverse" "multiply" )
# 指定输出目录
output_dir="data/newtask/train/not_weighted/"
# 每个数据集的样本数
limit=10000

# 创建输出目录
mkdir -p "$output_dir"
#for max_digit_length in $(seq 10 10 100); do
for op in "${operators[@]}"; do
    echo "生成任务：$op 的数据集（长度 1 到 100）..."

        python dataset/create_data_parity_copy.py --operator "$op" --min_digit_length 1 --max_digit_length 10 --limit "100000" --output_file "$output_dir/${op}_test_10.txt" --operand2 99 --multi_digit --test

#done
done



echo "全部数据集生成完成。"
