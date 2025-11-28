#!/bin/bash

# 定义任务列表
operators=("parity" "binarysum" "copy" "reverse" "oneDigitSort" "multiply" "hexadecimal")
# 指定输出目录
output_dir="data/newtask"
# 每个数据集的样本数
limit=500

# 创建输出目录
mkdir -p "$output_dir"
#evaluation 
# 对于每个任务，对长度从 1 到 100 生成数据集
for op in "${operators[@]}"; do
    echo "生成任务：$op 的数据集（长度 1 到 100）..."
    mkdir -p "$output_dir/$op"
    for length in $(seq 101 150); do
        output_file="${output_dir}/$op/${op}_${length}.txt"
        echo "生成长度为 ${length} 的数据集，保存到 ${output_file}"
        if [ "$op" == "multiply" ]; then
            # 对于 multiply 操作，设置 operand2 参数为 2（可根据需要调整）
            python dataset/create_data_parity_copy.py --operator "$op" --min_digit_length "$length" --max_digit_length "$length" --limit "$limit" --operand2 99 --output_file "$output_file" --multi_digit
        else
            python dataset/create_data_parity_copy.py --operator "$op" --min_digit_length "$length" --max_digit_length "$length" --limit "$limit" --output_file "$output_file"
        fi
    done
done

echo "所有固定长度的数据集生成完成。"
