#!/bin/bash

# 创建主输出文件夹
OUTPUT_ROOT="data/newtask/eval/stringsort"
mkdir -p "${OUTPUT_ROOT}"

# 遍历数组长度和数字位数（均从 1 到 10）
for array_len in {2..10}; do
  for digit_len in {1..10}; do
    # 为当前组合创建一个输出文件夹
    output_dir="${OUTPUT_ROOT}/"
    mkdir -p "${output_dir}"
    
    # 调用 Python 脚本，生成数据集
    # 这里将最小和最大数组长度均设为 array_len，
    # 同时将最小和最大数字位数均设为 digit_len，
    # 使用 --train_limit 设定每个数据集生成的数据条数（例如这里设为 100）
    python dataset/create_string_sort.py \
      --dir_name "${output_dir}" \
      --min_array_len "${array_len}" --max_array_len "${array_len}" \
      --min_str_len 1 --max_str_len "${digit_len}" \
      --test_limit 500 \
      --mode test \
     # --special_mode
    echo "生成数据集：数组长度 ${array_len}，数字位数 ${digit_len}"
  done
done
