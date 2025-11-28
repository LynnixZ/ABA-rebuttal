#!/bin/bash

# 定义任务列表
# 指定输出目录
output_dir="data/newtask/"
# 每个数据集的样本数
limit=10000

# 创建输出目录
mkdir -p "$"


    python dataset/create_string_sort.py \
      --dir_name "${output_dir}" \
      --min_array_len "2" --max_array_len "5" \
      --min_str_len "1" --max_str_len "5" \
      --test_limit 100000 \
      --max_array_padding_len 0 \
      --mode "train" \



echo "全部数据集生成完成。"
