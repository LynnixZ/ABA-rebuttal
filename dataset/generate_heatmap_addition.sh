#!/usr/bin/env bash

# 目标输出目录
OUT_DIR="data/val/addition/finaltest/heatmap"

# 创建输出目录（若不存在则自动创建）
mkdir -p "$OUT_DIR"

# 双重循环，i 为第一个操作数位数，j 为第二个操作数位数
for i in {1..100}; do
  for j in {1..100}; do
    
    # 调用 Python 脚本生成数据
    python dataset/create_data_addition_samelength.py \
      --digit1 "$i" \
      --digit2 "$j" \
      --limit 100 \
      --dir "$OUT_DIR"

    # 可在此添加 echo 语句，查看进度
    echo "Generated dataset for digit1=${i}, digit2=${j}"
  done
done
