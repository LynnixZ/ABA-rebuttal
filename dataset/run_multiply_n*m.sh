#!/usr/bin/env bash
set -e  # 如果任意命令出错，就停止脚本（可以帮助你发现错误）
  # 让脚本在执行时把每条命令打印出来，方便调试
#chmod +x dataset/run_multiply_n*m.sh && ./dataset/run_multiply_n*m.sh
#./dataset/run_multiply_n*m.sh
############################
# 1) 生成大训练集 (10M) 
############################



python dataset/create_data_parity_copy.py \
  --operator multiply \
  --min_digit_length 0 \
  --max_digit_length 20 \
  --limit 10000000 \
  --multi_digit \
  --operand2 99 \
  --output_file data/bal/newtask/multiply/2digits/train/multiply_2digits_train_20_10M.txt

# python dataset/create_data_parity_copy.py \
#   --operator multiply \
#   --min_digit_length 0 \
#   --max_digit_length 20 \
#   --limit 10000000 \
#   --multi_digit \
#   --operand2 99 \
#   --output_file data/bal/newtask/multiply/2digits/train/multiply_2digits_train_10_10M.txt

#############################################
#2) 生成测试集，每隔10递减 k=80..10
#############################################
for ((k=20; k>=1; k=k-10)); do
  max_digit_length=$k
  min_digit_length=1
  if (( min_digit_length < 0 )); then
    min_digit_length=0
  fi

  # 创建输出目录
  mkdir -p data/bal/newtask/multiply/2digits/finaltest/

  python dataset/create_data_parity_copy.py \
    --operator multiply \
    --min_digit_length $min_digit_length \
    --max_digit_length $max_digit_length \
    --limit 10000 \
    --multi_digit \
    --operand2 99 \
    --test \
    --output_file data/bal/newtask/multiply/2digits/finaltest/multiply_2digit_test_${max_digit_length}-${min_digit_length}_10000.txt
done

# python dataset/create_data_parity_copy.py \
#   --operator multiply \
#   --min_digit_length 0 \
#   --max_digit_length 20 \
#   --limit 500 \
#   --multi_digit \
#   --operand2 99 \
#   --test \
#   --output_file data/bal/newtask/multiply/2digits/test/multiply_2digit_test_${max_digit_length}-${min_digit_length}_500.txt

# ##############################################
# # 3) max_digit_length = 1..80, 每次 -1 
# ##############################################
# for max_digit_length in {1..80}; do
#   min_digit_length=$((max_digit_length))
#   if (( min_digit_length < 0 )); then
#     min_digit_length=0
#   fi

#   mkdir -p data/bal/newtask/multiply/2digits/eachdigittest/

#   python dataset/create_data_parity_copy.py \
#     --operator multiply \
#     --min_digit_length $min_digit_length \
#     --max_digit_length $max_digit_length \
#     --limit 300 \
#     --multi_digit \
#     --operand2 99 \
#     --test \
#     --output_file data/bal/newtask/multiply/2digits/eachdigittest/multiply_2digit_test_${max_digit_length}_300.txt
# done
echo "All commands finished successfully!"
