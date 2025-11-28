#!/usr/bin/env bash

# 如果 create_multi_add.py 不在当前目录，记得修改路径
# 比如：PY_SCRIPT="/path/to/create_multi_add.py"
PY_SCRIPT="./dataset/create_multi_add.py"

# 你也可以在当前目录下创建一个总输出文件夹，如 data/fixed_collections
# 这里我直接在循环里按组合命名子目录
BASE_DIR="data/newtask/eval/multi_add"

# 避免重复时可以先创建基础目录
mkdir -p "${BASE_DIR}"

# 遍历 digit_len = 1..10
for dl in {1..10}; do
  # 遍历 operand_count = 1..10
  for oc in {1..10}; do
    
    # 构造输出目录，如 data/fixed_combinations/dl3_oc5
    OUT_DIR="${BASE_DIR}/"
    mkdir -p "${OUT_DIR}"

    echo "[Info] Generating dataset => digit_len=${dl}, operand_count=${oc}"
    
    # 调用 create_multi_add.py
    python "${PY_SCRIPT}" \
      --dir_name "${OUT_DIR}" \
      --max_digit_len "${dl}" \
      --max_operand_count "${oc}" \
      --seed 42 \
      --special_mode \
      --num_samples 1000 \
      --mode test 

    INTERMEDIATE_FILE="${OUT_DIR}/test_add_${dl}.txt"
    # Use the actual final filename pattern you decided on:
    FINAL_FILE="${OUT_DIR}/multi_add_d${dl}_o${oc}_test.txt" # Or dl${dl}_oc{oc}.txt etc.

    if [[ -f "${INTERMEDIATE_FILE}" ]]; then
      echo "Renaming ${INTERMEDIATE_FILE} to ${FINAL_FILE}"
      mv "${INTERMEDIATE_FILE}" "${FINAL_FILE}"
    else
      echo "[Warning] Intermediate file ${INTERMEDIATE_FILE} not found. Skipping rename for dl=${dl}, oc=${oc}."
    fi
  done
done

echo "[Done] All 100 datasets generated in '${BASE_DIR}'."
for dl in {1..10}
python gen_multiadd.py \
  --mode test \
  --num_samples 10000 \
  --dir_name data/multiadd_force_col \
  --max_digit_len 5 \
  --max_operand_count 10 \
  --enforce_high_digit_window \
  --high_digits 56789 \
  --window_mode min_len \
  --seed 42