#!/usr/bin/env bash

# 假设你有一个脚本叫 generate_samedigit_add.py
# 它的用法（简化）大致如下：
# python generate_samedigit_add.py --min MINLEN --max MAXLEN --limit LIMIT --dir OUTDIR --test
OUT_DIR="data/val/addition/test_exclude10/samelength/medium"   # 目标输出目录，可根据需求修改
LIMIT=2000             # 每个长度生成的算式数量

mkdir -p "${OUT_DIR}"

# 从 1 遍历到 150
for i in $(seq 300 10 400); do
    echo "Generating test data for digit length = $i"
    python dataset/create_data_addition_samelength_train.py \
        --min 10 \
        --max $i \
        --limit ${LIMIT} \
        --dir "${OUT_DIR}" \
        --test
done

    python dataset/create_data_addition_samelength_train.py \
        --min 30 \
        --max 30 \
        --limit 1000000 \
        --dir "data/bal" \

echo "全部长度(1~150)的测试集已生成到目录: ${OUT_DIR}"

#!/usr/bin/env bash

# 你希望的 multi_digit 数
multi_digit=300

# 你希望的 limit
limit=500


for k in 1 $(seq 10 10 "${multi_digit}")
do
  # 注意：当 k=1 时，min_required_digit_len = 0
  min_req=$((k-1))

  # 目录名可以自行改动，这里演示区分成 train_k
  dir_name="data/val/addition/finaltest/multi_digit_test_samelength_10"

  echo "Generating dataset: max_digit_len=${k}, min_required_digit_len=${min_req}"=
     python dataset/create_data_addition_samelength.py \
        --min ${k} \
        --max ${k} \
        --limit 500 \
        --test \
        --dir ${dir_name}       
done

