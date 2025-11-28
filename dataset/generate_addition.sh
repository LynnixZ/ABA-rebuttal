#!/usr/bin/env bash
# dataset/generate_addition.sh
python dataset/create_data_addition_carry_train.py \
  --dir_name data/carry \
  --op + \
  --max_digit_len 10 \
  --min_required_digit_len 1 \
  --limit 100000 \
  --carry_chain_min 0 \
  --carry_chain_max 3

#   python dataset/create_data_addition_carry.py \
#   --dir_name data/carry \
#   --op + \
#   --max_digit_len 20 \
#   --min_required_digit_len 1 \
#   --limit 50 \
#   --carry_chain_min 1 \
#   --carry_chain_max 20

  for i in $(seq 1 1 20); do
python dataset/create_data_addition_carry_test.py \
  --mode len2_uniform \
  --n ${i} \
  --limit 1000 \
  --out_dir data/carry/test \
  --seed 42

    echo "Generating test data for digit length = $i"

done