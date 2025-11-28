#!/bin/bash

CUDA_VISIBLE_DEVICES=1

for k in $(seq 1 1 3)
do

  python train.py config/addition_10digits.py \
    --positional_embedding=RPEBias --max_iters=10000 \
    --blank_space_in_equation_number=202 --digit_test_number=200 \
    --blank_space_exact=True \
    --out_dir="out/parity/5_20_200/run${k}" \
    --dataset=newtask \
    --train_data_path="parity/m5_50_train.txt" \
    --start='FILE:data/newtask/parity/m5_200_test.txt' \
    --start_train='FILE:data/newtask/parity/m5_50_test.txt' \
    --operator=copy \
    --seed=${k}

done