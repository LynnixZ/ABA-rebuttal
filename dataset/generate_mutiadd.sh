python gen_high_carry_multiadd.py \
  --out_path data/high_carry_multiadd/test_highcarry_6to10op_1to5digits_789.txt \
  --num_samples 10000 \
  --min_operands 6 \
  --max_operands 10 \
  --max_digit_len 5 \
  --digit_choices 789 \
  --seed 42

python dataset/create_multi_add.py \
--mode test --num_samples 100000 \
    --dir_name data/bal/train/not_weighted \
    --max_digit_len 5 \
    --max_operand_count 5 \


# 5) 纯训练（不生成测试集）
python dataset/create_multi_add.py --mode test --num_samples 1000 \
    --dir_name data/bal/multi_add \
    --max_digit_len 5 \
    --max_operand_count 5 \

python dataset/create_multi_add.py \
    --mode test --num_samples 1000 \
    --dir_name data/bal/multi_add \
    --max_digit_len 10 \
    --max_operand_count 10 \

python dataset/create_multi_add.py \
    --mode test --num_samples 10000 \
    --dir_name data/newtask/final_test/multiadd \
    --max_digit_len 10 \
    --max_operand_count 10 \