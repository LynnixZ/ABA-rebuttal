
python dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 200 \
  --seq_len_min 1 \
  --seq_len_max 20 \
  --out_path data/parity/m5_20_test.txt \
  --seed 42

python dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 1000 \
  --seq_len_min 1 \
  --seq_len_max 200 \
  --out_path data/parity/m5_200_test.txt \
  --seed 42

  python  dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 10000 \
  --seq_len_min 1 \
  --seq_len_max 50 \
  --out_path data/parity/m5_50_train.txt \
  --seed 42

  python  dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 200 \
  --seq_len_min 1 \
  --seq_len_max 50 \
  --out_path data/parity/m5_50_test.txt \
  --seed 42
