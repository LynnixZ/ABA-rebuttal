python dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 10000 \
  --seq_len_min 1 \
  --seq_len_max 10 \
  --out_path data/newtask/parity_scratchpad/parity_m5_10_train.txt \
  --seed 123

python dataset/create_data_parity_scratchpad.py \
  --m 2 \
  --limit 10000 \
  --seq_len_min 1 \
  --seq_len_max 10 \
  --out_path data/newtask/parity_scratchpad/parity_m2_10_train.txt \
  --seed 123

python dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 1000 \
  --seq_len_min 1 \
  --seq_len_max 20 \
  --out_path data/newtask/parity_scratchpad/parity_m5_20.txt \
  --seed 123

python dataset/create_data_parity_scratchpad.py \
  --m 2 \
  --limit 1000 \
  --seq_len_min 1 \
  --seq_len_max 20 \
  --out_path data/newtask/parity_scratchpad/parity_m2_20.txt \
  --seed 123


python dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 1000 \
  --seq_len_min 1 \
  --seq_len_max 10 \
  --out_path data/newtask/parity_scratchpad/parity_m5_10.txt \
  --seed 123

python dataset/create_data_parity_scratchpad.py \
  --m 2 \
  --limit 1000 \
  --seq_len_min 1 \
  --seq_len_max 10 \
  --out_path data/newtask/parity_scratchpad/parity_m2_10.txt \
  --seed 123
