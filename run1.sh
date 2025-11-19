CUDA_VISIBLE_DEVICES=1

  python train.py config/addition_10digits.py \
    --positional_embedding=RPEBias --max_iters=30000 \
    --blank_space_in_equation_number=500 --digit_test_number=500 \
    --train_data_path='+_maxLen_50_limit_100000_train_minReq_1.txt' \
    --start="FILE:data/val/addition/exclude50/add_samedigit_min50_max500_limit1000_test.txt"  \
    --start_train="FILE:data/val/addition/exclude50/add_samedigit_min1_max50_limit1000_test.txt"  \
    --blank_space_exact=True  --eval_interval=2000 \
    --out_dir='out/addition_50_ABA/run2' \
    --wandb_run_name=run3 --seed=137 --device=cuda:1 --block_size=2024 --time_limit=60000

python eval.py  \
  --out_dir out/addition_50_ABA/run2/run3 \
  --ckpt out/addition_50_ABA/run2/addition_10_best_ood.pt \
  --mode final --operator + --final_file 'data/val/addition/finaltest/multi_digit_test_samelength_10/add_samedigit_min490_max490_limit500_test.txt' --plot

python eval.py  \
  --out_dir out/addition_50_ABA/run3/run3 \
  --ckpt out/addition_50_ABA/run3/addition_10_best_ood.pt \
  --mode length_10 --operator + --digit_test_number 500 --plot
