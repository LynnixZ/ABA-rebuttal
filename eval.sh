

for i in {1..10}; do
python dataset/create_multi_add_fullcarry.py \
  --num_samples 1000 \
  --out_path data/newtask/eval/multi_add/force_carry_full/ABA/data/newtask/eval/multi_add/force_carry/test_add_5_${i}.txt \
  --min_operands ${i} \
  --max_operands ${i} \
  --max_digit_len 5 \
  --digit_choices 789 \
  --seed 42
done

# python eval.py  \
#   --out_dir out/addition_10/run3/run3 \
#   --ckpt out/addition_10/run3/addition_10_best_ood.pt \
#   --mode length --operator + --digit_test_number 20 --plot \
#   --test_fullcarry


for k in $(seq 2 1 3)
do
python eval.py  \
  --out_dir out/multiadd/run${k}/addition_10 \
  --ckpt out/multiadd/run${k}/monet_6layer1mixing1poly_5000iter_final.pt \
  --mode length --operator multi_add --digit_test_number 10 --plot \

done


# for i in {1..3}; do
# for j in {30,40,50}; do

# python eval.py  \
#   --out_dir out/addition_10_ABA-var_finetune/${j}/run${i}/addition_10 \
#   --ckpt out/addition_10_ABA-var_finetune/${j}/run${i}/addition_10_best_ood.pt \
#   --mode length --operator + --digit_test_number ${j} --plot \

# done 
# done