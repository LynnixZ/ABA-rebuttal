# Step 1: Include your code to generate train_data and val_data
import os
import time
import math
import pickle
import pandas as pd
import yaml

from contextlib import nullcontext
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from monet_masking_no_embedding_modeltype import GPTConfig, GPT
from prepare_data import get_data_list, generate_data_str
from main_utils import create_meta_file 

max_dollar_pad = 5  # Set your desired maximum number of $ signs
one_dollar_probability = 0.5  # Probability that only one $ is inserted before the answer
train_data_path="data/bal/hardtask/samelength/add_samedigit_min1_max10_limit1000_train.txt"
#data/bal/+_n_5_m_5_examples_100.txt
#data/bal/multi_add/test_add_10.txt
#data/bal/hardtask/samelength/add_samedigit_min1_max10_limit1000_train.txt
operator = '+'

arithmetic_batch=True

#data_dir = os.path.join('dataset')
#train_data_path = os.path.join(data_dir, train_data_path)
# val_data = os.path.join(data_dir, val_data_path)
val_data_list = get_data_list(train_data_path, operator=operator)  # get_data_list(val_data, operator='+')

train_data_list = get_data_list(train_data_path, operator=operator)
train_data_str = generate_data_str(
    train_data_list, operator=operator, format='reverse', train=True, shuffle=False,
     reverse_ab=True, index_hint=False, zeropad_max_length=0,
    reverse_c=True,  blank_space_in_equation_number=21, pad_answer=True, 
    fix_blank_space_position=True, blank_space_number_exact=True, blank_space_split_number=True,
zero_pad=True,max_operand_number=0, operand_number_exact=True, pad_before=False) # New parameter)   
val_data_str = generate_data_str(
    train_data_list, operator=operator, format='reverse', train=False, shuffle=False, 
    reverse_ab=True, index_hint=False, zeropad_max_length=11,
    reverse_c=True,  blank_space_in_equation_number=21, pad_answer=True, 
    fix_blank_space_position=True, blank_space_number_exact=True, blank_space_split_number=True,
    equal_bls_in3opr=False, zero_pad=True,max_operand_number=0, operand_number_exact=True,pad_before=False) # New parameter)   
meta, meta_path, data_encoder, data_decoder = create_meta_file(
    vocabulary='all_ascii_chars' , input_data_str=train_data_str, tokenizer= 'char' 
)
meta_vocab_size = meta['vocab_size']
train_data = data_encoder(train_data_str)
val_data = data_encoder(train_data_str)

# Step 1.1: Define the token IDs for '=', '$', and '0'
equal_token = "="
equal_token_id = data_encoder(equal_token)
if isinstance(equal_token_id, list):
    equal_token_id = equal_token_id[0]
else:
    equal_token_id = int(equal_token_id.item())

dollar_token = "$"
dollar_token_id = data_encoder(dollar_token)
if isinstance(dollar_token_id, list):
    dollar_token_id = dollar_token_id[0]
else:
    dollar_token_id = int(dollar_token_id.item())



star_token = "*"
star_token_id = data_encoder(star_token)
if isinstance(star_token_id, list):
    star_token_id = star_token_id[0]
else:
    star_token_id = int(star_token_id.item())

pad_token = "."
pad_token_id = data_encoder(pad_token)
if isinstance(pad_token_id, list):
    pad_token_id = pad_token_id[0]
else:
    pad_token_id = int(pad_token_id.item())

zero_token = '0'
zero_token_id = data_encoder(zero_token)
if isinstance(zero_token_id, list):
    zero_token_id = zero_token_id[0]
else:
    zero_token_id = int(zero_token_id.item())

newline_token = '\n'
newline_token_id = data_encoder(newline_token)
if isinstance(newline_token_id, list):
    newline_token_id = newline_token_id[0]
else:
    newline_token_id = int(newline_token_id.item())



batch_size = 5
block_size = 1024
train_both = False 
zero_pad_in_target = False

# Step 2: Modify the get_batch function
# Define the parameters
dollar_pad_no_loss = False  # Set to True if you want to unmask the last '$' before the answer and mask all the other '$' signs before the answer,
# or Set to False if you want to unmask all '$' signs before the answer
zero_pad_in_target = False  # Set to True if you want masked positions to be '0'
no_loss_everywhere= False # Set to True if you want to unmask all positions expect the answer or False if you want to unmask only the last $ before the answer and the answer

train_arithmetic_tokenized = []
if arithmetic_batch:

    arithmetic_lines_str = train_data_str.split('\n')[:-1]
    for line in arithmetic_lines_str:
    # 只做一次 token 转换
        tokens = data_encoder(line+'.')  # 不做额外拼接 '.', 也不去掉 '$'
        # 存成一个 torch.Tensor 或者 numpy
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        train_arithmetic_tokenized.append(tokens_tensor)

val_arithmetic_tokenized = []
if arithmetic_batch:
    arithmetic_lines_str = train_data_str.strip().split('\n') 
    for line in arithmetic_lines_str:
    # 只做一次 token 转换
        tokens = data_encoder(line+'.')  # 不做额外拼接 '.', 也不去掉 '$'
        # 存成一个 torch.Tensor 或者 numpy
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        val_arithmetic_tokenized.append(tokens_tensor)

split_pointers = {
    'train': 0,
    'val': 0
}

def get_batch(split,arithmetic_batch):

    # Use the appropriate global pointer
    global split_pointers

    if arithmetic_batch:
        data_tokenized = train_arithmetic_tokenized if split == 'train' else val_arithmetic_tokenized
        
        # 1. Get the starting index for the current batch
        start_ix = split_pointers[split]
        end_ix = start_ix + batch_size

        # 2. Check if we've reached the end of the dataset
        if end_ix > len(data_tokenized):
            # If we're past the end, grab the last chunk and reset for the next epoch
            line_ix = torch.arange(start_ix, len(data_tokenized))
            split_pointers[split] = 0 # Reset for next call
        else:
            # Otherwise, grab the next sequential chunk
            line_ix = torch.arange(start_ix, end_ix)
            split_pointers[split] = end_ix # Update pointer for next call

        # The actual batch size might be smaller if it's the last batch
        actual_batch_size = len(line_ix)
        if actual_batch_size == 0:
            # This can happen if the last batch was perfectly sized
            # and the pointer was reset. We just call ourselves again.
            return get_batch(split, arithmetic_batch)

        line_list = []
        for idx in line_ix:
            token_ids = data_tokenized[idx] 
            if len(token_ids) < 2:
                token_ids = torch.tensor([0, 0], dtype=torch.long)
            line_list.append(token_ids)

        max_len = max(len(t) for t in line_list)
        # 3. Use the actual_batch_size instead of the global batch_size
        samples = torch.ones(actual_batch_size, max_len, dtype=torch.long) * pad_token_id

        for i in range(actual_batch_size):
            seq_len = len(line_list[i])
            samples[i, :seq_len] = line_list[i]

            x = samples[:, :-1].clone()
            y = samples[:, 1:].clone()

    else:
        data = train_data if split == 'train' else val_data
        # ====== 继续原先的 block_size 随机切片逻辑 ======
        if train_both:
            data2 = train_data2 if split == 'train' else val_data2
            batch_size2 = int(batch_size*data_ratio)
            ix = torch.randint(len(data) - block_size, (batch_size-batch_size2,))
            ix2 = torch.randint(len(data2) - block_size, (batch_size2,))
        else:
            ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if train_both:
            x2 = torch.stack([torch.from_numpy((data2[i:i+block_size]).astype(np.int64)) for i in ix2])
            y2 = torch.stack([torch.from_numpy((data2[i+1:i+1+block_size]).astype(np.int64)) for i in ix2])
            x = torch.cat([x,x2])
            y = torch.cat([y,y2])  
 

    # 在这里添加掩码处理
    # 假设问题部分的长度为 `question_length`
    # 我们需要找到每个样本中问题部分的实际长度

    # 初始化掩码矩阵，默认全部掩码（True 表示掩码）
    mask = torch.ones_like(y, dtype=torch.bool)

# 遍历每个样本，设置掩码
    for batch_idx in range(y.size(0)):
        sample_x = x[batch_idx]
        sample_len = sample_x.size(0)
        # 找到 '=' 和 '$' 符号的位置
        equal_positions = (sample_x == equal_token_id).nonzero(as_tuple=False).squeeze()
        dollar_positions = (sample_x == dollar_token_id).nonzero(as_tuple=False).squeeze()
        # Check if dollar_positions is empty
        if dollar_positions.numel() == 0:
            # If dollar_positions is empty, use newline_positions instead
            dollar_positions = (sample_x == newline_token_id).nonzero(as_tuple=False).squeeze()

        # 确保 equal_positions 和 dollar_positions 是一维张量
        if equal_positions.dim() == 0:
            equal_positions = equal_positions.unsqueeze(0)
        if dollar_positions.dim() == 0:
            dollar_positions = dollar_positions.unsqueeze(0)

        equal_positions = equal_positions.tolist()
        dollar_positions = dollar_positions.tolist()

        # 对 dollar_positions 进行排序
        dollar_positions = sorted(dollar_positions)

        # 对于每个 '='，找到其后的第一个 '$'
        for eq_pos in equal_positions:
            # 找到第一个大于等于 eq_pos 的 dollar_pos
            next_dollar_positions = [pos for pos in dollar_positions if pos >= eq_pos]
            if not next_dollar_positions:
                # 如果没有找到 '$'，跳过此 '='
                continue
            dollar_pos = next_dollar_positions[0]

            operator_positions = (sample_x == dollar_token_id) 
            operator_positions = operator_positions.nonzero(as_tuple=False).squeeze(-1).tolist()

            # 判断：若在 eq_pos 之前，没有任何运算符，则判定为不完整
            if not any(op < eq_pos for op in operator_positions):
                # 不执行 unmask，直接跳过，继续保持原本被 mask 的状态
                continue
            # 将 [eq_pos, dollar_pos] 区间设置为不掩码（False）
            mask[batch_idx, eq_pos:dollar_pos] = False

    if True:
        if zero_pad_in_target:
            y[mask] = zero_token_id
        else:
            y[mask] = -100  # 将需要掩码的位置的标签设置为 -100

    return x, y

# Step 3: Write code to fetch a batch and decode x and y
# Get a batch

x, y = get_batch('train',arithmetic_batch)
# Iterate through each sequence in the batch
y[y == -100] = star_token_id
    # Replace -100 with star_token_id in y
# Decode x and y
decoded_x = [data_decoder(seq.tolist()) for seq in x]
decoded_y = [data_decoder(seq.tolist()) for seq in y]
# Print the decoded x and y
for i in range(len(decoded_x)):
    print(f"Sample {i}:")
    print(f"x = {decoded_x[i]}")
    print(f"y = {decoded_y[i]}")
    print("-" * 50)

x, y = get_batch('val',arithmetic_batch)
# Iterate through each sequence in the batch
y[y == -100] = star_token_id
    # Replace -100 with star_token_id in y
# Decode x and y
decoded_x = [data_decoder(seq.tolist()) for seq in x]
decoded_y = [data_decoder(seq.tolist()) for seq in y]
# Print the decoded x and y
for i in range(len(decoded_x)):
    print(f"Sample {i}:")
    print(f"x = {decoded_x[i]}")
    print(f"y = {decoded_y[i]}")
    print("-" * 50)
