
# Model and Training Configuration
out_dir = f'out/monet/masking/6layer1mixing1poly/'  # Dynamic output directory based on layers and mixings
eval_interval = 1000  # As specified
eval_iters = 10
log_interval = 100
digit_test_number=10
arithmetic_batch=True

always_save_checkpoint = False
wandb_log = False  
wandb_project = 'addition'
wandb_run_name = 'addition_reverse'

data_type = 'text'
data_format = 'reverse'
operator = '*'
dataset = 'bal'
batch_size = 8 
block_size = 1024
train_data_path = '+_maxLen_5_limit_1000000_train_minReq_0.txt'
start = 'FILE:data/bal/+_maxLen_8_limit_20_test_minReq_5.txt'
start_train="FILE:data/bal/+_maxLen_5_limit_20_test_minReq_0.txt"
ckpt_path_name = f'monet_6layer1mixing1poly_5000iter.pt' 
eval_addition = True
eval_addition_train = True
num_digit=20

# Model architecture settings
n_layer = 6
n_embd = 384
n_head = 6
dropout = 0.2
positional_embedding = 'learned'
bias=True
learning_rate = 0.0005
gradient_accumulation_steps = 8 
max_iters = 5000
lr_decay_iters = 100000
beta2 = 0.99


warmup_iters = 2000
device = 'cuda'  # As specified


# Training settings for reverse and padding
reverse_c = True
index_hint = False
zero_pad = True
zeropad_max_length = 0
blank_space_in_equation_number = 21  # Placeholder; ensure `blank_space` is defined
fix_blank_space_position = True
blank_space_exact=False
evaluation = True
