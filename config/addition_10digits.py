# Model and Training Configuration
out_dir = f'out/'  # Dynamic output directory based on layers and mixings
eval_interval = 1000 
eval_iters = 10
log_interval = 100
digit_test_number=100
arithmetic_batch=True

always_save_checkpoint = False
wandb_log = False 
wandb_project = 'arithmetic'
wandb_run_name = 'addition_10'

data_type = 'text'
data_format = 'reverse'
operator = '+'
dataset = 'bal'
batch_size = 64  
block_size = 1024
train_data_path = '+_maxLen_10_limit_100000_train_minReq_0.txt'
start = ''
start_train="FILE:data/bal/+_maxLen_10_limit_1000_test_minReq_0.txt"                        
ckpt_path_name = f'addition_10.pt'  
eval_addition = True
eval_addition_train = True
num_digit=100
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
device = 'cuda'  
# Training settings for reverse and padding
reverse_c = True
reverse_ab = True
index_hint = False
zero_pad_in_target = False
zero_pad_in_training = True
max_number_length = 0
blank_space_in_equation_number = 101 
pad_answer = True  
fix_blank_space_position = True
blank_space_exact=True
blank_space_split_number=True
