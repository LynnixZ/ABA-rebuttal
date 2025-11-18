# Model and Training Configuration
out_dir = f'out/monet/masking/6layer1mixing1poly/'  # Dynamic output directory based on layers and mixings
eval_interval = 100  # As specified
eval_iters = 200
log_interval = 10
digit_test_number=10
arithmetic_batch=True

always_save_checkpoint = False
wandb_log = False  # Disable wandb logging per command line
wandb_project = 'addition'
wandb_run_name = 'addition_reverse'
operator_digit_test="strsort"
data_type = 'text'
data_format = 'plain'
operator = 'strsort'
dataset = 'newtask'
batch_size = 128  # Adjusted as per command
block_size = 512
train_data_path = 'strsort_train_nopad.txt'
start = 'FILE:data/newtask/test_str_sort_5len_5array.txt'
start_train="FILE:data/newtask/test_str_sort_10len_10array.txt"
ckpt_path_name = f'monet_6layer1mixing1poly_5000iter.pt'  # Checkpoint name as per layers and mixings
eval_addition = True
eval_addition_train = True
num_digit=10

# Model architecture settings
n_layer = 6  # Layer count as per command line
n_embd = 384  # Updated embedding size
n_head = 6
dropout = 0.2
learning_rate = 0.001  # Updated learning rate
gradient_accumulation_steps = 8  # Added as per command line

max_iters = 5000
lr_decay_iters = 5000
beta2 = 0.99

warmup_iters = 100
device = 'cuda'  # As specified

# Mixing and positional configurations
mixing_number = 1
poly_number = 2
max_mixing_length = 64
position_embedding = 'none'
norm_in_mixing_layer = 'LayerNorm'
norm_in_polyblock = 'none'
norm_before_output = 'none'

# Training settings for reverse and padding
reverse_ab = False
reverse_c= False
index_hint = False
zero_pad_in_training = False
max_number_length = 0
noorder = False
blank_space_in_equation_number = 0  # Placeholder; ensure `blank_space` is defined
pad_answer = True  # Placeholder; ensure `pad_answer` is defined
fix_blank_space_position = True
blank_space_exact=False
blank_space_split_number=True
evaluation = True
