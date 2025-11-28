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
operator = 'multi_add'
operator_digit_test="multi_add"

dataset = 'newtask'
batch_size = 128  # Adjusted as per command
block_size = 256
max_operand_number_multiadd=10
blank_space_in_equation_number = 11
train_data_path = 'multiadd_train_5.txt'
start = 'FILE:data/newtask/multiadd_test_10.txt'
start_train="FILE:data/newtask/multiadd_test_5.txt"
ckpt_path_name = f'monet_6layer1mixing1poly_5000iter.pt'  # Checkpoint name as per layers and mixings
reverse_c = True
eval_addition = True
eval_addition_train = True
num_digit=12
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
zero_pad = True
max_number_length = 0
blank_space_in_equation_number = 10
pad_answer = True  
fix_blank_space_position = True
blank_space_exact=True
blank_space_split_number=True
max_operand_number = 10
operand_number_exact_multiadd = False