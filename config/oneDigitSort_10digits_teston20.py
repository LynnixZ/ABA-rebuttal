# Model and Training Configuration
out_dir = f'out/monet/masking/6layer1mixing1poly/'  # Dynamic output directory based on layers and mixings
eval_interval = 100  # As specified
eval_iters = 200
log_interval = 10

always_save_checkpoint = False
wandb_log = False  # Disable wandb logging per command line
wandb_project = 'addition'
wandb_run_name = 'addition_reverse'
operator_digit_test="oneDigitSort"
data_type = 'text'
data_format = 'reverse'
operator = 'copy'
dataset = 'bal'
batch_size = 64  # Adjusted as per command
block_size = 256
train_data_path = 'newtask/oneDigitSort_train_10.txt'
start = 'FILE:data/bal/newtask/oneDigitSort_test_20.txt'
start_train="FILE:data/bal/newtask/oneDigitSort_test_10.txt" #不知道为什么无法在command line改
ckpt_path_name = f'monet_6layer1mixing1poly_5000iter.pt'  # Checkpoint name as per layers and mixings
reverse_c = True
eval_addition = True
eval_addition_train = True
num_digit=30

# Model architecture settings
n_layer = 6  # Layer count as per command line
n_embd = 512  # Updated embedding size
n_head = 6
dropout = 0.2
learning_rate = 0.001  # Updated learning rate
gradient_accumulation_steps = 4  # Added as per command line

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
reverse_ab = True
index_hint = False
zero_pad_in_target = False
zero_pad_in_training = False
max_number_length = 0
noorder = False
blank_space_in_equation_number = 21  # Placeholder; ensure `blank_space` is defined
pad_answer = True  # Placeholder; ensure `pad_answer` is defined
fix_blank_space_position = True
blank_space_exact=False
blank_space_split_number=True
evaluation = True
model_type = 'Monet' 
proxy_number= 0
monet_modeltype="poly+mix"
mixinglayer_in_the_beggining=3