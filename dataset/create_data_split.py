from transformers import PreTrainedTokenizer
import random
import os
import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, DatasetDict
import pandas as pd
import datasets
import json
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import cm
import re
from dataset_analysis import main as data_analysis_main
import numpy as np

# ============== 修改点 #1 =============
# 原先 pick_char_set 中包含了很多希腊字母，这里去掉它们
def pick_char_set(max_len):
    """Pick a set of characters in a cyclic method for index hints (去掉希腊字符)."""
    # 这里仅保留了 A-Z, a-z, 以及一些常见符号, 没有任何希腊字母
    set_of_chars = [
        'A','B','C','D','E','F','G','H','I','J','K','L','M',
        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','c','d','e','f','g','h','i','j','k','l','m',
        'n','o','p','q','r','s','t','u','v','w','x','y','z',
        '!', '@', '£', '#', '$', '%', '^', '&', '*', '(',
        ')', '~', '?', '.', ',', '<', '>', '{', '}', '[',
        ']', ':', ';','/','|'
    ]
    output = []
    start = random.randint(0, len(set_of_chars))
    if start + max_len > len(set_of_chars):  # cycle round
        return set_of_chars[start:] + set_of_chars[:(start + max_len - len(set_of_chars))]
    else:
        return set_of_chars[start:start + max_len]

def hints_helper(num_str, chars):
    # returns the positional hints with the number
    result = ""
    for char, digit in zip(chars, num_str):
        result += f"{char}{digit}"
    return result

# ============== 这段原先已存在 =============
def generate_no_carry_addition(n, m):
    """No carries addition, brute force implementation"""
    num1 = random.randint(10**(n-1), 10**n - 1)
    num2 = random.randint(10**(m-1), 10**m - 1)

    while has_carry(num1, num2):
        num1 = random.randint(10**(n-1), 10**n - 1)
        num2 = random.randint(10**(m-1), 10**m - 1)

    return num1, num2, num1 + num2

def has_carry(num1, num2):
    # Check if there is a carry in any column during addition
    for digit1, digit2 in zip(str(num1)[::-1], str(num2)[::-1]):
        if int(digit1) + int(digit2) >= 10:
            return True
    return False

# ============== 修改点 #2 & #3 =============
# 新增: weighted_method_gen, 用于演示“训练与测试使用不同digit长度分布”以及 min_required_digit_len
def weighted_method_gen(
    max_digit_len=5,
    operation='+',
    limit=1000,
    p=0,
    no_carry_addition=False,
    reverse_answer=False,
    reverse_all=False,
    keep_0_for_len_1=False,
    min_required_digit_len=0,  # <<< 如果 >0，需要 max(len1, len2) > min_required_digit_len
    is_test=False,             # <<< 新增参数，如果是测试集 => digit_weights=[1,1,...,1]
    Flags=None
):
    """
    使用自定义(或等)权重来随机生成两个操作数的位数，再产生对应运算的数据集。
      - 若 is_test=True，则 digit_weights = [1, 1, ..., 1] (等权)
      - 若 is_test=False，则 digit_weights = [1, 2, ..., max_digit_len] (示例递增)
      - 若 min_required_digit_len>0，则只有当 max(len1, len2) > min_required_digit_len 时才保留该样本。
    """

    dataset = []

    # digit_lengths = [1..max_digit_len]
    digit_lengths = list(range(1, max_digit_len + 1))
    
    # 根据 is_test 来决定权重分布
    if is_test:
        digit_weights = [1 for _ in digit_lengths]
    else:
        # 如: 训练集时给出某种非均匀分布, 这里示例用 [1,2,...,max_digit_len]
        digit_weights = [i for i in digit_lengths]

    total_weight = sum(digit_weights)

    def sample_digit_length():
        # 离散加权采样
        r = random.randint(1, total_weight)
        cumulative = 0
        for length, w in zip(digit_lengths, digit_weights):
            cumulative += w
            if r <= cumulative:
                return length
        return digit_lengths[-1]  # 理论上不该到这里

    while len(dataset) < limit:
        # 1) 抽 operand1、operand2 的位数
        len1 = sample_digit_length()
        len2 = sample_digit_length()

        # 2) 若 min_required_digit_len>0, 要求 max(len1, len2) > min_required_digit_len
        if max(len1, len2) <= min_required_digit_len:
            continue

        # 3) 生成操作数
        if keep_0_for_len_1 and len1 == 1:
            start_i = 0
            end_i   = 10**len1 - 1
        else:
            start_i = 10**(len1 - 1)
            end_i   = 10**len1 - 1

        if keep_0_for_len_1 and len2 == 1:
            start_j = 0
            end_j   = 10**len2 - 1
        else:
            start_j = 10**(len2 - 1)
            end_j   = 10**len2 - 1

        if no_carry_addition and operation == '+':
            num1, num2, _ = generate_no_carry_addition(len1, len2)
        else:
            num1 = random.randint(start_i, end_i)
            num2 = random.randint(start_j, end_j)

        num1_str = str(num1)
        num2_str = str(num2)

        # 4) 做运算
        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == 'x':
            result = num1 * num2
        else:
            raise ValueError("Invalid operation")

        result_str = str(result)

        # 5) 反转逻辑
        if reverse_all:
            num1_str = num1_str[::-1]
            num2_str = num2_str[::-1]
            result_str = result_str[::-1]
        elif reverse_answer:
            result_str = result_str[::-1]

        # 6) 随机空格插入
        dataset_entry = f"{num1_str}{operation}{num2_str}={result_str}"
        if p > 0:
            spaced_string = ""
            for ch in dataset_entry:
                space_p = p
                while random.random() < space_p:
                    space_p *= 0.1
                    spaced_string += " "
                spaced_string += ch
            dataset_entry = spaced_string

        dataset.append(dataset_entry)

    return dataset


# Function to generate the arithmetic dataset
def generate_dataset(dir_name, operation, n, m, num_examples, base_folder_name, keep_places, exact, prepend_zeros, reverse_answer, reverse_all, p=0, no_carry_addition=False, seed=42, interleave=False):
    """
    generate a dataset, NOT using the bucket method!
    p = probability for random padding to be inserted
    """
    if p < 0 or p >= 1:
        raise ValueError("Probability p must be strictly between 0 and 1.")

    random.seed()
    dataset = []

    for _ in range(num_examples):
        if exact: # exactly length n,m 
            num1 = random.randint(10**(n-1), 10**n - 1)
            num2 = random.randint(10**(m-1), 10**m - 1)
            
        elif no_carry_addition and operation == '+':
            num1, num2, _ = generate_no_carry_addition(n,m)
        else:
            # num1 = random.randint(0, 10**n - 1)
            # num2 = random.randint(0, 10**m - 1)
            num1 = random.randint(10**(n-1), 10**n - 1)
            num2 = random.randint(10**(m-1), 10**m - 1)

        if keep_places: # fill with zeros so it is always the same length
            num1_str = str(num1).zfill(n)
            num2_str = str(num2).zfill(m)
        else:
            num1_str = str(num1)
            num2_str = str(num2)

        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == 'x':
            result = num1 * num2
        else:
            raise ValueError("Invalid operation")

        result = str(result)

        if prepend_zeros > 0:
            zeros = "0"*prepend_zeros
            num1_str = zeros + num1_str
            num2_str = zeros + num2_str
            result = "0" + zeros + result

        orgional_p = p

        if reverse_all: # reversals 
            result = result[::-1]
            num1_str = num1_str[::-1]
            num2_str = num2_str[::-1]
        elif reverse_answer:
            result = result[::-1]
        

        dataset_entry = f"{num1_str}{operation}{num2_str}={result}"
        if interleave: # interleave the operands so the digits of the same significance are  next to eachother
            dataset_entry = ''.join([a + b for a, b in zip(num1_str, num2_str)]) + num1_str[len(num2_str):] + num2_str[len(num1_str):]+f"={result}"
        p = orgional_p
        if p > 0: # adds random spaces, exponentially decaying
            dataset_entry = f"{num1_str}{operation}{num2_str}={result}"
            if interleave:
                dataset_entry = ''.join([a + b for a, b in zip(num1_str, num2_str)]) + num1_str[len(num2_str):] + num2_str[len(num1_str):]+f"={result}"
            spaced_string = ""
            for char in dataset_entry:
                space_p = p
                while random.random() < space_p:
                    space_p *= 0.1
                    spaced_string += " "
                spaced_string += char
            dataset_entry = spaced_string
        dataset.append(dataset_entry)

    for i in range(0,min(len(dataset),5)):
        print(dataset[i])
    
    folder_name = f"{base_folder_name}/{dir_name}"
    os.makedirs(folder_name, exist_ok=True)
    # automated file name
    file_name = f"{operation}_n_{n}_m_{m}_examples_{num_examples}{'_diff_lens' if not keep_places else ''}{'_exact' if exact else ''}{f'_prepend_{prepend_zeros}zeros' if prepend_zeros>0 else ''}{f'_reverse_ans' if reverse_answer else ''}{f'_prob_space_{p}' if p>0 else ''}_seed_{seed}.txt"
    file_path = os.path.join(folder_name, file_name)

    with open(file_path, 'w') as file:
        for entry in dataset:
            file.write(entry + '\n')
    print(f"created: {file_path}")
    return dataset, folder_name, file_path


def tokenize_and_save_dataset(dataset, tokenizer, directory, test_split_ratio=0.05, pad_sequences=False):
    # tokenization, slow but gets the job done

    os.makedirs(directory, exist_ok=True)

    # Tokenize the dataset and add EOS token at the end of each entry
    eos_token_id = tokenizer.vocab[tokenizer.eos_token]
    tokenized_dataset = [tokenizer(entry)["input_ids"] + [eos_token_id] for entry in dataset]

    # print some of them say 5 input and its tokenized version
    print("Some examples of tokenized dataset:")
    for i in range(0,min(len(dataset),5)):
        print(f"Input: {dataset[i]}")
        print(f"Tokenized: {tokenized_dataset[i]}")
        decoded = tokenizer.decode(tokenized_dataset[i])
        print(f"Decoded: {decoded}")
        print()

    # Optionally pad the sequences
    if pad_sequences:
        max_length = max(len(entry) for entry in tokenized_dataset)
        pad_token_id = tokenizer.pad_token_id
        tokenized_dataset = [entry + [pad_token_id] * (max_length - len(entry)) for entry in tokenized_dataset]

    save_to_json_intermed = False # save the tokenized dataset to a json instead of hf
    if save_to_json_intermed:
        print(tokenized_dataset)
        data_path = os.path.join(directory, "dataset.json")
        with open(data_path, "w") as outfile:
            # Iterate over each dictionary in the list
            for entry in tokenized_dataset:
                # Convert dictionary to JSON string and write it to the file
                json.dump({'input_ids': entry}, outfile)
                # Write a newline character to separate each JSON object
                outfile.write('\n')
        exit()

    # Split the data into train and test sets
    test_size = int(len(tokenized_dataset) * test_split_ratio)
    train_data = tokenized_dataset[:-test_size]
    test_data = tokenized_dataset[-test_size:]
    # Convert to Hugging Face datasets with 'input_ids' column
    train_dataset = Dataset.from_pandas(pd.DataFrame({"input_ids": train_data}))
    test_dataset = Dataset.from_pandas(pd.DataFrame({"input_ids": test_data}))

    # Create a DatasetDict with train and test splits
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Save the dataset to disk
    hf_dataset_path = os.path.join(directory, "hf_tokenized_dataset")
    dataset_dict.save_to_disk(hf_dataset_path)

    # # Save tokenizer
    # print(f"Tokenized data saved to {tokenized_data_path}")
    print(f"HuggingFace Dataset saved to {hf_dataset_path}")

    # return dataset_dict, tokenized_data_path, hf_dataset_path #, tokenizer_dir
    return dataset_dict, hf_dataset_path

def character_histogram(dir_name, condense_white_space=False):
    """Histogram of character occurences"""
    base_directory = ""
    dir_name = os.path.join(base_directory, dir_name)

    # open all data files and append to big list
    dataset = []
    for filename in os.listdir(dir_name):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir_name, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                stripped_lines = [line.replace("\n", "") for line in lines]
                if condense_white_space:
                    stripped_lines = [re.sub('\s+',' ', line) for line in lines]
                dataset.extend(stripped_lines)

    for i in range(0,min(len(dataset),5)):
        print(dataset[i])

    max_length = max(map(len, dataset))
    
    counters_list = [Counter() for _ in range(max_length)]

    for string in dataset:
        for index, char in enumerate(string):
            counters_list[index][char] += 1

    # Plot the occurrences for each index
    plt.figure(figsize=(10, 6))
    indices = list(range(max_length))
    bottom = [0] * max_length
    sorted_chars = sorted(set(''.join(dataset)))

    colors = cm.get_cmap('tab20', len(sorted_chars))

    for char, color in zip(sorted_chars, colors.colors):
        occurrences = [counter[char] for counter in counters_list]
        legend_char = char if char != " " else "\' \'"
        plt.bar(indices, occurrences, label=legend_char, bottom=bottom, color=color)
        bottom = [b + o for b, o in zip(bottom, occurrences)]

    plt.xlabel('Index')
    plt.ylabel('Occurrences')
    plt.title("Character Frequency")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=10)
    plt.savefig(f"{dir_name}/char_histogram{'_condensed_ws' if condense_white_space else ''}", bbox_inches='tight')

def token_histogram(dir_name, tokenizer_type="normal"):
    """Histogram of token occurences"""
    base_directory = "./cramming-data/data/arithmetic_data"
    dir_name = os.path.join(base_directory, dir_name)
    hf_dir_name = os.path.join(dir_name, "hf_tokenized_dataset")
    tokenized_dataset = datasets.load_from_disk(hf_dir_name)
    train_part = tokenized_dataset["train"]
    test_part = tokenized_dataset["test"]
    
    tokenizer = get_tokenizer(tokenizer_type)
    EOS_token = tokenizer._convert_token_to_id("[EOS]")
    
    dataset = []
    for example in train_part:
        tokens = example["input_ids"]
        eos_index = tokens.index(EOS_token) if EOS_token in tokens else len(tokens) # not including the EOS token
        tokens = tokens[:eos_index]
        dataset.append(tokens)
    for example in test_part:
        tokens = example["input_ids"]
        eos_index = tokens.index(EOS_token) if EOS_token in tokens else len(tokens) # not including the EOS token
        tokens = tokens[:eos_index]
        dataset.append(tokens)

    for i in range(0,min(len(dataset),5)):
        print(dataset[i])

    max_length = max(map(len, dataset))
    counters_list = [Counter() for _ in range(max_length)]

    for string in dataset:
        for index, char in enumerate(string):
            counters_list[index][str(char)] += 1

    plt.figure(figsize=(10, 6))
    indices = list(range(max_length))
    bottom = [0] * max_length
    print(tokenizer.vocab.values())
    sorted_chars = [str(x) for x in sorted(tokenizer.vocab.values())]
    
    colors = cm.get_cmap('tab20', len(sorted_chars))

    for char, color in zip(sorted_chars, colors.colors):
        occurrences = [counter[char] for counter in counters_list]
        tokenizer_char = tokenizer._convert_id_to_token(int(char))
        tokenizer_char = tokenizer_char if tokenizer_char != " " else "\' \'"
        legend_char = f"{char} => {tokenizer_char}"
        plt.bar(indices, occurrences, label=legend_char, bottom=bottom, color=color)
        bottom = [b + o for b, o in zip(bottom, occurrences)]

    plt.xlabel('Index')
    plt.ylabel('Occurrences')
    plt.title("Token Frequency")
    legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=6)
    legend.set_title("token => char")

    plt.savefig(f"{dir_name}/token_histogram", bbox_inches='tight')

def main_dataset_gen(dir_name, op, n, m, num_samples, exact=False, keep_places=False, prepend_zeros=0, reverse_answer=False, reverse_all=False, p=0, no_carry_addition=False, seed=42, interleave=False):
    """Main method for non bucket datasets"""
    base_directory = "./cramming-data/data"
    os.makedirs(base_directory, exist_ok=True)
    base_directory = f"{base_directory}/arithmetic_data"
    os.makedirs(base_directory, exist_ok=True)
    
    dataset, data_folder_name, _ = generate_dataset(dir_name, op, n, m, num_samples, base_directory, keep_places, exact, prepend_zeros, reverse_answer, reverse_all, p, no_carry_addition, seed=seed, interleave=interleave)

def tokenize_main(dir_name, tokenizer_type, test_split_ratio=0.05):
    """Main tokenizer method"""
    base_directory = "./cramming-data/data/arithmetic_data"
    dir_name = os.path.join(base_directory, dir_name)
    data_folder_name = dir_name

    # Initialize the tokenizer
    tokenizer = get_tokenizer(tokenizer_type)

    # open all data files and append to big list
    dataset = []

    for filename in os.listdir(dir_name):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir_name, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                # stripped_lines = [line.strip() for line in lines]
                stripped_lines = [line.replace("\n", "") for line in lines]
                dataset.extend(stripped_lines)
    random.shuffle(dataset) # shuffling all the datasets together

    dataset_dict, hf_dataset_path = tokenize_and_save_dataset(dataset, tokenizer, data_folder_name,
                                                                                   pad_sequences=True,
                                                                                   test_split_ratio=test_split_ratio)
    tokenized_dataset = datasets.load_from_disk(hf_dataset_path)
    print(tokenized_dataset)


def pick_char_set(max_len):
    """Pick a set of characters in a cyclic method for index hints"""
    # 102 characters
    set_of_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', '!', '@', '£', '#', '$', '%', '^', '&', '*', '(', ')', '~', '?', '.', ',', '<', '>', '{', '}', '[', ']', ':', ';','/','|','β','Γ', 'Δ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ','Λ', 'λ', 'μ', 'Ξ', 'ξ','Π', 'π','Σ', 'ς', 'τ', 'Φ', 'φ', 'χ', 'Ψ', 'ψ', 'Ω', 'ω']
    
    output = []
    start = random.randint(0, len(set_of_chars))
    if start + max_len > len(set_of_chars): # i.e. cycle round
        return set_of_chars[start:len(set_of_chars)] + set_of_chars[:start + max_len-len(set_of_chars)]
    else:
        return set_of_chars[start:start + max_len]

def hints_helper(num_str, chars):
    # returns the positional hints with the number
    result = ""
    for char, digit in zip(chars, num_str):
        result += f"{char}{digit}"
    return result
def bucket_method_gen(n=3, m=3, operation='+', limit=1000, p=0, no_carry_addition=False, reverse_answer=False, start=1, reverse_all=False, keep_0_for_len_1=False, Flags=None):
    """Bucket method generator, samples all operand lengths equally"""
    dataset = []
    while True:
        for i in range(start, n + 1):
            for j in range(start, m + 1):
                if operation == '@':
                    num1 = random.randint(0, 2**i - 1)
                    num2 = random.randint(0, 2**j - 1)
                    num1_str = bin(num1)[2:]
                    num2_str = bin(num2)[2:]
                    result = bin(num1 + num2)[2:]
                else:
                    start_i = 10**(i - 1)
                    start_j = 10**(j - 1)
                    if keep_0_for_len_1 and i == 1:  # i.e. use naturals including 0, we just use naturals
                        start_i = 0
                    if keep_0_for_len_1 and j == 1:
                        start_j = 0
                    num1 = random.randint(start_i, 10**i - 1)
                    num2 = random.randint(start_j, 10**j - 1)

                    if no_carry_addition and operation == '+':
                        num1, num2, _ = generate_no_carry_addition(i, j)
                    num1_str = str(num1)
                    num2_str = str(num2)

                    if operation == '+':
                        result = num1 + num2
                    elif operation == '-':
                        result = num1 - num2
                    elif operation == 'x':
                        result = num1 * num2
                    else:
                        raise ValueError("Invalid operation")

                result = str(result)
                if reverse_answer: # reversals
                    result = result[::-1]
                if reverse_all:
                    result = result[::-1]
                    num1_str = num1_str[::-1]
                    num2_str = num2_str[::-1]
                if Flags.index_hints: # adding the index hints
                    max_len = max(len(result), max(len(num1_str),len(num2_str)))
                    chars = pick_char_set(max_len)
                    result = hints_helper(result, chars)
                    num1_str = hints_helper(num1_str, chars)
                    num2_str = hints_helper(num2_str, chars)
                else:
                    dataset_entry = f"{num1_str}{operation}{num2_str}={result}"

                    if p > 0: # adds random spaces
                        spaced_string = ""
                        for char in dataset_entry:
                            space_p = p
                            while random.random() < space_p:
                                space_p *= 0.1
                                spaced_string += " "
                            spaced_string += char
                        dataset_entry = spaced_string
                
                dataset.append(dataset_entry)
                if len(dataset) == limit:
                    return dataset

def bucket_method_main(n, m, operation, limit, dir_name, p=0, no_carry_addition=False, reverse_answer=False, start=1, reverse_all=False, keep_0_for_len_1=False, Flags=None):
    """Mains method for bucket style generation"""
    dataset = bucket_method_gen(n, m, operation, limit, p, no_carry_addition, reverse_answer, start, reverse_all=reverse_all, keep_0_for_len_1=keep_0_for_len_1, Flags=Flags)
    for i in range(0,10):
        print(dataset[i])
    
    base_directory = "./cramming-data/data"
    os.makedirs(base_directory, exist_ok=True)
    base_directory = f"{base_directory}/arithmetic_data"
    os.makedirs(base_directory, exist_ok=True)
    
    folder_name = f"{base_directory}/{dir_name}"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"{operation}_n_{n}_m_{m}_examples_{limit}.txt"
    file_path = os.path.join(folder_name, file_name)

    random.seed()
    random.shuffle(dataset)
    with open(file_path, 'w') as file:
        for entry in dataset:
            file.write(entry + '\n')
    print(f"created: {file_path}")
    return dataset, folder_name, file_path


def uniform_distribution_sort_basic(maximum_number_of_digts, maximum_length, limit, FLAGS):
    """sorting dataset generator"""
    dataset = []
    for i in range(0, limit):
        dataset_entry = ""
        chars = pick_char_set(maximum_length)
        local_chars = pick_char_set(maximum_number_of_digts)
        all_nums = []
        for j in range(0, maximum_length):
            # choose a random number of digit between 1 and maximum_number_of_digts
            num_digit = random.randint(1, maximum_number_of_digts)
            # pick a number with num_digit digits
            num = random.randint(10**(num_digit-1), 10**num_digit - 1)
            all_nums.append([chars[j], num])

            num = str(num)
            if FLAGS.reverse_all:
                num = num[::-1]
            if FLAGS.index_hints:
                num = hints_helper(num, local_chars)
            dataset_entry += f"{chars[j]}:{num},"

        dataset_entry = dataset_entry[:-1]
        all_nums = sorted(all_nums, key=lambda x: x[1]) # get the answer
        sorted_chars = [x[0] for x in all_nums]
        dataset_entry += f"={','.join(sorted_chars)}" # convert them into a string separated by ,
        dataset.append(dataset_entry)

    return dataset

def bucket_uniform_distribution(maximum_number_of_digts, maximum_length, limit, FLAGS):
    """Use a uniform distribution over -- i.e. bucket method for sorting"""
    bucket_limit = limit // (maximum_length * maximum_number_of_digts)
    dataset = []
    for i in range(0, maximum_length):
        for j in range(0, maximum_number_of_digts):
            dataset += uniform_distribution_sort_basic(j+1, i+1, bucket_limit, FLAGS)
    return dataset

def uniform_distribution_sort_main(FLAGS, dir_name):
    """Main method for sorting generation"""
    maximum_number_of_digts = FLAGS.n
    maximum_length = FLAGS.m
    limit = FLAGS.limit

    dataset = bucket_uniform_distribution(maximum_number_of_digts, maximum_length, limit, FLAGS)

    for i in range(0, 10):
        print(dataset[i])

    base_directory = "./cramming-data/data"
    os.makedirs(base_directory, exist_ok=True)
    base_directory = f"{base_directory}/arithmetic_data"
    os.makedirs(base_directory, exist_ok=True)

    folder_name = f"{base_directory}/{dir_name}"
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"sort_maximum_number_of_digts_{FLAGS.n}" \
                f"_maximum_length_{FLAGS.m}_examples_{limit}.txt"
    file_path = os.path.join(folder_name, file_name)

    random.seed()
    random.shuffle(dataset)
    with open(file_path, 'w') as file:
        for entry in dataset:
            file.write(entry + '\n')
    print(f"created: {file_path}")
    return dataset, folder_name, file_path

# ============== 在 main() 中增加处理 weighted_method_gen 的逻辑 =============
def main_dataset_gen(dir_name, op, n, m, num_samples, exact=False, keep_places=False,
                     prepend_zeros=0, reverse_answer=False, reverse_all=False, p=0,
                     no_carry_addition=False, seed=42, interleave=False):
    """
    Main method for non-bucket datasets(加法/乘法等),
    这里保持原逻辑不变, 仅演示如何集成 weighted_method_gen.
    """
    base_directory = "./cramming-data/data"
    os.makedirs(base_directory, exist_ok=True)
    base_directory = f"{base_directory}/arithmetic_data"
    os.makedirs(base_directory, exist_ok=True)
    
    dataset, data_folder_name, _ = generate_dataset(
        dir_name, op, n, m, num_samples, base_directory,
        keep_places, exact, prepend_zeros, reverse_answer, reverse_all,
        p, no_carry_addition, seed=seed, interleave=interleave
    )

# ...........................
# 其它已有函数 (generate_dataset, tokenize_main, uniform_distribution_sort_main, 等) 不变
# ...........................

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    # 原有参数
    parser.add_argument("--dir_name", type=str, required=True, help='name of dataset')
    parser.add_argument("--op", type=str, default='+', help="operation e.g. +,-,x")
    parser.add_argument("--n", default=2, type=int, help="num digits in first number")
    parser.add_argument("--m", default=2, type=int, help="num digits in second number")
    parser.add_argument("--num_samples", default=100, type=int, help="number of samples")
    parser.add_argument("--seed", default=42, type=int, help="seed for random generation")
    parser.add_argument('--keep_places', action='store_true')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--special', action='store_true')
    parser.add_argument('--p', default=0.0, type=float, help="prob for adding padding")
    parser.add_argument("--prepend_zeros", default=0, type=int, help="prepend this number of zeros")
    parser.add_argument('--reverse_answer', action='store_true', help="reverses the answer")
    parser.add_argument('--reverse_all', action='store_true', help="reverses the inputs and answer")
    parser.add_argument('--no_carry_addition', action='store_true', help="no carried in the addition")
    parser.add_argument('--test_split_ratio', default=0.05, type=float, help="test split percentage")
    parser.add_argument('--interleave', action='store_true')
    parser.add_argument('--keep_0_for_len_1', action='store_true')
    
    # bucket
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument("--limit", default=1000000, type=int, help="number of samples if using the bucket method")
    parser.add_argument('--index_hints', action='store_true')
    
    # sort
    parser.add_argument('--uniform_distribution_sort_data', action='store_true')
    parser.add_argument("--extra_path", type=str, default=None)
    
    # tokenize
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument("-tt", "--tokenizer_type", type=str, default="pad")
    
    # ============== 新增参数 ==============
    parser.add_argument('--weighted', action='store_true',
                        help="若指定则使用 weighted_method_gen 生成数据 (训练/测试分布可不同)")
    parser.add_argument('--is_test_distribution', action='store_true',
                        help="若指定则表示这是测试集分布, digit_weights = [1,1,1,...]")
    parser.add_argument('--min_required_digit_len', default=0, type=int,
                        help="若>0, 仅保留 max(len1,len2) > 此值 的样本")
    
    FLAGS = parser.parse_args()
    random.seed(FLAGS.seed)

    if FLAGS.no_carry_addition and FLAGS.op != '+':
        print("no carries is only for addition")
        exit()
        
    # 优先级1: bucket模式
    if FLAGS.bucket:
        index_hints = "_with_index_hints_circular" if FLAGS.index_hints else ""
        folder_name = f"{FLAGS.op}_bucket_method_n_{FLAGS.n}_m_{FLAGS.m}_{FLAGS.limit}_p_{str(FLAGS.p).replace('.','')}{'_reverse_ans' if FLAGS.reverse_answer else ''}{'_reverse_all' if FLAGS.reverse_all else ''}{'_keep_0_for_len_1' if FLAGS.keep_0_for_len_1 else ''}{index_hints}"
        if FLAGS.no_carry_addition:
            folder_name = FLAGS.dir_name
        bucket_method_main(FLAGS.n, FLAGS.m, FLAGS.op, FLAGS.limit,
                           folder_name, FLAGS.p, FLAGS.no_carry_addition,
                           FLAGS.reverse_answer, reverse_all=FLAGS.reverse_all,
                           keep_0_for_len_1=FLAGS.keep_0_for_len_1,
                           Flags=FLAGS)
        print("dataset made")
        character_histogram(folder_name)
        print("char histogram made")
        data_analysis_main(folder_name)
        exit()

    # 优先级2: sort任务
    if FLAGS.uniform_distribution_sort_data:
        index_hints = "_with_index_hints_circular" if FLAGS.index_hints else ""
        folder_name = f"sort_bucket_uniform_distribution_max_digits_n_{FLAGS.n}_max_length_m_{FLAGS.m}_{FLAGS.limit}_p_{str(FLAGS.p).replace('.','')}{'_reverse_all' if FLAGS.reverse_all else ''}{index_hints}"
        if FLAGS.extra_path is not None:
            folder_name = f"{FLAGS.extra_path}/{folder_name}"
        print(f"folder name = {folder_name}")

        uniform_distribution_sort_main(FLAGS, folder_name)
        FLAGS.dir_name = folder_name

    # 优先级3: 若要 tokenize
    if FLAGS.tokenize:
        if FLAGS.tokenizer_type != "sort":
            character_histogram(FLAGS.dir_name)
            print("char histogram made")
        tokenize_main(FLAGS.dir_name, FLAGS.tokenizer_type, test_split_ratio=FLAGS.test_split_ratio)
        print("tokenized")
        if FLAGS.tokenizer_type != "sort":
            token_histogram(FLAGS.dir_name, FLAGS.tokenizer_type)
            print("token histogram made")
            data_analysis_main(FLAGS.dir_name)
    else:
        # 普通模式下，若指定 --weighted 则走 weighted_method_gen
        if FLAGS.weighted:
            print("[Info] Using weighted_method_gen to create dataset ...")
            dataset = weighted_method_gen(
                max_digit_len = max(FLAGS.n, FLAGS.m),
                operation = FLAGS.op,
                limit = FLAGS.num_samples,
                p=FLAGS.p,
                no_carry_addition = FLAGS.no_carry_addition,
                reverse_answer = FLAGS.reverse_answer,
                reverse_all = FLAGS.reverse_all,
                keep_0_for_len_1 = FLAGS.keep_0_for_len_1,
                min_required_digit_len = FLAGS.min_required_digit_len,
                is_test = FLAGS.is_test_distribution,
                Flags=FLAGS
            )

            # 打印前10行观察
            for i in range(min(len(dataset), 10)):
                print(dataset[i])

            # 保存到文件
            base_directory = "./cramming-data/data"
            os.makedirs(base_directory, exist_ok=True)
            base_directory = f"{base_directory}/arithmetic_data"
            os.makedirs(base_directory, exist_ok=True)
            folder_name = f"{base_directory}/{FLAGS.dir_name}"
            os.makedirs(folder_name, exist_ok=True)

            w_str = "_testDist" if FLAGS.is_test_distribution else "_trainDist"
            file_name = f"{FLAGS.op}_weighted_maxdigit_{max(FLAGS.n,FLAGS.m)}_limit_{FLAGS.num_samples}{w_str}_minRequired_{FLAGS.min_required_digit_len}.txt"
            file_path = os.path.join(folder_name, file_name)

            random.shuffle(dataset)
            with open(file_path, 'w') as f:
                for line in dataset:
                    f.write(line + '\n')
            print(f"created: {file_path}")
        else:
            # 否则就走原先的 generate_dataset 流程
            main_dataset_gen(FLAGS.dir_name, FLAGS.op, FLAGS.n, FLAGS.m,
                             FLAGS.num_samples, FLAGS.exact, FLAGS.keep_places,
                             FLAGS.prepend_zeros, FLAGS.reverse_answer, FLAGS.reverse_all,
                             FLAGS.p, FLAGS.no_carry_addition, FLAGS.seed,
                             interleave=FLAGS.interleave)

if __name__ == "main":
    main()