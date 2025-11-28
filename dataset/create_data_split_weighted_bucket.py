#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse

#######################
# 1. 辅助函数
#######################

def has_carry(num1, num2):
    """
    检查 num1 + num2 的每一位相加是否产生进位
    """
    s1 = str(num1)[::-1]
    s2 = str(num2)[::-1]
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if int(s1[i]) + int(s2[i]) >= 10:
            return True
    return False

def generate_no_carry_addition(n, m):
    """
    生成不产生进位的两个数 (exact n, m 位)，以及它们的和
    """
    start_n = 10**(n - 1)
    end_n   = 10**n - 1
    start_m = 10**(m - 1)
    end_m   = 10**m - 1

    while True:
        num1 = random.randint(start_n, end_n)
        num2 = random.randint(start_m, end_m)
        if not has_carry(num1, num2):
            return num1, num2, num1 + num2

def pick_char_set(max_len):
    """
    用于 index_hints 的示例函数，返回一个足够长的字符列表
    """
    all_chars = [
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        '0','1','2','3','4','5','6','7','8','9','!','@','#','$','%','^','&','*','(',')','[',']','{','}','<','>',
        '?','~','β','Γ','Δ','δ','ε','ζ','η','θ','κ','Λ','λ','μ','Ξ','ξ','Π','π','Σ','ς','τ','Φ','φ','χ','Ψ','ψ','Ω','ω'
    ]
    random_start = random.randint(0, len(all_chars) - 1)
    repeat_needed = max_len - (len(all_chars) - random_start)
    chosen = all_chars[random_start:]
    while repeat_needed > 0:
        chosen += all_chars
        repeat_needed -= len(all_chars)
    return chosen[:max_len]

def hints_helper(num_str, chars):
    """
    将 num_str 的每个 digit 和 chars 对应组合
    """
    result = ""
    for c, d in zip(chars, num_str):
        result += f"{c}{d}"
    return result


#######################
# 2. 核心函数: weighted_method_gen
#######################

def weighted_method_gen(
    max_digit_len=5,
    operation='+',
    limit=1000,
    p=0,
    no_carry_addition=False,
    reverse_answer=False,
    reverse_all=False,
    keep_0_for_len_1=False,
    min_required_digit_len=0,  # <<< 如果 >0，需要 max(len1,len2) > min_required_digit_len
    is_test=False,             # <<< 新增参数，如果是测试集 => digit_weights=[1,1,...,1]
    Flags=None
):
    """
    使用自定义(或等)权重来随机生成两个操作数的位数，再产生对应运算的数据集。
      - 若 is_test=True，则 digit_weights = [1, 1, ..., 1]
      - 若 is_test=False，则 digit_weights = [1,2,...,max_digit_len]
      - 若 min_required_digit_len>0，则只有当 max(len1,len2) > min_required_digit_len 才保留该样本。
    """

    dataset = []

    # digit_lengths = [1..max_digit_len]
    digit_lengths = list(range(1, max_digit_len + 1))
    
    # 根据 is_test 来决定权重分布
    if is_test:
        # 等权重
        digit_weights = [1 for _ in digit_lengths]
    else:
        # 例如默认 [1,2,3,...,max_digit_len]
        digit_weights = [i for i in digit_lengths]

    total_weight = sum(digit_weights)

    def sample_digit_length():
        # 离散采样
        r = random.randint(1, total_weight)
        cumulative = 0
        for length, w in zip(digit_lengths, digit_weights):
            cumulative += w
            if r <= cumulative:
                return length
        return digit_lengths[-1]  # 理论上不会到此

    while len(dataset) < limit:
        # 1) 抽 operand1、operand2 的位数
        len1 = sample_digit_length()
        len2 = sample_digit_length()

        # 2) 若 min_required_digit_len>0，则要求至少一个操作数位数 > 该值
        #    => 只有当 max(len1,len2) > min_required_digit_len 时才保留
         
        if min_required_digit_len>1 and max(len1, len2) <= min_required_digit_len:
            continue

        # 3) 随机生成对应操作数
        start_i = 10**(len1 - 1)
        end_i   = 10**len1 - 1
        start_j = 10**(len2 - 1)
        end_j   = 10**len2 - 1

        if keep_0_for_len_1 and len1 == 1:
            start_i = 0
        if keep_0_for_len_1 and len2 == 1:
            start_j = 0

        if no_carry_addition and operation == '+':
            num1, num2, _ = generate_no_carry_addition(len1, len2)
        else:
            num1 = random.randint(start_i, end_i)
            num2 = random.randint(start_j, end_j)

        num1_str = str(num1)
        num2_str = str(num2)

        # 4) 计算结果
        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == 'x':
            result = num1 * num2
        else:
            raise ValueError("Invalid operation, only +,-,x supported.")

        result_str = str(result)

        # 5) 根据 flags 处理是否反转
        if reverse_answer:
            result_str = result_str[::-1]
        if reverse_all:
            result_str = result_str[::-1]
            num1_str   = num1_str[::-1]
            num2_str   = num2_str[::-1]

        # 如果需要 index hints:
        if Flags and getattr(Flags, 'index_hints', False):
            max_len_temp = max(len(result_str), len(num1_str), len(num2_str))
            chars = pick_char_set(max_len_temp)
            result_str = hints_helper(result_str, chars)
            num1_str   = hints_helper(num1_str, chars)
            num2_str   = hints_helper(num2_str, chars)
            dataset_entry = f"{num1_str}{operation}{num2_str}={result_str}"
        else:
            dataset_entry = f"{num1_str}{operation}{num2_str}={result_str}"

        # 6) 随机插入空格
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


#######################
# 3. 主函数: weighted_method_main
#######################

def weighted_method_main(
    max_digit_len,
    operation,
    limit,
    dir_name,
    p=0,
    no_carry_addition=False,
    reverse_answer=False,
    reverse_all=False,
    keep_0_for_len_1=False,
    min_required_digit_len=0,
    is_test=False,
    Flags=None
):
    """
    生成并保存加权(或等权)位数分布的数据集，
    - 若 is_test=True => digit_weights=[1,1,...,1]
    - 若 min_required_digit_len>0 => 需要 max(len1,len2)>该值才保留
    """
    dataset = weighted_method_gen(
        max_digit_len=max_digit_len,
        operation=operation,
        limit=limit,
        p=p,
        no_carry_addition=no_carry_addition,
        reverse_answer=reverse_answer,
        reverse_all=reverse_all,
        keep_0_for_len_1=keep_0_for_len_1,
        min_required_digit_len=min_required_digit_len,
        is_test=is_test,
        Flags=Flags
    )

    # 预览前10条
    print("Sample data:")
    for i in range(min(10, len(dataset))):
        print(dataset[i])

    # 存储到文件
    base_directory = "./data/aaweighted"
    os.makedirs(base_directory, exist_ok=True)

    file_name = (
        f"{operation}"
        f"_maxLen_{max_digit_len}"
        f"_limit_{limit}"
        f"{'_test' if is_test else '_train'}"
        f"_minReq_{min_required_digit_len}"
        f".txt"
    )
    folder_name = os.path.join(base_directory, dir_name)
    folder_name = os.path.join(base_directory, file_name)
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, file_name)

    random.shuffle(dataset)
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(entry + '\n')
    from create_data_split import character_histogram, data_analysis_main
    character_histogram(folder_name)
    print("char histogram made")
    data_analysis_main(folder_name) # more automated analysis
    exit()
    print(f"\nCreated dataset file: {file_path}")
    return dataset, folder_name, file_path


#######################
# 4. 如果需要 argparse 的 main()
#######################

def main():
    parser = argparse.ArgumentParser(description="Weighted digit-length dataset generator with min_required_digit_len and test mode.")
    parser.add_argument("--dir_name", type=str, required=True, help="Name of dataset folder.")
    parser.add_argument("--op", type=str, default='+', help="Operation: +, -, x")
    parser.add_argument("--max_digit_len", type=int, default=10, help="Max digit length (1..N).")
    parser.add_argument("--limit", type=int, default=10000, help="Number of samples to generate.")
    parser.add_argument("--p", type=float, default=0.0, help="Probability of inserting random spaces.")
    parser.add_argument("--no_carry_addition", action='store_true', help="If true, only generate no-carry addition.")
    parser.add_argument("--reverse_answer", action='store_true', help="Reverse the answer string.")
    parser.add_argument("--reverse_all", action='store_true', help="Reverse the inputs and the answer.")
    parser.add_argument("--keep_0_for_len_1", action='store_true', help="Allow operand=0 if length=1.")
    parser.add_argument("--index_hints", action='store_true', help="Whether to add index hints.")
    parser.add_argument("--min_required_digit_len", type=int, default=0,
                        help="If >0, require max(len1,len2) > this value to keep sample.")
    parser.add_argument("--test", action='store_true',
                        help="If true, use digit_weights=[1,1,...,1]; else use [1,2,...,max_digit_len].")

    args = parser.parse_args()

    weighted_method_main(
        max_digit_len=args.max_digit_len,
        operation=args.op,
        limit=args.limit,
        dir_name=args.dir_name,
        p=args.p,
        no_carry_addition=args.no_carry_addition,
        reverse_answer=args.reverse_answer,
        reverse_all=args.reverse_all,
        keep_0_for_len_1=args.keep_0_for_len_1,
        min_required_digit_len=args.min_required_digit_len,
        is_test=args.test,
        Flags=args
    )

if __name__ == "__main__":
    main()
#python dataset/create_data_split_weighted_bucket.py --dir_name train --op + --max_digit_len 30 --min_required_digit_len 1 --limit 1000000 --test