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
    （注意：这里和原来一样，只检查两数共有长度的部分）
    """
    s1 = str(num1)[::-1]
    s2 = str(num2)[::-1]
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if int(s1[i]) + int(s2[i]) >= 10:
            return True
    return False


def max_carry_chain(num1, num2):
    """
    计算 num1 + num2 的最长连续进位链长度。
    这里把高位由于最后一次进位产生的“额外一位”也算在链长里。
    """
    s1 = str(num1)[::-1]
    s2 = str(num2)[::-1]
    max_len = max(len(s1), len(s2))

    carry = 0
    chain = 0
    max_chain = 0

    for i in range(max_len):
        d1 = int(s1[i]) if i < len(s1) else 0
        d2 = int(s2[i]) if i < len(s2) else 0
        s = d1 + d2 + carry
        carry_out = 1 if s >= 10 else 0

        if carry_out == 1:
            chain += 1
        else:
            chain = 0
        if chain > max_chain:
            max_chain = chain

        carry = carry_out

    # 若最高位还产生了一个进位，也算进链长
    if carry == 1:
        chain += 1
        if chain > max_chain:
            max_chain = chain

    return max_chain


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


#######################
# 2. 核心函数: weighted_method_gen
#######################

def weighted_method_gen(
    max_digit_len=5,
    operation='+',
    limit=1000,
    no_carry_addition=False,
    keep_0_for_len_1=False,
    min_required_digit_len=0,  # 如果 >0，需要 max(len1,len2) > min_required_digit_len
    is_test=False,             # 如果是测试集 => digit_weights=[1,1,...,1]
    carry_chain_min=None,      # 最短允许的最长进位链
    carry_chain_max=None,      # 最长允许的最长进位链
):
    """
    使用自定义(或等)权重来随机生成两个操作数的位数，再产生对应运算的数据集。
      - 若 is_test=True，则 digit_weights = [1, 1, ..., 1]
      - 若 is_test=False，则 digit_weights = [1,2,...,max_digit_len]
      - 若 min_required_digit_len>0，则只有当 max(len1,len2) > min_required_digit_len 才保留该样本。
      - 若 carry_chain_min / carry_chain_max 不为 None，且 op 为 '+'，
        则只保留其最长进位链长度在 [carry_chain_min, carry_chain_max] 区间内的样本
        （边界为 None 时表示不约束该侧）。
    """

    if carry_chain_min is not None and carry_chain_max is not None:
        if carry_chain_min > carry_chain_max:
            raise ValueError("carry_chain_min cannot be larger than carry_chain_max")

    dataset = []

    # digit_lengths = [1..max_digit_len]
    digit_lengths = list(range(1, max_digit_len + 1))
    
    # 根据 is_test 来决定权重分布
    if is_test:
        digit_weights = [1 for _ in digit_lengths]
    else:
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
        if min_required_digit_len > 0 and max(len1, len2) <= min_required_digit_len:
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

        # 3.5) 若需要控制 carry chain，则检查并过滤
        if operation == '+' and (carry_chain_min is not None or carry_chain_max is not None):
            chain_len = max_carry_chain(num1, num2)
            if carry_chain_min is not None and chain_len < carry_chain_min:
                continue
            if carry_chain_max is not None and chain_len > carry_chain_max:
                continue

        # 4) 计算结果
        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == 'x':
            result = num1 * num2
        else:
            raise ValueError("Invalid operation, only +,-,x supported.")

        dataset_entry = f"{num1}{operation}{num2}={result}"
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
    no_carry_addition=False,
    keep_0_for_len_1=False,
    min_required_digit_len=0,
    is_test=False,
    filename=None,
    analysis=False,
    carry_chain_min=None,
    carry_chain_max=None,
):
    """
    生成并保存加权(或等权)位数分布的数据集，
    - 若 is_test=True => digit_weights=[1,1,...,1]
    - 若 min_required_digit_len>0 => 需要 max(len1,len2)>该值才保留
    - 若设置了 carry_chain_min / carry_chain_max，则对加法样本的最长进位链做约束
    - 输出路径为 dir_name/filename
    """


    dataset = weighted_method_gen(
        max_digit_len=max_digit_len,
        operation=operation,
        limit=limit,
        no_carry_addition=no_carry_addition,
        keep_0_for_len_1=keep_0_for_len_1,
        min_required_digit_len=min_required_digit_len,
        is_test=is_test,
        carry_chain_min=carry_chain_min,
        carry_chain_max=carry_chain_max,
    )

    # 预览前10条
    print("Sample data:")
    for i in range(min(10, len(dataset))):
        print(dataset[i])

    # 目录与文件名
    os.makedirs(dir_name, exist_ok=True)

    if filename is None:
        if carry_chain_min is not None and carry_chain_max is not None:
            filename = (
                f"{operation}"
                f"_maxLen_{max_digit_len}"
                f"_limit_{limit}"
                f"{'_test' if is_test else '_train'}"
                f"_minReq_{min_required_digit_len}"
                f"_carryChain_{carry_chain_min}-{carry_chain_max}"
                f".txt"
            )
        else:
            filename = (
                f"{operation}"
                f"_maxLen_{max_digit_len}"
                f"_limit_{limit}"
                f"{'_test' if is_test else '_train'}"
            f"_minReq_{min_required_digit_len}"
            f".txt"
        )

    file_path = os.path.join(dir_name, filename)

    # 写入文件
    random.shuffle(dataset)
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(entry + '\n')

    print(f"\nCreated dataset file: {file_path}")

    # 可选分析
    if analysis:
        try:
            from create_data_split import character_histogram, data_analysis_main
            character_histogram(dir_name)
            print("char histogram made")
            data_analysis_main(dir_name)
        except ImportError:
            print("Warning: create_data_split not found, skip analysis.")

    return dataset, dir_name, file_path


#######################
# 4. argparse main()
#######################

def main():
    parser = argparse.ArgumentParser(
        description="Weighted digit-length dataset generator with optional min_required_digit_len, test mode, and carry-chain control."
    )
    parser.add_argument("--dir_name", type=str, required=True,
                        help="Output directory for dataset.")
    parser.add_argument("--op", type=str, default='+',
                        help="Operation: +, -, x")
    parser.add_argument("--max_digit_len", type=int, default=10,
                        help="Max digit length (1..N).")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Number of samples to generate.")
    parser.add_argument("--no_carry_addition", action='store_true',
                        help="If true, only generate no-carry addition (for +).")
    parser.add_argument("--keep_0_for_len_1", action='store_true',
                        help="Allow operand=0 if length=1.")
    parser.add_argument("--min_required_digit_len", type=int, default=0,
                        help="If >0, require max(len1,len2) > this value to keep sample.")
    parser.add_argument("--test", action='store_true',
                        help="If true, use digit_weights=[1,1,...,1]; else use [1,2,...,max_digit_len].")
    parser.add_argument("--filename", type=str, default=None,
                        help="Output filename (optional). If not set, use a default pattern.")
    parser.add_argument("--analysis", action='store_true',
                        help="If set, run character_histogram and data_analysis_main on the output directory.")
    parser.add_argument("--carry_chain_min", type=int, default=None,
                        help="Minimum allowed length of the longest carry chain (for +).")
    parser.add_argument("--carry_chain_max", type=int, default=None,
                        help="Maximum allowed length of the longest carry chain (for +).")

    args = parser.parse_args()

    weighted_method_main(
        max_digit_len=args.max_digit_len,
        operation=args.op,
        limit=args.limit,
        dir_name=args.dir_name,
        no_carry_addition=args.no_carry_addition,
        keep_0_for_len_1=args.keep_0_for_len_1,
        min_required_digit_len=args.min_required_digit_len,
        is_test=args.test,
        filename=args.filename,
        analysis=args.analysis,
        carry_chain_min=args.carry_chain_min,
        carry_chain_max=args.carry_chain_max,
    )

if __name__ == "__main__":
    main()
