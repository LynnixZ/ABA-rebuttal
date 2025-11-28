#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse

def sample_length(min_len, max_len, is_test, length_weights=None):
    """
    根据是否 test 模式，从 [min_len..max_len] 中按指定权重随机采样一个长度。
    
    - 如果 is_test=True，默认等权重 [1,1,1,...]
    - 如果 is_test=False，默认递增权重 [1,2,3,...]
    - 也可手动传入 length_weights（若不为 None，则覆盖上述默认策略）
    """
    digit_lengths = list(range(min_len, max_len + 1))
    if length_weights is None:
        if is_test:
            # 等权重
            length_weights = [1] * len(digit_lengths)
        else:
            # 递增权重，如 [1,2,3,...]
            length_weights = list(range(1, len(digit_lengths) + 1))
    # 累积和随机
    total_weight = sum(length_weights)
    r = random.randint(1, total_weight)
    cum = 0
    for l, w in zip(digit_lengths, length_weights):
        cum += w
        if r <= cum:
            return l
    # 理论上不会走到这
    return digit_lengths[-1]


def generate_data(operator='parity',
                  min_digit_length=1,
                  max_digit_length=5,
                  limit=10,
                  is_test=False,
                  length_weights=None,
                  multi_digit=False,
                  operand2=5):
    """
    根据 operator 参数、长度采样策略和 limit，生成数据。
      - operator='parity':    生成二进制串+奇偶性，如 '10101011=1'
      - operator='binarysum': 生成二进制串并统计 1 的数量，如 '10101011=5'
      - operator='copy':      生成二进制串的复制，如 '1010=1010'
      - operator='reverse':   生成二进制串的反转，如 '1010=0101'
      - operator='oneDigitSort': 生成十进制串并按升序排序，如 '31415=11345'
      - operator='multiply':  生成十进制串与指定或随机乘数的乘法，如 '123*3=369'
         * 当 multi_digit=False 时，乘数固定为 operand2
         * 当 multi_digit=True 时，乘数从 [10..operand2] 之间随机
      - operator='hexadecimal': 十进制转 16 进制，如 '166=A6'

    :param operator: 'parity', 'binarysum', 'copy', 'reverse', 'oneDigitSort', 'multiply', 'hexadecimal', ...
    :param min_digit_length: 最短长度
    :param max_digit_length: 最长长度
    :param limit: 总样本数
    :param is_test: 是否为测试模式 (决定长度分布是否等权重)
    :param length_weights: 若不为 None，则使用该自定义权重(覆盖默认策略)；应与 [min_digit_length..max_digit_length] 区间长度相匹配
    :param multi_digit: 是否在 [10..operand2] 内随机一个乘数
    :param operand2: 乘数的上限或固定值
    :return: [sample_line1, sample_line2, ...]
    """
    data_samples = []
    for _ in range(limit):
        # 1) 按权重采样长度
        if operator == 'multiply' and min_digit_length == 0:
            min_digit_length = 1
        length = sample_length(min_digit_length, max_digit_length,
                               is_test=is_test,
                               length_weights=length_weights)

        if operator == 'parity':
            bin_str = ''.join(random.choice(['0', '1']) for _ in range(length))
            ones_count = bin_str.count('1')
            parity_value = ones_count % 2
            sample_line = f"{bin_str}={parity_value}"

        elif operator == 'binarysum':
            bin_str = ''.join(random.choice(['0', '1']) for _ in range(length))
            ones_count = bin_str.count('1')
            sample_line = f"{bin_str}={ones_count}"

        elif operator == 'copy':
            seq = ''.join(random.choice(['0', '1']) for _ in range(length))
            sample_line = f"{seq}={seq}"

        elif operator == 'reverse':
            seq = ''.join(random.choice(['0', '1']) for _ in range(length))
            reversed_seq = seq[::-1]
            sample_line = f"{seq}={reversed_seq}"

        elif operator == 'oneDigitSort':
            if length == 0:
                seq = "0"
            else:
                seq = ''.join(random.choice('0123456789') for _ in range(length))
            sorted_seq = ''.join(sorted(seq))
            sample_line = f"{seq}={sorted_seq}"

        elif operator == 'multiply':
            # 随机生成十进制串
            start_i = 10**(length - 1)
            end_i   = 10**length - 1
            if start_i == 0:
                start_i = 1
            seq = random.randint(start_i, end_i)

            if multi_digit:
                multiplier = random.randint(1, operand2)
            else:
                multiplier = operand2

            product = int(seq) * multiplier
            sample_line = f"{seq}*{multiplier}={product}"

        elif operator == 'hexadecimal':
            # 生成一个十进制数，并转换为 16 进制字符串
            start_i = 10**(length - 1)
            end_i = 10**length - 1
            if start_i == 0:
                start_i = 1
            num = random.randint(start_i, end_i)
            hex_str = hex(num)[2:].upper()
            sample_line = f"{num}={hex_str}"

        else:
            raise ValueError(f"Unknown operator '{operator}'. "
                             f"Must be one of: 'parity', 'binarysum', 'copy', "
                             f"'reverse', 'oneDigitSort', 'multiply', 'hexadecimal', ...")

        data_samples.append(sample_line)

    return data_samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate data for multiple tasks (parity, binarysum, copy, reverse, oneDigitSort, multiply, hexadecimal, etc.)"
    )
    parser.add_argument("--operator", type=str, default="parity",
                        help="Which task to generate: 'parity', 'binarysum', 'copy', 'reverse', 'oneDigitSort', 'multiply', 'hexadecimal', etc.")
    parser.add_argument("--min_digit_length", type=int, default=1,
                        help="Minimum sequence length.")
    parser.add_argument("--max_digit_length", type=int, default=5,
                        help="Maximum sequence length.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of samples to generate.")
    parser.add_argument("--test", action='store_true',
                        help="If set, use equal weights for lengths, else use 1..N incremental.")
    parser.add_argument("--multi_digit", action='store_true',
                        help="If set and operator='multiply', then the multiplier is randomly chosen in [10..operand2].")
    parser.add_argument("--operand2", type=int, default=5,
                        help="Multiplier's upper bound (if multi_digit=True) or fixed value (if multi_digit=False).")
    parser.add_argument("--length_weights", type=str, default=None,
                        help="Custom length weights, comma-separated. Must match the range [min_digit_length..max_digit_length]. E.g. '1,2,3,3,10'")
    parser.add_argument("--output_file", type=str, default="data/newtask/binarysum_test_10.txt",
                        help="Where to save the generated data.")

    args = parser.parse_args()

    # 如果传入了 --length_weights，就将其解析为整数列表
    if args.length_weights is not None:
        lw_str_list = args.length_weights.split(',')
        lw = [int(x.strip()) for x in lw_str_list]
        expected_len = args.max_digit_length - args.min_digit_length + 1
        if len(lw) != expected_len:
            raise ValueError(f"length_weights size {len(lw)} != expected {expected_len} "
                             f"(for range {args.min_digit_length}..{args.max_digit_length})")
    else:
        lw = None

    dataset = generate_data(
        operator=args.operator,
        min_digit_length=args.min_digit_length,
        max_digit_length=args.max_digit_length,
        limit=args.limit,
        is_test=args.test,
        length_weights=lw,
        multi_digit=args.multi_digit,
        operand2=args.operand2
    )

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line + '\n')

    print(f"Done! Generated {len(dataset)} lines in '{args.output_file}'.\n"
          f"operator={args.operator}, multi_digit={args.multi_digit}, operand2={args.operand2}, "
          f"min_len={args.min_digit_length}, max_len={args.max_digit_length}, test={args.test}, "
          f"length_weights={lw}")
    
if __name__ == "__main__":
    main()

"""  
python dataset/create_data_parity_copy.py --operator parity --min_digit_length 11 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/parity_test_20.txt
python dataset/create_data_parity_copy.py --operator binarysum --min_digit_length 11 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/binarysum_test_20.txt
python dataset/create_data_parity_copy.py --operator copy --min_digit_length 11 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/copy_test_20.txt
python dataset/create_data_parity_copy.py --operator reverse --min_digit_length 11 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/reverse_test_20.txt
python dataset/create_data_parity_copy.py --operator oneDigitSort --min_digit_length 11 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/oneDigitSort_test_20.txt

python dataset/create_data_parity_copy.py --operator multiply --min_digit_length 0 --max_digit_length 70 --limit 1000 --test --operand2 2 --output_file data/newtask/multiply_fix5.txt

python dataset/create_data_parity_copy.py --operator parity --min_digit_length 0 --max_digit_length 10 --limit 10000 --output_file data/newtask/parity_train_10.txt
python dataset/create_data_parity_copy.py --operator binarysum --min_digit_length 0 --max_digit_length 10 --limit 10000 --output_file data/newtask/binarysum_train_10.txt
python dataset/create_data_parity_copy.py --operator copy --min_digit_length 0 --max_digit_length 10 --limit 10000 --output_file data/newtask/copy_train_10.txt
python dataset/create_data_parity_copy.py --operator reverse --min_digit_length 0 --max_digit_length 10 --limit 10000 --output_file data/newtask/reverse_train_10.txt
python dataset/create_data_parity_copy.py --operator oneDigitSort --min_digit_length 0 --max_digit_length 10 --limit 10000 --output_file data/newtask/oneDigitSort_train_10.txt

python dataset/create_data_parity_copy.py --operator parity --min_digit_length 0 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/parity_test_10.txt
python dataset/create_data_parity_copy.py --operator binarysum --min_digit_length 0 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/binarysum_test_10.txt
python dataset/create_data_parity_copy.py --operator copy --min_digit_length 0 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/copy_test_10.txt
python dataset/create_data_parity_copy.py --operator reverse --min_digit_length 0 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/reverse_test_10.txt
python dataset/create_data_parity_copy.py --operator oneDigitSort --min_digit_length 0 --max_digit_length 20 --limit 1000 --test --output_file data/newtask/oneDigitSort_test_10.txt

      - operator='parity':    生成二进制串+奇偶性，如 '10101011=1'
      - operator='binarysum': 生成二进制串并统计 1 的数量，如 '10101011=5'
      - operator='copy':      生成二进制串的复制，如 '1010=1010'
      - operator='reverse':   生成二进制串的反转，如 '1010=0101'
      - operator='oneDigitSort': 生成十进制串并按升序排序，如 '31415=11345'
"""