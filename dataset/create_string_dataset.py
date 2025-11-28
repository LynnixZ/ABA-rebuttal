#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse
import string

def sample_length(min_len, max_len, is_test, length_weights=None):
    """
    根据是否 test 模式，从 [min_len..max_len] 中按指定权重随机采样一个长度。
    
    - 如果 is_test=True，默认等权重 [1,1,1,...]
    - 如果 is_test=False，默认递增权重 [1,2,3,...]
    - 也可手动传入 length_weights（若不为 None，则覆盖上述默认策略）
    """
    possible_lengths = list(range(min_len, max_len + 1))
    if not possible_lengths:
        return 0
        
    if length_weights is None:
        if is_test:
            # 等权重
            length_weights = [1] * len(possible_lengths)
        else:
            # 递增权重，如 [1,2,3,...]
            length_weights = list(range(1, len(possible_lengths) + 1))
            
    # 从可能的长度和对应的权重中随机选择一个长度
    chosen_length = random.choices(possible_lengths, weights=length_weights, k=1)[0]
    return chosen_length


def generate_data(operator='string_reverse',
                  min_string_length=3,
                  max_string_length=10,
                  limit=100,
                  is_test=False):
    """
    根据 operator 参数、长度采样策略和 limit，生成数据。
      - operator='string_reverse': 生成字母串并将其反转，如 'abcdef=fedcba'
      - operator='char_shift':     将字母串的每个字符后移一位，如 'abcdef=bcdefg'

    :param operator: 'string_reverse' 或 'char_shift'
    :param min_string_length: 字符串最小长度
    :param max_string_length: 字符串最大长度
    :param limit: 总样本数
    :param is_test: 是否为测试模式 (决定长度分布是否等权重)
    :return: 一个包含生成的数据行的列表
    """
    data_samples = []
    # 使用 a-z 的小写字母作为字符集
    alphabet = string.ascii_lowercase
    
    for _ in range(limit):
        # 1) 按权重采样长度
        length = sample_length(min_string_length, max_string_length, is_test=is_test)

        if length == 0:
            continue

        # 2) 生成随机原始字符串
        source_str = ''.join(random.choice(alphabet) for _ in range(length))

        # 3) 根据操作符生成目标字符串
        if operator == 'string_reverse':
            target_str = source_str[::-1]
            
        elif operator == 'char_shift':
            shifted_chars = []
            for char in source_str:
                if char == 'z':
                    shifted_chars.append('a')
                else:
                    shifted_chars.append(chr(ord(char) + 1))
            target_str = "".join(shifted_chars)

        else:
            raise ValueError(f"未知的操作符 '{operator}'. "
                             f"必须是 'string_reverse' 或 'char_shift'.")

        sample_line = f"{source_str}={target_str}"
        data_samples.append(sample_line)

    return data_samples


def main():
    parser = argparse.ArgumentParser(
        description="为字符串任务 (string_reverse, char_shift) 生成数据集。"
    )
    parser.add_argument("--operator", type=str, default="string_reverse",
                        help="要生成的任务类型: 'string_reverse' 或 'char_shift'")
    parser.add_argument("--limit", type=int, default=100,
                        help="要生成的样本数量。")
    parser.add_argument("--min_string_length", type=int, default=3,
                        help="字符串的最小长度。")
    parser.add_argument("--max_string_length", type=int, default=10,
                        help="字符串的最大长度。")
    parser.add_argument("--is_test", action='store_true',
                        help="如果设置，则为测试模式，长度权重相等；否则权重递增。")
    parser.add_argument("--output_file", type=str, default="output_data/string_reverse_data.txt",
                        help="保存生成数据的文件路径。")

    args = parser.parse_args()

    # 调用生成函数
    dataset = generate_data(
        operator=args.operator,
        limit=args.limit,
        min_string_length=args.min_string_length,
        max_string_length=args.max_string_length,
        is_test=args.is_test
    )

    # 创建输出目录并写入文件
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line + '\n')

    print(f"成功! 在 '{args.output_file}' 文件中生成了 {len(dataset)} 行数据。")
    print(f"参数: operator={args.operator}, min_len={args.min_string_length}, max_len={args.max_string_length}, is_test={args.is_test}")
    
if __name__ == "__main__":
    main()