#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate addition equations where both operands have the same digit length."
    )
    parser.add_argument("--min", type=int, required=True, help="Minimum digit length")
    parser.add_argument("--max", type=int, required=True, help="Maximum digit length")
    parser.add_argument("--limit", type=int, default=10000, help="Number of equations to generate")
    parser.add_argument("--dir", type=str, default="data", help="Output directory")
    parser.add_argument("--filename", type=str, default=None, help="Output filename")
    parser.add_argument("--test", action="store_true", help="If set, use uniform distribution between [min, max]")
    args = parser.parse_args()

    min_len = args.min
    max_len = args.max
    limit   = args.limit
    out_dir = args.dir
    is_test = args.test
    filename = args.filename

    if min_len < 1 or max_len < min_len:
        raise ValueError("Please ensure 1 <= min <= max.")

    # 1) 构建位数列表
    digit_lengths = list(range(min_len, max_len + 1))

    # 2) 根据是否 test，来决定每个位数的采样权重
    if is_test:
        # 测试集：各位数等概率
        digit_weights = [1] * len(digit_lengths)
    else:
        # 训练集：min -> 权重1, min+1 -> 权重2, ..., max -> 权重(max-min+1)
        digit_weights = [i for i in range(1, len(digit_lengths) + 1)]

    total_weight = sum(digit_weights)

    def sample_digit_length():
        """
        根据 digit_weights 做离散采样，返回一个位数 m
        """
        r = random.randint(1, total_weight)
        running_sum = 0
        for d_len, w in zip(digit_lengths, digit_weights):
            running_sum += w
            if r <= running_sum:
                return d_len
        return digit_lengths[-1]  # 理论上不应出现

    # 3) 开始生成数据
    results = []
    while len(results) < limit:
        m = sample_digit_length()

        # 确定生成区间
        start_val = 10**(m - 1)
        end_val   = 10**m - 1

        
        # 生成两个 m 位数
        operand1 = random.randint(start_val, end_val)
        operand2 = random.randint(start_val, end_val)

        # 计算结果
        answer = operand1 + operand2

        # 拼接字符串
        eq_str = f"{operand1}+{operand2}={answer}"
        results.append(eq_str)

    # 4) 随机打乱数据（可选）
    random.shuffle(results)

    # 5) 写入文件
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = (
            f"add_samedigit_min{min_len}_max{max_len}"
            f"_limit{limit}"
            f"{'_test' if is_test else '_train'}.txt"
        )
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"数据已生成: {filepath}")
    print("示例预览:")
    for preview_line in results[:10]:
        print(preview_line)

if __name__ == "__main__":
    main()

    """ 
     python dataset/create_data_addition_samelength_train.py \
        --min 50 \
        --max 500 \
        --limit 1000 \
        --test \
        --dir "data/val/exclude50"        
       """