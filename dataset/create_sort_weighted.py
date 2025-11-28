#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random

def pick_char_set(length):
    """
    挑选 length 个字符作为索引，去掉了希腊字符，仅保留英文字母。
    你也可再加一些符号。
    """
    chars = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
    )
    start = random.randint(0, len(chars) - 1)
    if start + length <= len(chars):
        return chars[start:start + length]
    else:
        needed = (start + length) - len(chars)
        return chars[start:] + chars[:needed]

def sample_digit_length(is_test, max_digit_len, min_digit_len):
    """
    根据 is_test 决定位数分布：
      - is_test=False => 位数分布 = [min_digit_len, min_digit_len+1, ..., max_digit_len]，权重为递增
      - is_test=True  => 位数分布等权，每个长度权重相等
    """
    possible = list(range(min_digit_len, max_digit_len + 1))
    if is_test:
        weights = [1] * len(possible)
    else:
        weights = [i for i in range(1, len(possible) + 1)]
    
    total = sum(weights)
    r = random.randint(1, total)
    cumulative = 0
    for value, weight in zip(possible, weights):
        cumulative += weight
        if r <= cumulative:
            return value
    return max_digit_len  # 理论上不会走到这里

def generate_weighted_sort_dataset(
    max_digit_len,
    min_digit_len,
    max_array_len,
    min_array_len,
    max_array_padding_len,
    num_samples,
    is_test=False,
    reverse_all=False,
    special_mode=True
):
    """
    生成排序任务：
      - max_digit_len: 数字允许的最大位数
      - min_digit_len: 数字允许的最小位数
      - max_array_len: 一个数组最多包含的数字个数
      - min_array_len: 一个数组最少包含的数字个数
      - num_samples: 生成的样本数
      - is_test: 是否生成测试集 => 决定位数分布(等权/递增)
      - reverse_all: 是否把数字字符串反转
    """
    dataset = []
    for _ in range(num_samples):
        # 随机决定本条的“数组长度”
        arr_len = random.randint(min_array_len, max_array_len) 

        if arr_len < max_array_padding_len:
            pad_arr_len=random.randint(0,max_array_padding_len-arr_len)
        else:
            pad_arr_len=0
        
        # 为每个数字分配一个索引字符，如 a,b,c,d
        index_chars = pick_char_set(arr_len+pad_arr_len)
        index_chars_number_list = random.sample(index_chars, arr_len)
        index_chars_number_set = set(index_chars_number_list)
        index_chars_pad_set = set(index_chars) - index_chars_number_set

        # 存储 (索引字符, 数值, 数字字符串)
        arr = []
        if special_mode and  arr_len> 0:
             idx_force_max = random.randint(0, arr_len - 1)

        for idx_char in index_chars:
            if idx_char in index_chars_number_list:
                if special_mode and idx_char == index_chars_number_list[idx_force_max]:
                    d_len=max_digit_len
                else:
                    d_len = sample_digit_length(is_test, max_digit_len, min_digit_len)
                
                # 生成该位数的整数
                if d_len == 1:
                    low, high = 1, 9
                else:
                    low, high = 10**(d_len - 1), 10**d_len - 1
                number_val = random.randint(low, high)
                
                # 将数字转为字符串，若 reverse_all 为 True，则反转字符串
                num_str = str(number_val)
                if reverse_all:
                    num_str = num_str[::-1]
                
                arr.append((idx_char, number_val, num_str))
            elif idx_char in index_chars_pad_set:

                arr.append((idx_char, 0, ""))  # 填充的数字为0
        
        # 拼接左侧部分，如 "a:123,b:9999,c:56"
        
        left_part = ",".join(f"{x[0]}:{x[2]}" for x in arr)
        
        # 按数值排序并得到索引字符排序
        arr_sorted = sorted(arr, key=lambda x: x[1])
        right_part = ",".join(x[0] for x in arr_sorted)


        # 形成一条样本，格式示例: "a:123,b:9999,c:56=c,a,b"
        dataset.append(f"{left_part}={right_part}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser("Generate Weighted Sort Data (Train or Test)")
    parser.add_argument("--dir_name", type=str, default="weighted_sort_data",
                        help="输出文件夹名")
    parser.add_argument("--max_digit_len", type=int, default=5,
                        help="数字最大位数")
    parser.add_argument("--min_digit_len", type=int, default=1,
                        help="数字最小位数")
    parser.add_argument("--max_array_len", type=int, default=5,
                        help="一个数组包含的数字个数上限")
    parser.add_argument("--min_array_len", type=int, default=2,
                        help="一个数组包含的数字个数下限")
    parser.add_argument("--train_limit", type=int, default=1000,
                        help="训练集样本数")
    parser.add_argument("--test_limit", type=int, default=200,
                        help="测试集样本数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--reverse_all", action='store_true',
                        help="是否反转数字字符串")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="选择生成数据集的类型：train 或 test")
    parser.add_argument("--max_array_padding_len", type=int, default=0,
                        help="一个数组最多 pad 到的总长度")
    parser.add_argument("--special_mode", action='store_true',
                        help="heatmap选项")
    args = parser.parse_args()


    # 检查最小值与最大值的关系是否合理
    if args.min_array_len > args.max_array_len:
        raise ValueError("min_array_len 必须小于或等于 max_array_len")
    if args.min_digit_len > args.max_digit_len:
        raise ValueError("min_digit_len 必须小于或等于 max_digit_len")

    os.makedirs(args.dir_name, exist_ok=True)
    random.seed(args.seed)

    if args.mode == "train":
        # 生成训练集 => is_test=False => 位数分布为递增权重
        train_data = generate_weighted_sort_dataset(
            max_digit_len=args.max_digit_len,
            min_digit_len=args.min_digit_len,
            max_array_len=args.max_array_len,
            min_array_len=args.min_array_len,
            max_array_padding_len=args.max_array_padding_len,
            num_samples=args.train_limit,
            special_mode=False,
            is_test=False,          # 训练集
            reverse_all=args.reverse_all
        )
        random.shuffle(train_data)
        train_file = os.path.join(args.dir_name, "train_sort.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            for line in train_data:
                f.write(line + "\n")
        print(f"[Info] Created train set => {train_file} (size={len(train_data)})")
    
    elif args.mode == "test":
        # 生成测试集 => is_test=True => 位数分布等权
        test_data = generate_weighted_sort_dataset(
            max_digit_len=args.max_digit_len,
            min_digit_len=args.min_digit_len,
            max_array_len=args.max_array_len,
            min_array_len=args.min_array_len,
            num_samples=args.test_limit,
            max_array_padding_len=0,
            special_mode=args.special_mode,
            is_test=True,           # 测试集
            reverse_all=args.reverse_all
        )
        random.shuffle(test_data)
        test_file = os.path.join(args.dir_name, f"test_sort_{args.max_digit_len}digit_{args.max_array_len}array.txt")
        print(f"[Info] max_digit_len: {args.max_digit_len}")
        with open(test_file, "w", encoding="utf-8") as f:
            for line in test_data:
                f.write(line + "\n")
        print(f"[Info] Created test set => {test_file} (size={len(test_data)})")

if __name__ == "__main__":
    main()
