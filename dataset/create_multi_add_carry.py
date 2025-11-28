#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys
import argparse

# =========================
# 采样工具（保持你的原接口）
# =========================
def sample_by_weights(is_test, max_value, split='operand_length'):
    if split == 'operand_length':
        start_value = 1
        values = list(range(start_value, max_value + 1))
    elif split == 'operand_count':
        start_value = 2
        values = list(range(start_value, max_value + 1))
    else:
        raise ValueError(f"Unknown split {split!r}")

    if not values:
        if split == 'operand_length': return 1
        elif split == 'operand_count': return 2

    if is_test:
        weights = [1] * len(values)
    else:
        weights = list(range(1, len(values) + 1))

    return random.choices(values, weights=weights, k=1)[0]

# =========================
# 新增：窗口长度计算与列强制函数
# =========================
def choose_window_len(numbers_str, window_mode, window_size):
    """
    从最低位开始的窗口长度：
    - min_len: 取样本中最短操作数位数
    - fixed:   取 window_size，但不超过最短操作数位数
    """
    shortest = min(len(s) for s in numbers_str)
    if window_mode == "min_len":
        return shortest
    # fixed
    if window_size is None or window_size <= 0:
        return 0
    return max(0, min(shortest, window_size))

def force_all_operands_high_on_one_column(numbers_str, window_len, high_digits, rng=None, fixed_column_index=None):
    """
    在 [0..window_len-1] 里选一列 pos（0=个位），
    把每个操作数该列的数字都替换为 high_digits 中的一个随机数。
    若 fixed_column_index 不为 None，则使用该列（仍需 < window_len）。
    返回新的字符串列表。
    """
    if rng is None:
        rng = random
    if window_len <= 0:
        return numbers_str

    if fixed_column_index is None:
        pos = rng.randrange(window_len)
    else:
        if not (0 <= fixed_column_index < window_len):
            # 列不可用，直接返回原样
            return numbers_str
        pos = fixed_column_index

    highs = [ch for ch in str(high_digits) if ch.isdigit()]

    out = []
    for s in numbers_str:
        # idx 是从左到右的下标
        idx = len(s) - 1 - pos
        # 安全检查：如果某个数位数比窗口短（理论上不会，因为 window_len 已按最短截断）
        if idx < 0:
            out.append(s)
            continue
        d = rng.choice(highs)
        out.append(s[:idx] + d + s[idx+1:])
    return out

# =========================
# 主生成函数（保持原形参，末尾新增可选约束相关参数）
# =========================
def generate_multi_add_dataset(
    max_digit_len,
    max_operand_count,
    num_samples,
    is_test=False,
    reverse_all=False,
    special_mode=False,
    # 新增：窗口内强制一列为高数字
    enforce_high_digit_window=False,
    high_digits="56789",
    window_mode="min_len",        # {'min_len','fixed'}
    window_size=None,             # window_mode='fixed' 时使用
    fixed_column_index=None       # 若给出，则固定选中列（0=个位），必须 < 实际窗口长度
):
    """
    生成多加任务数据集：
      - 位数：1..max_digit_len（按权重采样或等权）
      - 加数个数：2..max_operand_count（按权重采样或等权）
      - special_mode=True: operand_count=max_operand_count，且至少一个操作数位数=max_digit_len
      - enforce_high_digit_window=True: 窗口内选一列，使所有 operand 在该列都来自 high_digits
    返回 ["x1+x2+...+xN=y", ...]
    """
    dataset = []

    if max_operand_count < 1:
        raise ValueError(f"max_operand_count must be at least 1, got {max_operand_count}")
    if max_digit_len < 1:
        raise ValueError(f"max_digit_len must be at least 1, got {max_digit_len}")

    if enforce_high_digit_window:
        if window_mode not in ("min_len", "fixed"):
            raise ValueError("window_mode must be 'min_len' or 'fixed'")
        if window_mode == "fixed":
            if window_size is None or window_size <= 0:
                raise ValueError("window_size must be a positive int when window_mode='fixed'")
        if not high_digits or any((not ch.isdigit()) for ch in str(high_digits)):
            raise ValueError("high_digits must be a non-empty string of digits")

    generated_count = 0
    attempts = 0
    max_attempts = num_samples * 20  # 防爆刷

    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1

        # 1) 采样加数个数
        if special_mode:
            operand_count = max_operand_count
        else:
            operand_count = sample_by_weights(is_test, max_operand_count, split='operand_count')

        numbers_str = []
        numbers_val = []

        # 2) 采样各操作数长度并生成数值
        idx_force_max = -1
        if special_mode and operand_count > 0:
            idx_force_max = random.randint(0, operand_count - 1)

        valid_operands_generated = 0
        for i in range(operand_count):

            d_len = sample_by_weights(is_test, max_digit_len, split='operand_length')

            if d_len < 1:
                continue

            if d_len == 1:
                low, high = 1, 9
            else:
                low = 10 ** (d_len - 1)
                high = 10 ** d_len - 1

            if low > high:
                continue

            val = random.randint(low, high)
            s = str(val)

            numbers_val.append(val)
            numbers_str.append(s)
            valid_operands_generated += 1

        if valid_operands_generated != operand_count or not numbers_str:
            continue

        # 3) 窗口内选择一列，强制所有 operand 该列为 high_digits
        if enforce_high_digit_window:
            wlen = choose_window_len(numbers_str, window_mode, window_size)
            if wlen == 0:
                continue
            numbers_str = force_all_operands_high_on_one_column(
                numbers_str,
                window_len=wlen,
                high_digits=high_digits,
                rng=random,
                fixed_column_index=fixed_column_index
            )
            # 用修改后的字符串重建数值
            numbers_val = [int(s) for s in numbers_str]

        # 4) 组装输出；reverse_all 仅反转字符串表现
        y_val = sum(numbers_val)
        left_part = "+".join(numbers_str)
        y_str = str(y_val)
        if reverse_all:
            left_part = "+".join(s[::-1] for s in numbers_str)
            y_str = y_str[::-1]

        dataset.append(f"{left_part}={y_str}")
        generated_count += 1

    if generated_count < num_samples:
        print(
            f"Warning: Only generated {generated_count} / {num_samples} after {attempts} attempts. "
            f"Consider relaxing constraints or increasing attempts.",
            file=sys.stderr
        )
    return dataset

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser("Generate Multi-add Data (Train or Test) with optional forced-high column in a window")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'test'],
                        help="Generate 'train' or 'test' data.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--dir_name", type=str, default="multi_add_data", help="Output directory.")
    parser.add_argument("--max_digit_len", type=int, default=5, help="Max digits per operand (1..max_digit_len).")
    parser.add_argument("--max_operand_count", type=int, default=5, help="Max number of operands (2..max_operand_count).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--reverse_all", action='store_true',
                        help="Reverse the printed strings of operands and sum.")
    parser.add_argument("--special_mode", action='store_true',
                        help="If set: operand_count=max_operand_count and at least one operand has length=max_digit_len.")

    # 新增：窗口内强制一列为高数字
    parser.add_argument("--enforce_high_digit_window", action='store_true',
                        help="Enable: pick a column within the window and set ALL operands' digit at that column to be in high_digits.")
    parser.add_argument("--high_digits", type=str, default="56789",
                        help="Digits used as 'high' (string), e.g., '56789'.")
    parser.add_argument("--window_mode", type=str, choices=["min_len", "fixed"], default="min_len",
                        help="How to set window length from LSD: shortest length (min_len) or fixed size (fixed).")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Window size when --window_mode=fixed (will be clipped by the sample's shortest length).")
    parser.add_argument("--fixed_column_index", type=int, default=None,
                        help="If set, force the chosen column index (0-based from LSD). Must be smaller than the actual window length.")

    args = parser.parse_args()

    if args.max_digit_len < 1:
        print("Error: --max_digit_len must be at least 1.", file=sys.stderr); sys.exit(1)
    if args.max_operand_count < 1:
        print("Error: --max_operand_count must be at least 1.", file=sys.stderr); sys.exit(1)
    if args.num_samples <= 0:
        print("Error: --num_samples must be positive.", file=sys.stderr); sys.exit(1)

    os.makedirs(args.dir_name, exist_ok=True)
    random.seed(args.seed)
    is_test_run = (args.mode == 'test')

    print(f"[Info] Generating {args.num_samples} samples for '{args.mode}' ...")
    data = generate_multi_add_dataset(
        max_digit_len=args.max_digit_len,
        max_operand_count=args.max_operand_count,
        num_samples=args.num_samples,
        is_test=is_test_run,
        reverse_all=args.reverse_all,
        special_mode=args.special_mode,
        enforce_high_digit_window=args.enforce_high_digit_window,
        high_digits=args.high_digits,
        window_mode=args.window_mode,
        window_size=args.window_size,
        fixed_column_index=args.fixed_column_index
    )

    if not data:
        print("Warning: No data generated.", file=sys.stderr); sys.exit(0)

    random.shuffle(data)
    out_name = "train_add.txt" if args.mode == 'train' else f"test_add_{args.max_digit_len}_{args.max_operand_count}.txt"
    out_path = os.path.join(args.dir_name, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")
    print(f"[Info] Created {args.mode} set => {out_path} (size={len(data)})")

if __name__ == "__main__":
    main()
