#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random

def generate_one_sample(
    min_operands: int,
    max_operands: int,
    max_digit_len: int,
    digit_choices: str,
    rng: random.Random,
):
    """
    生成一条多加样本：
      - 操作数个数 K ~ Uniform[min_operands, max_operands]
      - 每个操作数长度 L_j ~ Uniform[1, max_digit_len]
      - 每一位数字均从 digit_choices 中采样（例如 "789"）
    返回: "a1+...+aK=sum"
    """
    K = rng.randint(min_operands, max_operands)
    digits = list(digit_choices)

    operands = []
    for _ in range(K):
        length = rng.randint(1, max_digit_len)
        # 每一位从 {7,8,9} 采样，不存在前导 0 问题
        s = "".join(rng.choice(digits) for _ in range(length))
        val = int(s)
        operands.append(val)

    total = sum(operands)
    left = "+".join(str(x) for x in operands)
    return f"{left}={total}"

def generate_dataset(
    num_samples: int,
    min_operands: int = 6,
    max_operands: int = 10,
    max_digit_len: int = 5,
    digit_choices: str = "789",
    seed: int = 42,
):
    """
    生成多条高进位、多加数的测试样本。
    """
    rng = random.Random(seed)
    data = []
    for _ in range(num_samples):
        line = generate_one_sample(
            min_operands=min_operands,
            max_operands=max_operands,
            max_digit_len=max_digit_len,
            digit_choices=digit_choices,
            rng=rng,
        )
        data.append(line)
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Generate high-carry, high-operand-count multi-addition dataset."
    )
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output file path, e.g., data/high_carry_multiadd/test_highcarry.txt")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of samples to generate.")
    parser.add_argument("--min_operands", type=int, default=6,
                        help="Minimum number of operands (inclusive). Default: 6")
    parser.add_argument("--max_operands", type=int, default=10,
                        help="Maximum number of operands (inclusive). Default: 10")
    parser.add_argument("--max_digit_len", type=int, default=5,
                        help="Maximum digits per operand. Default: 5 (matches training range).")
    parser.add_argument("--digit_choices", type=str, default="789",
                        help="Digits to sample from, as a string. Default: '789'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Default: 42")

    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if args.min_operands < 1 or args.max_operands < args.min_operands:
        raise ValueError("operand count range is invalid.")
    if args.max_digit_len < 1:
        raise ValueError("max_digit_len must be at least 1.")
    if not args.digit_choices or any(ch for ch in args.digit_choices if not ch.isdigit()):
        raise ValueError("digit_choices must be a non-empty string of digits, e.g. '789'.")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    data = generate_dataset(
        num_samples=args.num_samples,
        min_operands=args.min_operands,
        max_operands=args.max_operands,
        max_digit_len=args.max_digit_len,
        digit_choices=args.digit_choices,
        seed=args.seed,
    )

    with open(args.out_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")

    print(f"Saved {len(data)} samples to {args.out_path}")

if __name__ == "__main__":
    main()
