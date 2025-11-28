#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys


def generate_m_parity_sequence(m: int, seq_len: int, rng: random.Random):
    """
    生成一条 m-parity 序列样本：
      输入 x_1..x_L (每个在 [0, m-1])
      输出 y_i = (x_1 + ... + x_i) mod m
    返回 (xs, ys)，都是 int 列表。
    """
    xs = [rng.randint(0, m - 1) for _ in range(seq_len)]
    ys = []
    acc = 0
    for x in xs:
        acc = (acc + x) % m
        ys.append(acc)
    return xs, ys


def main():
    parser = argparse.ArgumentParser(
        description="Generate an m-parity dataset: prefix sums modulo m."
    )
    parser.add_argument("--m", type=int, required=True,
                        help="Modulus m. Symbols are in {0, ..., m-1}.")
    parser.add_argument("--limit", type=int, required=True,
                        help="Number of samples to generate.")
    parser.add_argument("--seq_len_min", type=int, required=True,
                        help="Minimum sequence length (inclusive).")
    parser.add_argument("--seq_len_max", type=int, required=True,
                        help="Maximum sequence length (inclusive).")
    parser.add_argument("--out_path", type=str, default="m_parity_data.txt",
                        help="Output file path.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    args = parser.parse_args()

    if args.m <= 1:
        print("Error: --m must be >= 2.", file=sys.stderr)
        sys.exit(1)
    if args.limit <= 0:
        print("Error: --limit must be > 0.", file=sys.stderr)
        sys.exit(1)
    if args.seq_len_min <= 0:
        print("Error: --seq_len_min must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.seq_len_max < args.seq_len_min:
        print("Error: --seq_len_max must be >= --seq_len_min.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    count = 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        while count < args.limit:
            L = rng.randint(args.seq_len_min, args.seq_len_max)
            xs, ys = generate_m_parity_sequence(args.m, L, rng)

            # 这里去掉所有空格，直接拼接
            left = "".join(str(x) for x in xs)
            right = "".join(str(y) for y in ys)
            line = f"{left}={right}"
            f.write(line + "\n")
            count += 1

    print(
        f"Generated {args.limit} samples with m={args.m}, "
        f"seq_len in [{args.seq_len_min}, {args.seq_len_max}] "
        f"-> {args.out_path}"
    )


if __name__ == "__main__":
    main()


'''


python dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 10000 \
  --seq_len_min 1 \
  --seq_len_max 20 \
  --out_path data/parity/m5_20_train.txt \
  --seed 42

python dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 1000 \
  --seq_len_min 1 \
  --seq_len_max 200 \
  --out_path data/parity/m5_200_test.txt \
  --seed 42

  python  dataset/create_data_parity_scratchpad.py \
  --m 5 \
  --limit 10000 \
  --seq_len_min 1 \
  --seq_len_max 50 \
  --out_path data/parity/m5_50_train.txt \
  --seed 42

'''