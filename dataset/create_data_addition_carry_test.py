#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random

def max_carry_chain_digits(num1, num2):
    s1 = str(num1)[::-1]
    s2 = str(num2)[::-1]
    m = max(len(s1), len(s2))
    carry = 0
    chain = 0
    best = 0
    for i in range(m):
        d1 = int(s1[i]) if i < len(s1) else 0
        d2 = int(s2[i]) if i < len(s2) else 0
        co = 1 if (d1 + d2 + carry) >= 10 else 0
        chain = chain + 1 if co else 0
        best = max(best, chain)
        carry = co
    return best

# ---------------- 模式 A：len2 ~ Uniform(1..n), L = n ----------------
def construct_pair_full_chain_len2_uniform(n: int, rng: random.Random):
    """
    生成 (num1, num2)，满足：
      - max(len(num1), len(num2)) = n
      - 最长连续进位链 L = n（每一位都进位）
      - len(num2) 在 1..n 等概率
    说明：
      - i=0 需要 a0 + b0 >= 10，因此 b0 不能取 0；
      - 对 i>=1，需要 ai + bi >= 9；
      - 当 i >= len2 时，bi = 0，为保持进位，ai 必须是 9（这会让 num1 高位偏 9，是不可避免的）。
    """
    len2 = rng.randint(1, n)

    a = [0]*n  # 低位在前
    b = [0]*n
    # 低位开始
    for i in range(n):
        if i >= len2:
            # num2 已经没有更高位了，bi=0
            if i == 0:
                # 不会发生：len2>=1 => i=0 < len2
                pass
            # 要求：i=0: a0+b0 >= 10（这里 i>=len2 不会是 0）；i>=1: ai + 0 + 1 >= 10 -> ai >= 9
            a[i] = 9
            b[i] = 0
        else:
            if i == 0:
                # 需要 a0 + b0 >= 10，且 b0 ∈ [1..9]（b0=0 会不可能）
                b0 = rng.randint(1, 9)
                # a0 ∈ [10 - b0 .. 9]
                lo = 10 - b0
                hi = 9
                a0 = rng.randint(lo, hi)
                a[i], b[i] = a0, b0
            elif i == len2 - 1:
                # 这一位是 num2 的最高位，需要 b 非零保证 len2 正好
                # 需要 ai + bi >= 9
                # 随机选 bi ∈ [1..9]，再给 ai 一个满足的区间
                bi = rng.randint(1, 9)
                lo = max(0, 9 - bi)  # ai >= 9 - bi
                hi = 9
                ai = rng.randint(lo, hi)
                a[i], b[i] = ai, bi
            else:
                # 中间位：需要 ai + bi >= 9，bi ∈ [0..9]
                bi = rng.randint(0, 9)
                lo = max(0, 9 - bi)
                hi = 9
                ai = rng.randint(lo, hi)
                a[i], b[i] = ai, bi

    # 保证 num1 有 n 位（最高位非零）；这里若 i >= len2，a[n-1] 已经是 9；否则强制 >=1 同时保持第 n-1 位也进位
    if a[n-1] == 0:
        # 这一位必须满足 ai + bi >= 9（因为 i>=1），调到 >=1 不会破坏 ai + bi >= 9（我们保持 ai>=max(1, 9-bi)）
        bi = b[n-1]
        a[n-1] = max(1, 9 - bi)

    num1 = int(''.join(str(d) for d in reversed(a)))
    num2 = int(''.join(str(d) for d in reversed(b)))

    # 校验：L == n，长度条件满足
    assert max(len(str(num1)), len(str(num2))) == n
    assert max_carry_chain_digits(num1, num2) == n
    return num1, num2

# ---------------- 模式 B：两数都 n 位，digits ∈ {5..9} ----------------
def construct_pair_full_chain_both_n_56789(n: int, rng: random.Random):
    """
    生成 (num1, num2)，满足：
      - len(num1) = len(num2) = n
      - 每一位从 {5,6,7,8,9} 均匀采样
      - L = n（必然成立）
    """
    digits = [5,6,7,8,9]
    a = [rng.choice(digits) for _ in range(n)]
    b = [rng.choice(digits) for _ in range(n)]
    # 最高位已经非零，天然是 n 位
    num1 = int(''.join(str(d) for d in a))
    num2 = int(''.join(str(d) for d in b))
    assert len(str(num1)) == n and len(str(num2)) == n
    assert max_carry_chain_digits(num1, num2) == n
    return num1, num2

def main():
    parser = argparse.ArgumentParser(description="Generate full-carry (L=n) addition datasets.")
    parser.add_argument("--mode", type=str, choices=["len2_uniform", "both_n_56789"], required=True,
                        help="len2_uniform: num2 长度在 1..n 等概率；both_n_56789: 两个数都为 n 位，位上取 {5..9}")
    parser.add_argument("--n", type=int, required=True, help="最大位数 n（也是链长 L）")
    parser.add_argument("--limit", type=int, default=1000, help="样本条数")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"addition_fullcarry_mode-{args.mode}_n{args.n}_limit{args.limit}.txt")

    with open(out_path, "w") as f:
        for _ in range(args.limit):
            if args.mode == "len2_uniform":
                a, b = construct_pair_full_chain_len2_uniform(args.n, rng)
            else:
                a, b = construct_pair_full_chain_both_n_56789(args.n, rng)
            if rng.random() < 0.5:
                a, b = b, a
            f.write(f"{a}+{b}={a+b}\n")

    print(f"Saved {args.limit} samples to {out_path}")

if __name__ == "__main__":
    main()
