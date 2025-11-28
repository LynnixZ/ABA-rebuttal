#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random

def sample_matrix_size(is_test, min_array_len, max_array_len):
    """
    根据 is_test 决定矩阵大小分布：
      - 若 is_test=False，则矩阵大小分布为 [min_array_len, ..., max_array_len]，
        权重依次增大（类似 [1,2,3,...] 的做法）。
      - 若 is_test=True，则各个大小等权。
    """
    candidates = list(range(min_array_len, max_array_len + 1))
    if is_test:
        # 等权
        weights = [1] * len(candidates)
    else:
        # 递增权重 (如矩阵行列数=3的权重比2更大)
        # 这里的逻辑可根据需要调整
        weights = [i for i in range(1, len(candidates) + 1)]

    total_weight = sum(weights)
    r = random.randint(1, total_weight)
    cumsum = 0
    for size, w in zip(candidates, weights):
        cumsum += w
        if r <= cumsum:
            return size
    return max_array_len  # 理论上不会到这里

def generate_random_matrix(n):
    """
    生成一个 n x n 的矩阵，每个元素都是 [0..9] 的随机整数。
    返回一个 list，长度为 n，每个元素形如 '123'。
    比如 n=3 => ['123','456','709'] 等等
    """
    matrix_rows = []
    for _ in range(n):
        row_digits = [str(random.randint(0, 9)) for _ in range(n)]
        matrix_rows.append("".join(row_digits))
    return matrix_rows

def transpose_matrix(matrix_rows):
    """
    对一个矩阵（行表示形式，如 ['123','456','789']）做转置，
    并返回转置后的行表示。
    例如 ['123','456','789'] => ['147','258','369']
    """
    n = len(matrix_rows)
    # matrix_rows[i] 是第 i 行的字符串
    # 转成二维数组再转置，或者直接用 zip
    mat_2d = [list(row_str) for row_str in matrix_rows]
    # zip(*mat_2d) 可以得到转置后的列，每列也是一个 list
    transposed_2d = list(zip(*mat_2d))
    # 再把转置后的列拼接成字符串
    transposed_rows = ["".join(col) for col in transposed_2d]
    return transposed_rows

def generate_matrix_transpose_dataset(
    min_array_len,
    max_array_len,
    num_samples,
    is_test=False
):
    """
    生成矩阵转置任务数据：
      - min_array_len ~ max_array_len: 矩阵的行/列数范围
      - num_samples: 样本数量
      - is_test: 控制矩阵大小的分布（等权 or 递增）
    返回：list，每个元素格式形如： '111,222,333=123,123,123'
    """
    dataset = []
    for _ in range(num_samples):
        # 1) 确定矩阵大小
        n = sample_matrix_size(is_test, min_array_len, max_array_len)

        # 2) 生成 n x n 随机矩阵
        matrix_rows = generate_random_matrix(n)

        # 3) 转置
        matrix_rows_transposed = transpose_matrix(matrix_rows)

        # 4) 拼接左、右部分
        #   左边：每行用逗号分隔，例如 "111,222,333"
        #   右边：转置结果也用逗号分隔，例如 "123,123,123"
        left_part = ",".join(matrix_rows)
        right_part = ",".join(matrix_rows_transposed)

        dataset.append(f"{left_part}={right_part}")

    return dataset

def main():
    parser = argparse.ArgumentParser("Generate Square Matrix Transpose Data")

    parser.add_argument("--dir_name", type=str, default="matrix_transpose_data",
                        help="输出文件夹名")
    parser.add_argument("--min_array_len", type=int, default=2,
                        help="最小矩阵维度")
    parser.add_argument("--max_array_len", type=int, default=5,
                        help="最大矩阵维度")
    parser.add_argument("--train_limit", type=int, default=1000,
                        help="训练集样本数")
    parser.add_argument("--test_limit", type=int, default=200,
                        help="测试集样本数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    os.makedirs(args.dir_name, exist_ok=True)
    random.seed(args.seed)

    # 1) 生成训练集 => is_test=False => 矩阵大小分布为 [min_array_len..max_array_len]，权重递增
    train_data = generate_matrix_transpose_dataset(
        min_array_len=args.min_array_len,
        max_array_len=args.max_array_len,
        num_samples=args.train_limit,
        is_test=False
    )
    random.shuffle(train_data)
    train_file = os.path.join(args.dir_name, "train_transpose.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line + "\n")
    print(f"[Info] Created train set => {train_file} (size={len(train_data)})")

    # 2) 生成测试集 => is_test=True => 矩阵大小分布为 [min_array_len..max_array_len]，等权
    test_data = generate_matrix_transpose_dataset(
        min_array_len=args.min_array_len,
        max_array_len=args.max_array_len,
        num_samples=args.test_limit,
        is_test=True
    )
    random.shuffle(test_data)
    test_file = os.path.join(args.dir_name, f"test_transpose_{args.max_array_len}.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        for line in test_data:
            f.write(line + "\n")
    print(f"[Info] Created test set => {test_file} (size={len(test_data)})")

if __name__ == "__main__":
    main()
