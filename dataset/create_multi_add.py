import os
import random
import sys # Import sys for error handling
import argparse # Need argparse import

# Assume sample_by_weights function is defined as provided earlier

def generate_multi_add_dataset(
    max_digit_len,
    max_operand_count,
    num_samples,
    is_test=False,
    reverse_all=False,
    special_mode=False
):
    """
    生成多加任务数据集:
      - max_digit_len: 数字允许的最大位数
      - max_operand_count: 一条加法里最大的加数个数
      - num_samples: 要生成多少条
      - is_test: 是否生成测试集 => 决定位数、加数个数的分布
      - reverse_all: 是否把数字字符串反转
      - special_mode: 若为 True，则:
          1) operand_count = max_operand_count
          2) 至少有一个数字的位数 = max_digit_len
          3) 其余数字位数走原先的分布采样
    返回: [ "x1+x2+...+xN=y",  ... ]
    """
    dataset = []
    # Validate inputs early
    if max_operand_count < 1: # Changed from <2 to <1 as single operand might be valid
        raise ValueError(f"max_operand_count must be at least 1, got {max_operand_count}")
    if max_digit_len < 1:
        raise ValueError(f"max_digit_len must be at least 1, got {max_digit_len}")

    generated_count = 0
    attempts = 0
    max_attempts = num_samples * 2 # Set a limit to prevent infinite loops if generation is hard

    # Use while loop to ensure exactly num_samples are generated if possible
    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1
        if special_mode:
            operand_count = max_operand_count
        else:
            operand_count = sample_by_weights(is_test, max_operand_count, split='operand_count')
             # Ensure at least one operand if count sampling somehow returned < 1 (shouldn't happen with current logic starting at 2)
             # if operand_count < 1: operand_count = 1 # Adjusted based on new min count=2 logic


        numbers_str = []
        numbers_val = []

        idx_force_max = -1
        if special_mode and operand_count > 0:
             idx_force_max = random.randint(0, operand_count - 1)

        valid_operands_generated = 0 # Track valid operands for this sample
        for i in range(operand_count):
            d_len = -1
            if special_mode and (i == idx_force_max):
                d_len = max_digit_len
            else:
                 d_len = sample_by_weights(is_test, max_digit_len, split='operand_length')

            if d_len < 1:
                 # This shouldn't happen if sample_by_weights works correctly
                 print(f"Warning: Invalid digit length {d_len} sampled. Skipping operand.", file=sys.stderr)
                 continue

            # --- Generate Number (Exclude 0 for d_len=1) ---
            if d_len == 1:
                # For length 1, generate from 1 to 9 (excluding 0)
                low, high = 1, 9  # <<< MODIFICATION HERE: Changed low from 0 to 1
            else:
                # Avoid leading zero for lengths > 1
                low = 10 ** (d_len - 1)
                high = 10 ** d_len - 1

            # Check if range is valid (e.g., just in case)
            if low > high:
                 print(f"Warning: Calculated low ({low}) > high ({high}) for d_len={d_len}. Skipping operand.", file=sys.stderr)
                 continue

            val = random.randint(low, high)
            # --- End Generate Number ---

            s = str(val)
            if reverse_all:
                s = s[::-1]

            numbers_val.append(val)
            numbers_str.append(s)
            valid_operands_generated += 1

        # Check if enough valid operands were generated for this sample
        # (especially relevant if count > 0 but generation failed for all)
        if valid_operands_generated != operand_count or not numbers_str:
            # Optionally print a warning if an attempt failed significantly
            # print(f"Warning: Failed to generate sufficient operands for a sample attempt. Retrying.", file=sys.stderr)
            continue # Try generating a new sample


        # Calculate sum (only if we have numbers)
        y_val = sum(numbers_val)
        y_str = str(y_val)
        if reverse_all:
            y_str = y_str[::-1]

        # Assemble line
        left_part = "+".join(numbers_str)
        line = f"{left_part}={y_str}"
        dataset.append(line)
        generated_count += 1 # Increment count of successfully generated samples

    if generated_count < num_samples:
         print(f"Warning: Only generated {generated_count} samples out of the requested {num_samples} after {attempts} attempts. "
               "Check parameters or potential issues in generation logic.", file=sys.stderr)


    return dataset
def satisfies_high_digit_window(numbers_str, window_len, high_digits, at_least_positions=1):
    high = set(high_digits)
    ok_positions = 0
    for pos in range(window_len):  # pos=0 表示个位
        hit = False
        for s in numbers_str:
            if pos < len(s):
                if s[-1 - pos] in high:
                    hit = True
                    break
        if hit:
            ok_positions += 1
    return ok_positions >= at_least_positions

# --- Keep the sample_by_weights and main functions as they were ---
# (Make sure they are included in the file if running standalone)

def sample_by_weights(is_test, max_value, split='operand_length'):
    # (Previous implementation of sample_by_weights is fine)
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

    chosen_val = random.choices(values, weights=weights, k=1)[0]
    return chosen_val


def main():
    # (Previous implementation of main function is fine)
    parser = argparse.ArgumentParser("Generate Multi-add Data (Train or Test)")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'test'], help="Specify whether to generate 'train' or 'test' data.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate for the specified mode.")
    parser.add_argument("--dir_name", type=str, default="multi_add_data", help="Output directory name.")
    parser.add_argument("--max_digit_len", type=int, default=5, help="Maximum number of digits for any operand.")
    parser.add_argument("--max_operand_count", type=int, default=5, help="Maximum number of operands in an addition expression.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--reverse_all", action='store_true', help="Reverse the string representation of all numbers (operands and sum).")
    parser.add_argument("--special_mode", action='store_true', help="If set, forces operand_count=max_operand_count and ensures at least one operand has length max_digit_len.")
    args = parser.parse_args()

    if args.max_digit_len < 1: print(f"Error: --max_digit_len must be at least 1.", file=sys.stderr); sys.exit(1)
    # Allow max_operand_count=1 if user wants single-operand "sums"
    if args.max_operand_count < 1: print(f"Error: --max_operand_count must be at least 1.", file=sys.stderr); sys.exit(1)
    if args.num_samples <= 0: print(f"Error: --num_samples must be positive.", file=sys.stderr); sys.exit(1)

    os.makedirs(args.dir_name, exist_ok=True)
    random.seed(args.seed)
    is_test_run = (args.mode == 'test')
    dataset_name = args.mode

    if args.special_mode: print(f"[Info] Special mode enabled: operand_count={args.max_operand_count}, at least one operand len={args.max_digit_len}.")

    print(f"[Info] Generating {args.num_samples} samples for '{dataset_name}' mode...")
    generated_data = generate_multi_add_dataset(
        max_digit_len=args.max_digit_len,
        max_operand_count=args.max_operand_count,
        num_samples=args.num_samples,
        is_test=is_test_run,
        reverse_all=args.reverse_all,
        special_mode=args.special_mode
    )

    if not generated_data: print(f"Warning: No data was generated for {dataset_name} mode.", file=sys.stderr); sys.exit(0)

    random.shuffle(generated_data)

    if args.mode == 'train': output_filename = "train_add.txt"
    else: output_filename = f"test_add_{args.max_digit_len}.txt"
    output_file_path = os.path.join(args.dir_name, output_filename)

    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in generated_data: f.write(line + "\n")

    print(f"[Info] Created {dataset_name} set => {output_file_path} (size={len(generated_data)})")

if __name__ == "__main__":
    main()