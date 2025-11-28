#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import string

def pick_char_set(length):
    """
    Selects 'length' characters to be used as indices.
    """
    chars = (
        "0123456789"
    )
    if length > len(chars):
        raise ValueError("Requested length for character set is too large.")
    
    start = random.randint(0, len(chars) - length)
    return chars[start:start + length]

def generate_random_string(min_len, max_len):
    """
    Generates a random lowercase string with a length between min_len and max_len.
    """
    length = random.randint(min_len, max_len)
    # Using only lowercase letters for the strings to be sorted
    letters = string.ascii_lowercase 
    return "".join(random.choice(letters) for _ in range(length))

def generate_string_sort_dataset(
    max_str_len,
    min_str_len,
    max_array_len,
    min_array_len,
    max_array_padding_len,
    num_samples,
):
    """
    Generates a string sorting task dataset.
      - max_str_len: Maximum allowed length for a string value.
      - min_str_len: Minimum allowed length for a string value.
      - max_array_len: Maximum number of strings in an array.
      - min_array_len: Minimum number of strings in an array.
      - max_array_padding_len: Maximum number of padding elements.
      - num_samples: The number of samples to generate.
    """
    dataset = []
    for _ in range(num_samples):
        # Determine the array length for the current sample
        arr_len = random.randint(min_array_len, max_array_len) 

        pad_arr_len = 0
        if arr_len < max_array_padding_len:
            pad_arr_len = random.randint(0, max_array_padding_len - arr_len)
        
        # Assign a unique index character (e.g., a, b, c) to each element
        total_len = arr_len + pad_arr_len
        index_chars = pick_char_set(total_len)
        
        # Separate characters for actual values vs. padding
        index_chars_value_list = random.sample(index_chars, arr_len)
        index_chars_value_set = set(index_chars_value_list)
        index_chars_pad_set = set(index_chars) - index_chars_value_set

        # Store tuples of (index_char, string_value)
        arr = []
        for idx_char in index_chars:
            if idx_char in index_chars_value_set:
                # Generate a random string
                str_val = generate_random_string(min_str_len, max_str_len)
                arr.append((idx_char, str_val))
            elif idx_char in index_chars_pad_set:
                # Padding elements have an empty string value
                arr.append((idx_char, ""))
        
        # Construct the left part of the sample, e.g., "a:hwg,b:vaq,c:hwr"
        left_part = ",".join(f"{x[0]}:{x[1]}" for x in arr)
        
        # Sort the array lexicographically by the string value
        # Empty strings will be sorted first by default
        arr_sorted = sorted(arr, key=lambda x: x[1])
        
        # Construct the right part with sorted index characters
        right_part = ",".join(x[0] for x in arr_sorted)

        # Combine into a single sample, e.g., "a:hwg,b:vaq,c:hwr=a,c,b"
        dataset.append(f"{left_part}={right_part}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser("Generate String Sort Data (Train or Test)")
    parser.add_argument("--dir_name", type=str, default="string_sort_data",
                        help="Output directory name")
    parser.add_argument("--max_str_len", type=int, default=8,
                        help="Maximum string length")
    parser.add_argument("--min_str_len", type=int, default=3,
                        help="Minimum string length")
    parser.add_argument("--max_array_len", type=int, default=5,
                        help="Maximum number of items in the array")
    parser.add_argument("--min_array_len", type=int, default=2,
                        help="Minimum number of items in the array")
    parser.add_argument("--train_limit", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--test_limit", type=int, default=200,
                        help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Choose dataset type to generate: train or test")
    parser.add_argument("--max_array_padding_len", type=int, default=0,
                        help="Pad arrays to a total maximum length")
    args = parser.parse_args()

    # Validate min/max values
    if args.min_array_len > args.max_array_len:
        raise ValueError("min_array_len must be less than or equal to max_array_len")
    if args.min_str_len > args.max_str_len:
        raise ValueError("min_str_len must be less than or equal to max_str_len")

    os.makedirs(args.dir_name, exist_ok=True)
    random.seed(args.seed)

    if args.mode == "train":
        train_data = generate_string_sort_dataset(
            max_str_len=args.max_str_len,
            min_str_len=args.min_str_len,
            max_array_len=args.max_array_len,
            min_array_len=args.min_array_len,
            max_array_padding_len=args.max_array_padding_len,
            num_samples=args.train_limit,
        )
        random.shuffle(train_data)
        train_file = os.path.join(args.dir_name, "train_str_sort.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            for line in train_data:
                f.write(line + "\n")
        print(f"[Info] Created train set => {train_file} (size={len(train_data)})")
    
    elif args.mode == "test":
        test_data = generate_string_sort_dataset(
            max_str_len=args.max_str_len,
            min_str_len=args.min_str_len,
            max_array_len=args.max_array_len,
            min_array_len=args.min_array_len,
            max_array_padding_len=args.max_array_padding_len,
            num_samples=args.test_limit,
        )
        random.shuffle(test_data)
        test_file = os.path.join(args.dir_name, f"test_strsort_{args.max_str_len}len_{args.max_array_len}array.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            for line in test_data:
                f.write(line + "\n")
        print(f"[Info] Created test set => {test_file} (size={len(test_data)})")

if __name__ == "__main__":
    main()