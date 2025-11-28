import random
import os
from tqdm import tqdm
import argparse 

TOKEN_MAP = {
    'aa': '3', 'ab': '4', 'bc': '5',
    'a': '0', 'b': '1', 'c': '2'
}
PRIORITY_TOKENS = {'aa', 'ab', 'bc'}

def tokenize_string(s: str):
    """Tokenize a string using the 'longest match first' rule."""
    tokens = []
    digits = []
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] in PRIORITY_TOKENS:
            token = s[i:i+2]
            tokens.append(token)
            digits.append(TOKEN_MAP[token])
            i += 2
        else:
            token = s[i]
            tokens.append(token)
            digits.append(TOKEN_MAP[token])
            i += 1
    return tokens, digits

def create_dataset_sample(
    min_str_len: int,
    max_str_len: int,
    target_length: int,
    fixed: bool,
    test: bool = False
):
    """
    Generate a single dataset sample. 
    The logic is: determine the final length first, then distribute spaces.
    """
    # 1. Randomly generate an original string and tokenize
    str_len = random.randint(min_str_len, max_str_len)
    original_string = "".join(random.choices(['a', 'b', 'c'], k=str_len))
    tokens, digits = tokenize_string(original_string)

    if not tokens:
        return None

    # 2. Determine the length of tokens and the final total length
    token_chars_len = len(digits)
    if token_chars_len > target_length:
        final_len = token_chars_len
    elif fixed:
        # Fixed mode: final length must equal target_length
        final_len = target_length
    else:
        final_len = random.randint(token_chars_len, target_length)

    # 3. Compute and distribute spaces
    total_spaces_to_add = final_len - token_chars_len
    # Number of gaps = number of tokens + 1 (including before the first and after the last)
    num_gaps = len(tokens) + 1
    space_counts = [0] * num_gaps
    if fixed and test:
        space_counts[-1] = total_spaces_to_add  # all spaces go to the last gap
    else:
        for _ in range(total_spaces_to_add):
            chosen_gap = random.randrange(num_gaps)
            space_counts[chosen_gap] += 1

    # 4. Build the final string
    string_builder = []
    digit_builder = []

    # Leading spaces
    string_builder.append(' ' * space_counts[0])
    digit_builder.append(' ' * space_counts[0])

    # Tokens and gaps
    for i in range(len(tokens)):
        string_builder.append(tokens[i])
        digit_builder.append(digits[i])
        string_builder.append(' ' * space_counts[i + 1])
        digit_builder.append(' ' * space_counts[i + 1])

    final_string_part = "".join(string_builder)
    final_digit_part = "".join(digit_builder)

    return f"{final_string_part}={final_digit_part}$"


def generate_dataset(
    num_samples: int,
    output_path: str,
    min_string_len: int,
    max_string_len: int,
    target_length: int,
    fixed: bool,
    test: bool
):
    """
    Main function: generate the dataset, handle files, and show progress.
    """
    mode = "Fixed length" if fixed else "Variable length"
    print(f"Starting dataset generation ({mode}, target length: {target_length})...")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        pbar = tqdm(total=num_samples, desc="Generating samples")
        generated_count = 0
        max_attempts = num_samples * 20
        attempts = 0
        while generated_count < num_samples and attempts < max_attempts:
            sample = create_dataset_sample(
                min_string_len, max_string_len, target_length, fixed, test
            )
            if sample:
                f.write(sample + '\n')
                generated_count += 1
                pbar.update(1)
            attempts += 1
        pbar.close()

        if generated_count < num_samples:
            print(f"\nWarning: Only generated {generated_count}/{num_samples} samples.")
            print("This may be due to overly strict parameters (e.g., target_length too small).")
            print("Try loosening the parameters.")
        else:
            print(f"\nSuccess! Dataset saved to: {output_path}")


# --- Main entry ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate datasets for the special tokenization task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--fixed', action='store_true',
                        help='Use fixed-length mode. If set, all samples will have total length exactly equal to --target-len.')
    parser.add_argument('--min-len', type=int, default=1,
                        help='Minimum length of the original string (abc...).')
    parser.add_argument('--max-len', type=int, default=10,
                        help='Maximum length of the original string (abc...).')
    parser.add_argument('--target-len', type=int, default=20,
                        help='Target total length (characters + spaces). In fixed mode, it is exact. '
                             'In variable mode, it is the maximum length. Set to 0 for compact mode (no extra spaces).')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of samples to generate.')
    parser.add_argument('--output', type=str, default='tokenization_dataset.txt',
                        help='Path to save the generated dataset.')
    parser.add_argument('--test', action='store_true',
                        help='If set, generate test mode data (equal length weights). Otherwise, training mode (increasing length weights).')

    args = parser.parse_args()

    generate_dataset(
        num_samples=args.samples,
        output_path=args.output,
        min_string_len=args.min_len,
        max_string_len=args.max_len,
        target_length=args.target_len,
        fixed=args.fixed,
        test=args.test
    )
