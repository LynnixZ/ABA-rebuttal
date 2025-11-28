import random
from collections import Counter

def insert_spaces_at_positions(s, space_positions):
    s_list = list(s)
    offset = 0
    position_counts = Counter(space_positions)
    for pos in sorted(position_counts.keys()):
        spaces_to_insert = position_counts[pos]
        s_list.insert(pos + offset, ' ' * spaces_to_insert)
        offset += 1
    return ''.join(s_list)

def generate_fixed_space_positions(length, num_spaces_to_insert):
    total_length = length + num_spaces_to_insert
    digits_positions = sorted(random.sample(range(total_length), k=length))
    space_positions = []
    space_positions.append(digits_positions[0])
    for i in range(1, len(digits_positions)):
        space_positions.append(digits_positions[i] - digits_positions[i-1] - 1)
    space_positions.append(total_length - digits_positions[-1] - 1)
    def expand_positions(space_positions):
        result = []
        for i, count in enumerate(space_positions):
            result.extend([i] * count)
        return result
    space_position = expand_positions(space_positions)
    return space_position

def generate_scratchpad(operand1, operand2, spaces1, spaces2,reverse=True):
    op1_str = str(operand1)
    op2_str = str(operand2)
    # 将第一个操作数补0，使其长度为两操作数数字总数
    zero_pad_length = len(op1_str) + len(op2_str)
    op1_str = op1_str.zfill(zero_pad_length)
    blankspaces1 = spaces1 - zero_pad_length
    # 生成第一个操作数的空格位置并插入空格
    pos1 = generate_fixed_space_positions(len(op1_str), blankspaces1)
    spaced_op1 = insert_spaces_at_positions(op1_str, pos1)
    
    # 对第二个操作数，随机生成插入空格的数量
    blankspaces2 = random.randint(0, spaces2 - len(op2_str))
    pos2 = generate_fixed_space_positions(len(op2_str), blankspaces2)
    spaced_op2 = insert_spaces_at_positions(op2_str, pos2)

    # 生成头部行
    if reverse:
        spaced_op1 = spaced_op1[::-1]
        spaced_op2 = spaced_op2[::-1]
    header = f"${spaced_op2}*{spaced_op1}:"
    
    # 对于第二个操作数中的每个字符，生成乘法计算过程
    multiplication_lines = []
    intermediate_products = []
    digit_count = -1
    for i, digit in enumerate(spaced_op2):
        if digit.strip():
            digit_count += 1
            prod = int(op1_str) * int(digit)
            prod_str = str(prod).zfill(zero_pad_length)
            spaced_prod = insert_spaces_at_positions(prod_str, pos1)
            # 将乘积左移 digit_count 位
            shifted_value = prod * (10 ** digit_count)
            shifted_str = str(shifted_value).zfill(zero_pad_length)
            shifted_prod = insert_spaces_at_positions(shifted_str, pos1)
            if reverse:
                spaced_prod = spaced_prod[::-1]
                shifted_prod = shifted_prod[::-1]
            line = f"{digit}*{spaced_op1}={spaced_prod}>{shifted_prod},"
            multiplication_lines.append(line)
            intermediate_products.append(shifted_prod)
        else:
            line = f" *{spaced_op1}={' ' * len(spaced_op1)}>{' ' * len(spaced_op1)},"
            multiplication_lines.append(line)
            intermediate_products.append(' ' * len(spaced_op1))
    
    # 生成最终加法行
    final_product = int(op1_str) * int(op2_str)
    final_product_str = str(final_product).zfill(zero_pad_length)
    spaced_final = insert_spaces_at_positions(final_product_str, pos1)
    if reverse:
        spaced_final = spaced_final[::-1]
    addition_line = "+".join(intermediate_products) + "=" + spaced_final + "$"
    
    scratchpad = "\n".join([header] + multiplication_lines + [addition_line])
    return scratchpad

def generate_scratchpad_dataset(min_digits, max_digits, limit, spaces1, spaces2,reverse=True):
    """
    生成一个scratch pad数据集。参数说明：
    - min_digits: 操作数最少的位数
    - max_digits: 操作数最多的位数
    - limit: 生成问题的个数
    - spaces1: 第一个操作数补充空格的总数（须大于等于两操作数数字总数）
    - spaces2: 第二个操作数补充空格的总数（须大于等于操作数位数）
    返回一个包含scratch pad字符串的列表。
    """
    dataset = []
    for _ in range(limit):
        # 随机确定两个操作数的位数
        digit_count1 = random.randint(min_digits, max_digits)
        digit_count2 = random.randint(min_digits, max_digits)
        # 保证多位数的第一个数字不为0
        if digit_count1 == 1:
            op1 = random.randint(1, 9)
        else:
            op1 = random.randint(10**(digit_count1 - 1), 10**digit_count1 - 1)
        if digit_count2 == 1:
            op2 = random.randint(1, 9)
        else:
            op2 = random.randint(10**(digit_count2 - 1), 10**digit_count2 - 1)
        scratchpad = generate_scratchpad(op1, op2, spaces1, spaces2,reverse)
        dataset.append(scratchpad)
    return dataset

# 示例：生成1到3位数乘法的scratch pad数据集，共生成5个问题，
# 其中第一个操作数总空格数设为 (操作数位数总和 + 2)，第二个操作数总空格数设为 (操作数位数 + 1)
if __name__ == "__main__":
    # 可以调整spaces1和spaces2以满足补空格要求，确保spaces1 >= op1位数总数, spaces2 >= op2位数数
    dataset = generate_scratchpad_dataset(min_digits=1, max_digits=3, limit=5, spaces1=10, spaces2=6,reverse=True)
    output_dir = "data/multiply_scratchpad"
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "scratchpad_dataset.txt"
    output_filepath = os.path.join(output_dir, output_filename)
    with open(output_filename, "w", encoding="utf-8") as f:
        for data in dataset:
            f.write(data + "\n")
            f.write("-" * 40 + "\n")
    
    print(f"数据集已写入到文件 {output_filename}")