import random
from collections import Counter
from utils.prepare_data import generate_scratchpad

if __name__ == "__main__":
    sp1 = generate_scratchpad(operand1=25321, operand2=375, spaces1=20, spaces2=10, reverse=True, train=True, blank_space_exact=True, simple=False)
    sp2 = generate_scratchpad(operand1=25321, operand2=375, spaces1=20, spaces2=10, reverse=True, train=True, blank_space_exact=True, simple=True)

    print(sp1)
    print(sp2)
