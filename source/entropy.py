"""
Source: https://github.com/ambron60/shannon-entropy-calculator
author: Gianni Perez
"""


import pandas as pd
import math
from collections import Counter

def shannon_entropy(data):
    if not data:
        print("Input string is empty.")
        return 0.0

    symbol_counts = Counter(data)
    length = len(data)
    # print("\nSymbol-occurrence frequencies:\n")
    for symbol, count in symbol_counts.items():
        freq = count / length
        # print(f"{symbol} --> {freq:.5f} -- {count}")
    return calculate_entropy(symbol_counts, length)

def calculate_entropy(symbol_counts, length):
    entropy = 0.0
    for count in symbol_counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy, 5)

def print_entropy(paragraph: str = "Hello World!"):
    bits = shannon_entropy(paragraph)
    if bits:
        print(f"\nH(X) = {bits} bits. Rounded to {round(bits)} bits/symbol, ")
        print(f"it will take {len(paragraph) * round(bits)} bits to optimally encode '{paragraph}'")
        print(f"\nMetric entropy: {bits / len(paragraph):.5f}")


def compute_entropy_for_paraph(paragraph: str = "Hello World!"):
    bits = shannon_entropy(paragraph)
    return round(bits / len(paragraph), 5)


def compute_entropy_dataset(df: pd.DataFrame, columns: list):
    for column in columns:
        df[str(column + "_ent")] = df[column].apply(lambda x: compute_entropy_for_paraph(x))



    return df
