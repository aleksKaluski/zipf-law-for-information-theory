"""
The way of calculating the entropy heavily relies on this:
Source:
author: Gianni Perez
"""


import pandas as pd
import math
from collections import Counter


def shannon_entropy(data):
    if not data:
        return 0.0

    symbol_counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in symbol_counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy/length, 5)


def compute_entropy_dataset(df: pd.DataFrame, columns: list):
    for column in columns:
        df[str(column + "_ent")] = df[column].apply(lambda x: shannon_entropy(x))
    return df


def print_entropy(paragraph: str = "Hello World!"):
    bits = shannon_entropy(paragraph)
    if bits:
        print(f"\nH(X) = {bits} bits. Rounded to {round(bits)} bits/symbol, ")
        print(f"it will take {len(paragraph) * round(bits)} bits to optimally encode '{paragraph}'")
        print(f"\nMetric entropy: {bits / len(paragraph):.5f}")
