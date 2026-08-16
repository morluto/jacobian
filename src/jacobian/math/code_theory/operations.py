"""Code theory operations via exact enumeration."""

from __future__ import annotations

from itertools import product

__all__ = ["minimum_distance", "weight_distribution"]


def _codewords(generator_matrix, field_order):
    n_rows = len(generator_matrix)
    n_cols = len(generator_matrix[0]) if n_rows > 0 else 0
    for coeffs in product(range(field_order), repeat=n_rows):
        codeword = []
        for j in range(n_cols):
            val = 0
            for i in range(n_rows):
                val += coeffs[i] * generator_matrix[i][j]
            codeword.append(val % field_order)
        yield tuple(codeword)


def minimum_distance(generator_matrix, field_order):
    if field_order < 2:
        raise ValueError("field_order must be at least 2")
    min_dist = float("inf")
    for codeword in _codewords(generator_matrix, field_order):
        weight = sum(1 for c in codeword if c != 0)
        if weight > 0 and weight < min_dist:
            min_dist = weight
    return int(min_dist) if min_dist != float("inf") else 0


def weight_distribution(generator_matrix, field_order):
    from collections import Counter

    weights = Counter()
    for codeword in _codewords(generator_matrix, field_order):
        weight = sum(1 for c in codeword if c != 0)
        weights[weight] += 1
    return sorted(weights.items())
