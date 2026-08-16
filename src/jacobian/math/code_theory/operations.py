"""Code theory operations via exact enumeration."""

from __future__ import annotations

from itertools import product

__all__ = ["minimum_distance", "weight_distribution"]


def _codewords(generator_matrix, field_order):  # type: ignore[no-untyped-def]
    from flint import nmod_mat

    n_rows = len(generator_matrix)
    generator = nmod_mat(generator_matrix, field_order)
    seen = set()
    for coeffs in product(range(field_order), repeat=n_rows):
        coefficient_row = nmod_mat([list(coeffs)], field_order)
        codeword = tuple((coefficient_row * generator).tolist()[0])
        if codeword not in seen:
            seen.add(codeword)
            yield codeword


def minimum_distance(generator_matrix, field_order):  # type: ignore[no-untyped-def]
    if field_order < 2:
        raise ValueError("field_order must be at least 2")
    min_dist = float("inf")
    for codeword in _codewords(generator_matrix, field_order):  # type: ignore[no-untyped-call]
        weight = sum(1 for c in codeword if c != 0)
        if weight > 0 and weight < min_dist:
            min_dist = weight
    return int(min_dist) if min_dist != float("inf") else 0


def weight_distribution(generator_matrix, field_order):  # type: ignore[no-untyped-def]
    from collections import Counter

    weights: Counter[int] = Counter()
    for codeword in _codewords(generator_matrix, field_order):  # type: ignore[no-untyped-call]
        weight = sum(1 for c in codeword if c != 0)
        weights[weight] += 1
    return sorted(weights.items())
