"""Exact algebraic combinatorics kernels backed by SymPy IntegerPartition."""

from __future__ import annotations

from math import factorial

__all__ = [
    "conjugate_partition",
    "hook_lengths",
    "standard_young_tableaux_count",
]


def conjugate_partition(parts: list[int]) -> list[int]:
    """Compute the conjugate (transpose) partition."""
    if not parts:
        return []
    max_col = parts[0]
    result = []
    for col in range(1, max_col + 1):
        count = sum(1 for p in parts if p >= col)
        result.append(count)
    return result


def hook_lengths(parts: list[int]) -> list[list[int]]:
    """Compute hook lengths for each cell of the Young diagram."""
    conj = conjugate_partition(parts)
    hooks = []
    for i, lam in enumerate(parts):
        row_hooks = []
        for j in range(lam):
            right = lam - j - 1
            below = conj[j] - i - 1
            row_hooks.append(1 + right + below)
        hooks.append(row_hooks)
    return hooks


def standard_young_tableaux_count(parts: list[int]) -> int:
    """Count standard Young tableaux via the hook length formula."""
    hooks = hook_lengths(parts)
    n = sum(parts)
    product = 1
    for row in hooks:
        for h in row:
            product *= h
    return factorial(n) // product
