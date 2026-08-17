"""Exact algebraic combinatorics kernels over Young diagrams.

The kernels operate on a partition ``lambda`` represented as a non-increasing
tuple/list of positive integers.  They use only exact integer arithmetic and
are private implementation details of the public operations.
"""

from __future__ import annotations

from math import factorial

__all__ = [
    "conjugate_partition",
    "hook_lengths",
    "standard_young_tableaux_count",
]


def conjugate_partition(parts: list[int]) -> list[int]:
    """Compute the conjugate (transpose) partition.

    The conjugate ``lambda'`` has ``lambda'_j`` equal to the number of parts of
    ``lambda`` that are at least ``j``, i.e. the column heights of the Ferrers
    diagram.
    """
    if not parts:
        return []
    max_column = parts[0]
    return [
        sum(1 for part in parts if part >= column)
        for column in range(1, max_column + 1)
    ]


def hook_lengths(parts: list[int]) -> list[list[int]]:
    """Compute the hook length of every cell of the Young diagram.

    The hook length of cell ``(i, j)`` (0-indexed) is
    ``lambda_i - j + lambda'_j - i - 1``: one arm step plus the cell itself
    plus the number of cells below it in its column.
    """
    conjugate = conjugate_partition(parts)
    hooks: list[list[int]] = []
    for row, length in enumerate(parts):
        row_hooks: list[int] = []
        for column in range(length):
            right = length - column - 1
            below = conjugate[column] - row - 1
            row_hooks.append(right + below + 1)
        hooks.append(row_hooks)
    return hooks


def standard_young_tableaux_count(parts: list[int]) -> int:
    """Count standard Young tableaux via the hook length formula.

    The number of standard Young tableaux of shape ``lambda`` is
    ``n! / prod_{(i,j) in lambda} h(i,j)`` where ``n = |lambda|`` and
    ``h(i,j)`` is the cell's hook length.
    """
    hooks = hook_lengths(parts)
    n = sum(parts)
    product = 1
    for row in hooks:
        for hook in row:
            product *= hook
    return factorial(n) // product
