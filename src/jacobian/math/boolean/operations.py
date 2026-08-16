"""Exact Boolean truth-table operations backed by SymPy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = ["walsh_hadamard_transform"]


def walsh_hadamard_transform(truth_table: list[int]) -> list[int]:
    """Return the exact integer Walsh-Hadamard transform of a truth table.

    The truth table is a list of ``0``/``1`` values whose length is a positive
    power of two.  The result is the exact integer spectrum in Hadamard order,
    computed by SymPy's fast Walsh-Hadamard transform (``fwht``).  No
    floating-point arithmetic is involved.
    """

    from sympy.discrete.transforms import fwht

    if not truth_table:
        raise ValueError("truth table must not be empty")
    n = len(truth_table)
    if n & (n - 1) != 0:
        raise ValueError("truth table length must be a power of two")
    if any(value not in (0, 1) for value in truth_table):
        raise ValueError("truth table entries must be 0 or 1")
    return [int(coefficient) for coefficient in fwht(truth_table)]
