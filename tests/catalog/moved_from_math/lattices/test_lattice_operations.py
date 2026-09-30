"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/lattices/test_lattice_operations.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from itertools import combinations, permutations
from math import gcd

from jacobian.math.lattices._models import (
    IntegerLattice,
)
from jacobian.math.matrices.values import (
    RationalMatrix,
)


def _lattice(ambient: int, basis: list[list[int]]) -> IntegerLattice:
    return IntegerLattice.model_validate(
        {
            "ambient_dimension": ambient,
            "basis": {"entries": [list(row) for row in basis]},
        }
    )


def _identity_entries(order: int) -> list[list[int]]:
    return [[int(row == column) for column in range(order)] for row in range(order)]


def _minor_determinant(
    matrix: list[list[int]], rows: tuple[int, ...], columns: tuple[int, ...]
) -> int:
    total = 0
    for ordering in permutations(range(len(rows))):
        inversions = sum(
            ordering[left] > ordering[right]
            for left in range(len(ordering))
            for right in range(left + 1, len(ordering))
        )
        total += (-1) ** inversions * reduce(
            lambda product, index: (
                product * matrix[rows[index]][columns[ordering[index]]]
            ),
            range(len(rows)),
            1,
        )
    return total


def _independent_invariant_factors(matrix: list[list[int]]) -> list[int]:
    row_count = len(matrix)
    column_count = len(matrix[0]) if row_count else 0
    previous = 1
    factors: list[int] = []
    for order in range(1, min(row_count, column_count) + 1):
        divisor = 0
        for rows in combinations(range(row_count), order):
            for columns in combinations(range(column_count), order):
                divisor = gcd(divisor, abs(_minor_determinant(matrix, rows, columns)))
        if divisor == 0:
            break
        factors.append(divisor // previous)
        previous = divisor
    return factors


def _rational_entries(matrix: RationalMatrix) -> list[list[Fraction]]:
    return [[entry.as_fraction() for entry in row] for row in matrix.entries]


def test_all_new_operations_registered_in_catalog() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    expected = {
        "lattice.rank_gram.compute",
        "lattice.canonical_basis.compute",
        "lattice.dual.compute",
        "lattice.saturation.compute",
        "lattice.sublattice_index.compute",
        "lattice.discriminant_group.compute",
        "lattice.orthogonal_complement.compute",
        "lattice.direct_sum.compute",
        "lattice.orthogonal_sum.compute",
    }
    assert expected <= ids
