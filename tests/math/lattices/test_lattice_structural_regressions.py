"""Defining identities and work regressions for issues #3196, #3198, #3199."""

from cProfile import Profile
from fractions import Fraction
from itertools import combinations
from math import gcd

import pytest
from sympy import Matrix
from sympy.matrices.matrixbase import MatrixBase

from jacobian.math.lattices import IntegerLattice
from jacobian.math.lattices._lattice_tools import DUAL_OPERATION, RANK_GRAM_OPERATION
from jacobian.math.lattices.operations import (
    compute_dual,
    compute_rank_gram,
    compute_saturation,
)
from jacobian.math.matrices.values import MAX_MATRIX_DIMENSION


def _lattice(rows: list[list[int]]) -> IntegerLattice:
    return IntegerLattice.model_validate(
        {
            "ambient_dimension": len(rows[0]),
            "basis": {"entries": [list(row) for row in rows]},
        }
    )


@pytest.mark.parametrize(
    "rows",
    [[[1, 1], [0, 2]], [[1, 0]], [[2, 2]], [[2, 1, 0], [0, 3, 1]]]
    + [[[1, shear], [0, 2]] for shear in (-3, -1, 2, 4)],
)
def test_dual_exact_pairing_and_returned_gram(rows: list[list[int]]) -> None:
    lattice = _lattice(rows)
    result = compute_dual(lattice)
    request = DUAL_OPERATION.request_type.model_validate_json(
        '{"lattice":' + lattice.model_dump_json() + "}", strict=True
    )
    assert DUAL_OPERATION.run(request).model_dump_json() == result.model_dump_json()
    dual = [
        [Fraction(int(value.num), int(value.den)) for value in row]
        for row in result.dual_basis.entries
    ]
    for i in range(len(rows)):
        for j in range(len(rows)):
            assert sum(a * b for a, b in zip(dual[i], rows[j], strict=True)) == int(
                i == j
            )
            gram_entry = result.dual_gram.entries[i][j]
            assert sum(
                a * b for a, b in zip(dual[i], dual[j], strict=True)
            ) == Fraction(int(gram_entry.num), int(gram_entry.den))
    # Pairing alone permits an unwanted component outside the rational span.
    assert Matrix(rows + dual).rank() == len(rows)


@pytest.mark.parametrize(
    ("rows", "squared_covolume", "rational"),
    [
        ([[1, 0], [0, 1]], 1, True),
        ([[1, 1]], 2, False),
        ([[1, 0]], 1, True),
        ([[3, 4]], 25, True),
        ([[6, 8]], 100, True),
        ([[2, 2]], 8, False),
        ([[1, 3], [0, 2]], 4, True),
        ([[1, 1, 0], [0, 0, 1]], 2, False),
        ([[1, 1, 3], [0, 0, 1]], 2, False),
        ([[10**40, 1]], 10**80 + 1, False),
    ],
)
def test_covolume_rationality(
    rows: list[list[int]], squared_covolume: int, rational: bool
) -> None:
    lattice = _lattice(rows)
    result = compute_rank_gram(lattice)
    assert int(result.squared_covolume) == squared_covolume
    assert result.covolume_rational is rational
    request = RANK_GRAM_OPERATION.request_type.model_validate_json(
        '{"lattice":' + lattice.model_dump_json() + "}", strict=True
    )
    assert (
        RANK_GRAM_OPERATION.run(request).model_dump_json() == result.model_dump_json()
    )


@pytest.mark.parametrize("scale", [1, 2])
def test_saturation_does_not_enumerate_determinants(scale: int) -> None:
    """Count real determinant calls without replacing mathematical computations."""
    rows = [[scale * int(i == j) for j in range(8)] for i in range(4)]
    lattice = _lattice(rows)
    with Profile() as profile:
        result = compute_saturation(lattice)
    determinant_calls = sum(
        entry.callcount
        for entry in profile.getstats()
        if entry.code == MatrixBase.det.__code__
    )
    assert result.saturation_index == scale**4
    assert determinant_calls <= 1


@pytest.mark.parametrize("scale", [1, -2])
def test_saturation_at_axis_boundary(scale: int) -> None:
    rank = MAX_MATRIX_DIMENSION // 2
    primitive = [
        [int(i == j) for j in range(MAX_MATRIX_DIMENSION)] for i in range(rank)
    ]
    rows = [[scale * value for value in row] for row in primitive]
    # An integer row shear preserves the lattice and defeats a diagonal-only fix.
    rows[0] = [a + 3 * b for a, b in zip(rows[0], rows[1], strict=True)]
    result = compute_saturation(_lattice(rows))
    saturated = Matrix(
        [[int(v) for v in row] for row in result.saturated_basis.entries]
    )
    inclusion = Matrix(
        [[int(v) for v in row] for row in result.inclusion_transform.entries]
    )
    assert saturated == Matrix(primitive)
    assert inclusion * saturated == Matrix(rows)
    assert result.saturation_index == abs(scale) ** rank


@pytest.mark.parametrize("rows", [[[2, 4, 6]], [[2, 1, 3], [0, -3, 3]]])
def test_saturation_index_agrees_with_small_minor_oracle(rows: list[list[int]]) -> None:
    basis = Matrix(rows)
    result = compute_saturation(_lattice(rows))
    saturated = Matrix(
        [[int(v) for v in row] for row in result.saturated_basis.entries]
    )
    inclusion = Matrix(
        [[int(v) for v in row] for row in result.inclusion_transform.entries]
    )
    assert inclusion * saturated == basis
    assert basis.col_join(saturated).rank() == len(rows)
    minors = combinations(range(basis.cols), basis.rows)
    assert result.saturation_index == gcd(
        *(int(basis[:, list(c)].det()) for c in minors)
    )
    sat_minors = combinations(range(saturated.cols), saturated.rows)
    assert gcd(*(int(saturated[:, list(c)].det()) for c in sat_minors)) == 1
