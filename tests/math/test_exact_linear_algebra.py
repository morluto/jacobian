"""Independent checks for shared exact linear-algebra facts."""

from collections.abc import Sequence
from fractions import Fraction
from itertools import product

from sympy import Matrix, oo

from jacobian.math._exact_linear_algebra import symmetric_inertia


def _principal_minor_sign_changes(
    matrix: Sequence[Sequence[int | Fraction]],
) -> tuple[int, int, int]:
    """Oracle via exact characteristic roots, separate from congruence code."""

    polynomial = Matrix(matrix).charpoly().as_poly()
    assert polynomial is not None
    zero = len(matrix) - int(Matrix(matrix).rank())
    squarefree_factors = polynomial.sqf_list()[1]
    negative = sum(
        multiplicity
        * (int(factor.count_roots(-oo, 0)) - int(factor.eval(0) == 0))
        for factor, multiplicity in squarefree_factors
    )
    positive = sum(
        multiplicity
        * (int(factor.count_roots(0, oo)) - int(factor.eval(0) == 0))
        for factor, multiplicity in squarefree_factors
    )
    return positive, negative, zero


def test_symmetric_inertia_matches_exact_root_oracle_for_small_integer_matrices() -> None:
    for dimension in range(1, 4):
        positions = tuple(
            (row, column)
            for row in range(dimension)
            for column in range(row, dimension)
        )
        for entries in product((-1, 0, 1), repeat=len(positions)):
            matrix = [[0] * dimension for _ in range(dimension)]
            for (row, column), value in zip(positions, entries, strict=True):
                matrix[row][column] = value
                matrix[column][row] = value
            assert symmetric_inertia(matrix) == _principal_minor_sign_changes(matrix)


def test_symmetric_inertia_handles_rational_zero_diagonal_pivot() -> None:
    matrix = [
        [Fraction(0), Fraction(2, 3), Fraction(1, 5)],
        [Fraction(2, 3), Fraction(0), Fraction(-1, 7)],
        [Fraction(1, 5), Fraction(-1, 7), Fraction(0)],
    ]
    assert symmetric_inertia(matrix) == _principal_minor_sign_changes(matrix)
