"""Exact matrix conversions at the contract-to-SymPy boundary."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.matrices import IntegerMatrix, RationalMatrix
from jacobian.contracts.matrix_operations import DeterminantRationalMatrix

__all__ = [
    "integer_matrix_from_sympy",
    "integer_matrix_to_sympy",
    "rational_from_sympy",
    "rational_matrix_from_sympy",
    "rational_matrix_to_sympy",
]


def rational_from_sympy(value: Any) -> CanonicalRational:
    """Preserve one exact SymPy rational in canonical contract form."""

    import sympy

    if not isinstance(value, sympy.Rational):
        raise ValueError("SymPy result is not an exact rational scalar")
    return CanonicalRational(
        num=format_canonical_integer(int(value.p)),
        den=format_canonical_integer(int(value.q)),
    )


def rational_matrix_to_sympy(
    matrix: RationalMatrix | DeterminantRationalMatrix,
) -> Any:
    import sympy

    return sympy.Matrix(
        [
            [sympy.Rational(value.num, value.den) for value in row]
            for row in matrix.entries
        ]
    )


def rational_matrix_from_sympy(matrix: Any) -> RationalMatrix:
    return RationalMatrix(
        entries=tuple(
            tuple(
                rational_from_sympy(matrix[row, column])
                for column in range(matrix.cols)
            )
            for row in range(matrix.rows)
        )
    )


def integer_matrix_to_sympy(matrix: IntegerMatrix) -> Any:
    import sympy

    return sympy.Matrix(
        [[parse_canonical_integer(value) for value in row] for row in matrix.entries]
    )


def integer_matrix_from_sympy(matrix: Any) -> IntegerMatrix:
    if any(value.is_Integer is not True for value in matrix):
        raise ValueError("SymPy result is not an exact integer matrix")
    return IntegerMatrix(
        entries=tuple(
            tuple(
                format_canonical_integer(int(matrix[row, column]))
                for column in range(matrix.cols)
            )
            for row in range(matrix.rows)
        )
    )
