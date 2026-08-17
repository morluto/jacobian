"""Exact matrix conversions at the contract-to-SymPy boundary."""

from __future__ import annotations

from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.matrices._operation_models import DeterminantRationalMatrix
from jacobian.math.matrices.values import IntegerMatrix, RationalMatrix, SmithNormalForm

__all__ = [
    "integer_matrix_from_sympy",
    "integer_matrix_to_sympy",
    "rational_from_sympy",
    "rational_matrix_from_sympy",
    "rational_matrix_to_sympy",
    "smith_normal_form_from_sympy",
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


def smith_normal_form_from_sympy(matrix: Any) -> SmithNormalForm:
    """Convert one SymPy Smith form into the backend-independent value."""

    import sympy

    if not isinstance(matrix, sympy.MatrixBase):
        raise TypeError("Smith backend returned a non-matrix value")
    diagonal_count = min(matrix.rows, matrix.cols)
    diagonal = tuple(int(matrix[index, index]) for index in range(diagonal_count))
    rank = next(
        (index for index, value in enumerate(diagonal) if value == 0),
        diagonal_count,
    )
    if any(diagonal[index] != 0 for index in range(rank, diagonal_count)):
        raise ValueError("Smith backend returned a nonzero factor after a zero")
    invariant_factors = tuple(abs(value) for value in diagonal[:rank])
    canonical = sympy.zeros(matrix.rows, matrix.cols)
    for index, value in enumerate(invariant_factors):
        canonical[index, index] = value
    return SmithNormalForm(
        normal_form=integer_matrix_from_sympy(canonical),
        rank=rank,
        invariant_factors=tuple(
            format_canonical_integer(value) for value in invariant_factors
        ),
    )
