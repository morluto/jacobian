"""Collatz-Wielandt quotient profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.collatz_wielandt._models import (
    CollatzWielandtResult,
)
from jacobian.math.matrices.values import RationalMatrix

__all__ = ["compute_collatz_wielandt_profile"]


def _admit_result_size(
    matrix: RationalMatrix,
    vector: tuple[CanonicalRational, ...],
) -> None:
    quotient_widths: list[int] = []
    for row_index, row in enumerate(matrix.entries):
        derived_digits = (
            sum(
                max(len(entry.num.lstrip("-")), len(entry.den))
                + max(len(vector[column].num.lstrip("-")), len(vector[column].den))
                for column, entry in enumerate(row)
            )
            + max(len(vector[row_index].num.lstrip("-")), len(vector[row_index].den))
            + 2
        )
        quotient_widths.append(derived_digits)
    if any(width > MAX_CANONICAL_RATIONAL_DIGITS for width in quotient_widths):
        raise OperationDomainValidationError(
            location=("matrix", "vector"),
            code="collatz_wielandt.quotient_growth",
            message=(
                "derived Collatz-Wielandt quotient rational arithmetic exceeds "
                f"the {MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
            ),
        )


def compute_collatz_wielandt_profile(
    matrix: RationalMatrix,
    vector: tuple[CanonicalRational, ...],
) -> CollatzWielandtResult:
    """Return the componentwise quotient profile (Ax)_i / x_i."""
    if not isinstance(matrix, RationalMatrix):
        raise TypeError("matrix must be a RationalMatrix")
    n = len(vector)
    if n == 0 or matrix.row_count != n or matrix.column_count != n:
        raise OperationDomainValidationError(
            location=("matrix", "vector"),
            code="collatz_wielandt.square_matrix",
            message="matrix must be nonempty, square, and aligned with the vector",
        )
    if any(value.as_fraction() < 0 for row in matrix.entries for value in row):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="collatz_wielandt.nonnegative_matrix",
            message="Collatz-Wielandt requires a nonnegative matrix",
        )
    if any(value.as_fraction() <= 0 for value in vector):
        raise OperationDomainValidationError(
            location=("vector",),
            code="collatz_wielandt.positive_vector",
            message="Collatz-Wielandt requires a strictly positive vector",
        )
    _admit_result_size(matrix, vector)
    mat = [[matrix.entries[i][j].as_fraction() for j in range(n)] for i in range(n)]
    vec = [v.as_fraction() for v in vector]

    quotients: list[Fraction] = []
    for i in range(n):
        ax_i = sum(mat[i][j] * vec[j] for j in range(n))
        quotients.append(ax_i / vec[i])

    if any(
        len(format_canonical_integer(component).lstrip("-"))
        > MAX_CANONICAL_RATIONAL_DIGITS
        for quotient in quotients
        for component in (quotient.numerator, quotient.denominator)
    ):
        raise OperationDomainValidationError(
            location=("matrix", "vector"),
            code="collatz_wielandt.quotient_bound",
            message=(
                "a derived Collatz-Wielandt quotient exceeds the canonical "
                f"rational {MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
            ),
        )

    max_q = max(quotients)

    return CollatzWielandtResult(
        matrix=matrix,
        vector=vector,
        quotients=tuple(CanonicalRational.from_fraction(q) for q in quotients),
        max_quotient=CanonicalRational.from_fraction(max_q),
    )
