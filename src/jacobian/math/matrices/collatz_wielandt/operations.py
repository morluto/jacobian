"""Collatz-Wielandt quotient profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import CanonicalLimits, format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.collatz_wielandt._models import (
    CollatzWielandtResult,
)

__all__ = ["compute_collatz_wielandt_profile"]


def _admit_result_size(
    matrix: tuple[tuple[CanonicalRational, ...], ...],
    vector: tuple[CanonicalRational, ...],
) -> None:
    source_bytes = (
        128
        + sum(len(value.num) + len(value.den) + 16 for row in matrix for value in row)
        + sum(len(value.num) + len(value.den) + 16 for value in vector)
    )
    quotient_bytes = (len(vector) + 1) * (2 * MAX_CANONICAL_RATIONAL_DIGITS + 64)
    if source_bytes + quotient_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("matrix", "vector"),
            code="collatz_wielandt.result_bound",
            message="the complete Collatz-Wielandt profile exceeds the canonical output bound",
        )
    for row_index, row in enumerate(matrix):
        derived_digits = (
            sum(
                max(len(entry.num.lstrip("-")), len(entry.den))
                + max(len(vector[column].num.lstrip("-")), len(vector[column].den))
                for column, entry in enumerate(row)
            )
            + max(
                len(vector[row_index].num.lstrip("-")), len(vector[row_index].den)
            )
            + 2
        )
        if derived_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise OperationDomainValidationError(
                location=("matrix", "vector"),
                code="collatz_wielandt.quotient_growth",
                message=(
                    "derived Collatz-Wielandt quotient rational arithmetic exceeds "
                    f"the {MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
                ),
            )


def compute_collatz_wielandt_profile(
    matrix: tuple[tuple[CanonicalRational, ...], ...],
    vector: tuple[CanonicalRational, ...],
) -> CollatzWielandtResult:
    """Return the componentwise quotient profile (Ax)_i / x_i."""
    n = len(vector)
    if n == 0 or len(matrix) != n or any(len(row) != n for row in matrix):
        raise OperationDomainValidationError(
            location=("matrix", "vector"),
            code="collatz_wielandt.square_matrix",
            message="matrix must be nonempty, square, and aligned with the vector",
        )
    if any(value.as_fraction() < 0 for row in matrix for value in row):
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
    mat = [[matrix[i][j].as_fraction() for j in range(n)] for i in range(n)]
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
