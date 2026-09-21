"""Shared private exact dense-matrix primitives.

These helpers own backend conversion only. Mathematical admission and result
semantics remain with each calling domain owner.
"""

from __future__ import annotations

from fractions import Fraction


def rational_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Multiply two nonempty composable rational matrices through FLINT."""

    from flint import fmpq, fmpq_mat

    product = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in left]
    ) * fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in right]
    )
    return tuple(
        tuple(
            Fraction(int(product[row, column].p), int(product[row, column].q))
            for column in range(product.ncols())
        )
        for row in range(product.nrows())
    )


def prime_field_product(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
    prime: int,
) -> tuple[tuple[int, ...], ...]:
    """Multiply two nonempty composable matrices over ``GF(prime)`` through FLINT."""

    from flint import nmod_mat

    left_backend = nmod_mat(
        len(left), len(left[0]), [value for row in left for value in row], prime
    )
    right_backend = nmod_mat(
        len(right), len(right[0]), [value for row in right for value in row], prime
    )
    product = left_backend * right_backend
    return tuple(
        tuple(int(product[row, column]) for column in range(product.ncols()))
        for row in range(product.nrows())
    )


def inverse_unimodular(matrix: list[list[int]]) -> list[list[int]]:
    """Invert one square unimodular integer matrix exactly."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("unimodular inverse requires a square matrix")
    if size == 0:
        return []

    from sympy import ZZ
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

    domain = DomainMatrix.from_list_sympy(size, size, matrix).convert_to(ZZ)
    try:
        numerator, denominator = domain.inv_den()
    except DMNonInvertibleMatrixError as exc:
        raise ValueError("matrix is singular") from exc
    numerator, denominator = numerator.cancel_denom(denominator)
    if denominator != ZZ.one:
        raise ValueError("matrix is not unimodular")
    return [[int(value) for value in row] for row in numerator.to_Matrix().tolist()]


__all__ = ["inverse_unimodular", "prime_field_product", "rational_product"]
