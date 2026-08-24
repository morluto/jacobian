"""Shared replay helpers for canonical-form result validators."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fractions import Fraction

    from sympy import Matrix, Poly

    from jacobian.math.matrices.canonical_forms._models import (
        MonicPolynomial,
        SquareMatrixRequest,
    )


def _matrix_entries(
    request: SquareMatrixRequest,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix.entries
    )


def _matrix_from_request(request: SquareMatrixRequest) -> Matrix:
    from sympy import Matrix

    return Matrix(_matrix_entries(request))


def _polynomial_degree(polynomial: MonicPolynomial) -> int:
    return len(polynomial.coefficients) - 1


def _coefficients_of(polynomial: MonicPolynomial) -> tuple[Fraction, ...]:
    """Return a monic polynomial's increasing-degree rational coefficients."""
    return tuple(coefficient.as_fraction() for coefficient in polynomial.coefficients)


def _poly_from_monic(polynomial: MonicPolynomial, generator: str = "x") -> Poly:
    """Convert one monic coefficient list (increasing degree) to a SymPy Poly."""
    from sympy import Poly, symbols

    x = symbols(generator)
    coefficients = [
        coefficient.as_fraction() for coefficient in polynomial.coefficients
    ]
    return Poly(list(reversed(coefficients)), x, domain="QQ")
