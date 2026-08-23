"""Domain functions for algebraic number arithmetic in Q(sqrt(d))."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.algebraic_number_arithmetic._models import (
    AlgebraicAdditionRequest,
    AlgebraicMultiplicationRequest,
)
from jacobian.math.real_quadratic import RealQuadraticValue


def _normalize(
    rational_part: Fraction,
    irrational_part: Fraction,
    radicand: int,
) -> RealQuadraticValue:
    """Normalize Fraction components to canonical bounded rationals."""

    return RealQuadraticValue(
        rational_part=CanonicalRational.from_fraction(rational_part),
        radical_coefficient=CanonicalRational.from_fraction(irrational_part),
        radicand=radicand,
    )


def _to_fractions(element: RealQuadraticValue) -> tuple[Fraction, Fraction]:
    return (
        element.rational_part.as_fraction(),
        element.radical_coefficient.as_fraction(),
    )


def compute_algebraic_add(
    request: AlgebraicAdditionRequest,
) -> RealQuadraticValue:
    """Compute (a + b*sqrt(d)) + (c + e*sqrt(d)) = (a+c) + (b+e)*sqrt(d)."""

    a, b = _to_fractions(request.left)
    c, e = _to_fractions(request.right)
    return _normalize(a + c, b + e, request.left.radicand)


def compute_algebraic_multiply(
    request: AlgebraicMultiplicationRequest,
) -> RealQuadraticValue:
    """Compute (a + b*sqrt(d)) * (c + e*sqrt(d)) = (ac + b*e*d) + (ae + bc)*sqrt(d)."""

    a, b = _to_fractions(request.left)
    c, e = _to_fractions(request.right)
    d = request.left.radicand
    return _normalize(a * c + b * e * d, a * e + b * c, d)


__all__ = [
    "compute_algebraic_add",
    "compute_algebraic_multiply",
]
