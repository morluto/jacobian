"""Domain functions for algebraic number arithmetic in Q(sqrt(d))."""

from __future__ import annotations

from fractions import Fraction

from jacobian.math.algebraic_number_arithmetic._models import (
    AlgebraicArithmeticRequest,
    AlgebraicArithmeticResult,
    QuadraticElement,
)


def _normalize(
    rational_part: Fraction,
    irrational_part: Fraction,
    radicand: int,
) -> AlgebraicArithmeticResult:
    """Normalize Fraction components to canonical (num, den) integers."""

    return AlgebraicArithmeticResult(
        rational_part_num=rational_part.numerator,
        rational_part_den=rational_part.denominator,
        irrational_part_num=irrational_part.numerator,
        irrational_part_den=irrational_part.denominator,
        radicand=radicand,
    )


def _to_fractions(element: QuadraticElement) -> tuple[Fraction, Fraction]:
    return (
        Fraction(element.rational_part_num, element.rational_part_den),
        Fraction(element.irrational_part_num, element.irrational_part_den),
    )


def compute_algebraic_add(
    request: AlgebraicArithmeticRequest,
) -> AlgebraicArithmeticResult:
    """Compute (a + b*sqrt(d)) + (c + e*sqrt(d)) = (a+c) + (b+e)*sqrt(d)."""

    a, b = _to_fractions(request.left)
    c, e = _to_fractions(request.right)
    return _normalize(a + c, b + e, request.left.radicand)


def compute_algebraic_multiply(
    request: AlgebraicArithmeticRequest,
) -> AlgebraicArithmeticResult:
    """Compute (a + b*sqrt(d)) * (c + e*sqrt(d)) = (ac + b*e*d) + (ae + bc)*sqrt(d)."""

    a, b = _to_fractions(request.left)
    c, e = _to_fractions(request.right)
    d = request.left.radicand
    return _normalize(a * c + b * e * d, a * e + b * c, d)


__all__ = [
    "compute_algebraic_add",
    "compute_algebraic_multiply",
]
