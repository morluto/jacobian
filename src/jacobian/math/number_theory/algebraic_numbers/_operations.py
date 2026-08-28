"""Domain functions for algebraic number arithmetic in Q(sqrt(d))."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers._models import (
    _MAX_RESULT_DIGITS,
    AlgebraicAdditionRequest,
    AlgebraicMultiplicationRequest,
)
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue


def _fits(value: Fraction) -> bool:
    return (
        len(str(abs(value.numerator))) <= _MAX_RESULT_DIGITS
        and len(str(value.denominator)) <= _MAX_RESULT_DIGITS
    )


def _reject(operation: str) -> None:
    raise OperationDomainValidationError(
        location=("left", "right"),
        code=f"algebraic_number_arithmetic.{operation}_result_exceeds_bound",
        message=(
            f"operands would produce an {operation} result exceeding the "
            f"{_MAX_RESULT_DIGITS}-digit canonical rational bound"
        ),
    )


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
    if not _fits(a + c) or not _fits(b + e):
        _reject("addition")
    return _normalize(a + c, b + e, request.left.radicand)


def compute_algebraic_multiply(
    request: AlgebraicMultiplicationRequest,
) -> RealQuadraticValue:
    """Compute (a + b*sqrt(d)) * (c + e*sqrt(d)) = (ac + b*e*d) + (ae + bc)*sqrt(d)."""

    a, b = _to_fractions(request.left)
    c, e = _to_fractions(request.right)
    d = request.left.radicand
    if not _fits(a * c + b * e * d) or not _fits(a * e + b * c):
        _reject("multiplication")
    return _normalize(a * c + b * e * d, a * e + b * c, d)


__all__ = [
    "compute_algebraic_add",
    "compute_algebraic_multiply",
]
