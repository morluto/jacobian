"""Exact arithmetic in real quadratic fields."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers._models import _MAX_RESULT_DIGITS
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue


def _fits(value: Fraction) -> bool:
    return (
        len(str(abs(value.numerator))) <= _MAX_RESULT_DIGITS
        and len(str(value.denominator)) <= _MAX_RESULT_DIGITS
    )


def _components(
    left: RealQuadraticValue,
    right: RealQuadraticValue,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    if left.radicand != right.radicand:
        raise OperationDomainValidationError(
            location=("right", "radicand"),
            code="algebraic_number_arithmetic.radicands_must_match",
            message="both operands must belong to the same quadratic field",
        )
    return (
        left.rational_part.as_fraction(),
        left.radical_coefficient.as_fraction(),
        right.rational_part.as_fraction(),
        right.radical_coefficient.as_fraction(),
    )


def _result(
    rational_part: Fraction,
    radical_coefficient: Fraction,
    radicand: int,
    *,
    operation: str,
) -> RealQuadraticValue:
    if not _fits(rational_part) or not _fits(radical_coefficient):
        raise OperationDomainValidationError(
            location=("left", "right"),
            code=f"algebraic_number_arithmetic.{operation}_result_exceeds_bound",
            message=(
                f"operands would produce an {operation} result exceeding the "
                f"{_MAX_RESULT_DIGITS}-digit canonical rational bound"
            ),
        )
    return RealQuadraticValue(
        rational_part=CanonicalRational.from_fraction(rational_part),
        radical_coefficient=CanonicalRational.from_fraction(radical_coefficient),
        radicand=radicand,
    )


def add_quadratic(
    left: RealQuadraticValue,
    right: RealQuadraticValue,
) -> RealQuadraticValue:
    """Return the exact sum of two values in one quadratic field."""

    a, b, c, e = _components(left, right)
    return _result(a + c, b + e, left.radicand, operation="addition")


def multiply_quadratic(
    left: RealQuadraticValue,
    right: RealQuadraticValue,
) -> RealQuadraticValue:
    """Return the exact product of two values in one quadratic field."""

    a, b, c, e = _components(left, right)
    radicand = left.radicand
    return _result(
        a * c + b * e * radicand,
        a * e + b * c,
        radicand,
        operation="multiplication",
    )


__all__ = ["add_quadratic", "multiply_quadratic"]
