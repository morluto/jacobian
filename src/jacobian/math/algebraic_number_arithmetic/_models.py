"""Typed wire contracts for algebraic number arithmetic."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.real_quadratic import RealQuadraticValue

_MAX_RESULT_DIGITS = 256


def _fits(frac: Fraction) -> bool:
    """Check one exact rational against the result digit bound."""

    if (
        len(str(abs(frac.numerator))) > _MAX_RESULT_DIGITS
        or len(str(frac.denominator)) > _MAX_RESULT_DIGITS
    ):
        return False
    try:
        CanonicalRational.from_fraction(frac)
    except ValueError:
        return False
    return True


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by algebraic-number-arithmetic contracts."""

    return PydanticCustomError(f"algebraic_number_arithmetic.{reason}", message)


class AlgebraicArithmeticRequest(StrictModel):
    """Two elements of Q(sqrt(d)) for an exact binary operation.

    Both operands must belong to one shared quadratic field: their
    radicands must be equal (and square-free, as ``RealQuadraticValue``
    already enforces).  Each concrete operation narrows admission further
    so that its own exact result fits the 256-digit canonical rational
    bound.
    """

    left: RealQuadraticValue = Field(
        description=(
            "Operand a + b*sqrt(d).  Its radicand d must be square-free "
            "and must match the other operand's radicand exactly."
        ),
    )
    right: RealQuadraticValue = Field(
        description=(
            "Operand a + b*sqrt(d) in the same field as ``left``.  Its "
            "square-free radicand must match ``left``'s radicand exactly."
        ),
    )

    @model_validator(mode="after")
    def require_same_radicand(self) -> Self:
        if self.left.radicand != self.right.radicand:
            raise _validation_error(
                "radicands_must_match",
                "both operands must belong to the same quadratic field",
            )
        return self


class AlgebraicAdditionRequest(AlgebraicArithmeticRequest):
    """Two elements of Q(sqrt(d)) whose component-wise sum is returnable.

    Admission rejects only requests whose own addition result
    ``(a+c) + (b+e)*sqrt(d)`` would exceed the 256-digit canonical
    rational bound; multiplication growth is irrelevant here.
    """

    @model_validator(mode="after")
    def require_addition_result_within_bound(self) -> Self:
        a, b = (
            self.left.rational_part.as_fraction(),
            self.left.radical_coefficient.as_fraction(),
        )
        c, e = (
            self.right.rational_part.as_fraction(),
            self.right.radical_coefficient.as_fraction(),
        )
        # Addition: (a+c) + (b+e)*sqrt(d)
        if not _fits(a + c) or not _fits(b + e):
            raise _validation_error(
                "addition_result_exceeds_bound",
                "operands would produce an addition result exceeding the "
                f"{_MAX_RESULT_DIGITS}-digit canonical rational bound",
            )
        return self


class AlgebraicMultiplicationRequest(AlgebraicArithmeticRequest):
    """Two elements of Q(sqrt(d)) whose exact product is returnable.

    Admission rejects only requests whose own product
    ``(ac+bed) + (ae+bc)*sqrt(d)`` would exceed the 256-digit canonical
    rational bound; addition growth is irrelevant here.
    """

    @model_validator(mode="after")
    def require_multiplication_result_within_bound(self) -> Self:
        a, b = (
            self.left.rational_part.as_fraction(),
            self.left.radical_coefficient.as_fraction(),
        )
        c, e = (
            self.right.rational_part.as_fraction(),
            self.right.radical_coefficient.as_fraction(),
        )
        d = self.left.radicand
        # Multiplication: (ac + bed) + (ae + bc)*sqrt(d)
        if not _fits(a * c + b * e * d) or not _fits(a * e + b * c):
            raise _validation_error(
                "multiplication_result_exceeds_bound",
                "operands would produce a multiplication result exceeding the "
                f"{_MAX_RESULT_DIGITS}-digit canonical rational bound",
            )
        return self


__all__ = [
    "AlgebraicAdditionRequest",
    "AlgebraicArithmeticRequest",
    "AlgebraicMultiplicationRequest",
]
