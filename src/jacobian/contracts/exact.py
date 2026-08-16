"""Canonical exact scalar wire values."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Self

from pydantic import StringConstraints, model_validator

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.base import ContractModel

CanonicalInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        strict=True,
    ),
]

MAX_CANONICAL_RATIONAL_DIGITS = 32_768


def require_bounded_rational(
    value: CanonicalRational,
    *,
    max_digits: int,
    label: str,
) -> None:
    """Reject a canonical rational whose components exceed a domain bound."""

    if (
        len(value.num.lstrip("-")) > max_digits
        or len(value.den.lstrip("-")) > max_digits
    ):
        raise ValueError(f"{label} exceeds the {max_digits}-digit bound")


class CanonicalRational(ContractModel):
    num: CanonicalInteger
    den: CanonicalInteger

    @model_validator(mode="after")
    def require_reduced_positive_denominator(self) -> Self:
        if (
            len(self.num.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS
            or len(self.den.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise ValueError(
                "rational components exceed the canonical 32,768-digit limit"
            )
        denominator = parse_canonical_integer(self.den)
        if denominator == 0:
            raise ValueError("rational denominator cannot be zero")
        value = Fraction(parse_canonical_integer(self.num), denominator)
        if self.num != format_canonical_integer(
            value.numerator
        ) or self.den != format_canonical_integer(value.denominator):
            raise ValueError(
                "rational must be reduced with a positive denominator and canonical zero"
            )
        return self

    def as_fraction(self) -> Fraction:
        return Fraction(*self.as_integer_ratio())

    def as_integer_ratio(self) -> tuple[int, int]:
        return parse_canonical_integer(self.num), parse_canonical_integer(self.den)

    @classmethod
    def from_integer_ratio(cls, numerator: int, denominator: int) -> CanonicalRational:
        try:
            fraction = Fraction(numerator, denominator)
        except ZeroDivisionError:
            raise ValueError("rational denominator cannot be zero") from None
        return cls.from_fraction(fraction)

    @classmethod
    def from_fraction(cls, value: Fraction) -> CanonicalRational:
        return cls(
            num=format_canonical_integer(value.numerator),
            den=format_canonical_integer(value.denominator),
        )
