"""Canonical exact scalar wire values."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Self

from pydantic import StringConstraints, model_validator

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.results import ContractModel

CanonicalInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        strict=True,
    ),
]

# Shared by request validators, search producers, and output Field(le=...) caps:
# all three use the exact deduplicated cartesian grid size, not a loose upper bound.
RATIONAL_SEARCH_GRID_LIMIT = 10_000
MAX_CANONICAL_RATIONAL_DIGITS = 32_768


def bounded_rational_scalars(
    max_abs_numerator: int, max_denominator: int
) -> tuple[Fraction, ...]:
    """Return sorted distinct rationals with |n| <= max_abs_numerator and 1 <= d <= max_denominator.

    Deduplication matters: Fraction reduces equivalents (e.g. 1/2 and 2/4), so the
    distinct count is strictly at most (2 * max_abs_numerator + 1) * max_denominator.
    Validators, producers, and output models must all use this exact set (or its
    size) when enforcing RATIONAL_SEARCH_GRID_LIMIT.
    """

    return tuple(
        sorted(
            {
                Fraction(numerator, denominator)
                for denominator in range(1, max_denominator + 1)
                for numerator in range(-max_abs_numerator, max_abs_numerator + 1)
            }
        )
    )


def bounded_rational_grid_size(
    max_abs_numerator: int, max_denominator: int, dimension: int
) -> int:
    """Cartesian size of the exact deduplicated rational grid in ``dimension`` variables."""

    scalar_count = len(bounded_rational_scalars(max_abs_numerator, max_denominator))
    size: int = scalar_count**dimension
    return size


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
