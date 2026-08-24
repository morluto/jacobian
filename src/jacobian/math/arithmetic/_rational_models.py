"""Pydantic contracts for exact rational arithmetic."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel

_MAX_CONTINUED_FRACTION_TERMS = 1_024


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class RationalValueRequest(StrictModel):
    """One canonical rational supplied to a unary rational operation."""

    value: CanonicalRational


class NonzeroRationalValueRequest(StrictModel):
    """One nonzero canonical rational supplied to a reciprocal operation."""

    value: CanonicalRational

    @model_validator(mode="after")
    def require_nonzero(self) -> Self:
        if self.value.as_fraction() == 0:
            raise ValueError("reciprocal requires a nonzero rational")
        return self


class RationalPairRequest(StrictModel):
    """Two canonical rationals supplied to a binary rational operation."""

    left: CanonicalRational
    right: CanonicalRational


class RationalDivisionRequest(StrictModel):
    """Two canonical rationals supplied to a division operation (right must be nonzero)."""

    left: CanonicalRational
    right: CanonicalRational

    @model_validator(mode="after")
    def require_nonzero_divisor(self) -> Self:
        if self.right.as_fraction() == 0:
            raise ValueError("quotient requires a nonzero divisor")
        return self


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------


class RationalValueResult(StrictModel):
    """One canonical rational produced by a rational operation."""

    value: CanonicalRational


class RationalIntegerResult(StrictModel):
    """One canonical integer produced by rounding a rational."""

    value: CanonicalInteger


class RationalComparisonResult(StrictModel):
    """Truth value of a rational predicate."""

    holds: bool


class RationalContinuedFractionResult(StrictModel):
    """The canonical finite simple continued fraction of one rational.

    Retains the source rational so validation replays the exact continuant
    reconstruction and enforces the one canonical representation: every
    partial quotient after the first is positive, and a multi-term
    expansion never ends in ``1`` (the two-representation ambiguity is
    resolved by merging a trailing ``[... , x, 1]`` into ``[..., x + 1]``).
    The sign convention follows floor-based expansion, so the first term
    carries the sign and every later term is positive; integers expand to a
    single term.
    """

    value: CanonicalRational
    terms: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=_MAX_CONTINUED_FRACTION_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_reconstruction(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        quotients = [parse_canonical_integer(term) for term in self.terms]
        if any(quotient < 1 for quotient in quotients[1:]):
            raise ValueError("every partial quotient after the first must be positive")
        if len(quotients) > 1 and quotients[-1] == 1:
            raise ValueError("a multi-term simple continued fraction must not end in 1")
        numerator_minus_2, numerator_minus_1 = 1, quotients[0]
        denominator_minus_2, denominator_minus_1 = 0, 1
        for quotient in quotients[1:]:
            numerator_minus_2, numerator_minus_1 = (
                numerator_minus_1,
                quotient * numerator_minus_1 + numerator_minus_2,
            )
            denominator_minus_2, denominator_minus_1 = (
                denominator_minus_1,
                quotient * denominator_minus_1 + denominator_minus_2,
            )
        if Fraction(numerator_minus_1, denominator_minus_1) != self.value.as_fraction():
            raise ValueError(
                "terms must reconstruct the retained rational through the "
                "continuant recurrence"
            )
        return self
