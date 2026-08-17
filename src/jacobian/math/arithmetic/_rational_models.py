"""Pydantic contracts for exact rational arithmetic."""

from __future__ import annotations

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
    """The finite simple continued fraction expansion of one rational."""

    terms: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=_MAX_CONTINUED_FRACTION_TERMS,
    )
