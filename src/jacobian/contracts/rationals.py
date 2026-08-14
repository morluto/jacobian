"""Pydantic contracts for exact rational arithmetic."""

from __future__ import annotations

from pydantic import Field

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalInteger, CanonicalRational

_MAX_CONTINUED_FRACTION_TERMS = 1_024


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class RationalValueRequest(ContractModel):
    """One canonical rational supplied to a unary rational operation."""

    value: CanonicalRational


class RationalPairRequest(ContractModel):
    """Two canonical rationals supplied to a binary rational operation."""

    left: CanonicalRational
    right: CanonicalRational


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------


class RationalValueResult(ContractModel):
    """One canonical rational produced by a rational operation."""

    value: CanonicalRational


class RationalIntegerResult(ContractModel):
    """One canonical integer produced by rounding a rational."""

    value: CanonicalInteger


class RationalComparisonResult(ContractModel):
    """Truth value of a rational predicate."""

    holds: bool


class RationalContinuedFractionResult(ContractModel):
    """The finite simple continued fraction expansion of one rational."""

    terms: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=_MAX_CONTINUED_FRACTION_TERMS,
    )
