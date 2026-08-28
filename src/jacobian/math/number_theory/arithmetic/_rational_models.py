"""Pydantic contracts for exact rational arithmetic."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel

MAX_RATIONAL_CONTINUED_FRACTION_TERMS = 1_024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by rational arithmetic contracts."""

    return PydanticCustomError(f"arithmetic.{reason}", message)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class RationalValueRequest(StrictModel):
    """One canonical rational supplied to a unary rational operation."""

    value: CanonicalRational


class NonzeroRationalValueRequest(StrictModel):
    """One nonzero canonical rational supplied to a reciprocal operation."""

    value: CanonicalRational


class RationalPairRequest(StrictModel):
    """Two canonical rationals supplied to a binary rational operation."""

    left: CanonicalRational
    right: CanonicalRational


class RationalDivisionRequest(StrictModel):
    """Two canonical rationals supplied to a division operation (right must be nonzero)."""

    left: CanonicalRational
    right: CanonicalRational


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

    Retains the source rational and the one canonical representation: every
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
        max_length=MAX_RATIONAL_CONTINUED_FRACTION_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_reconstruction(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        quotients = [parse_canonical_integer(term) for term in self.terms]
        if any(quotient < 1 for quotient in quotients[1:]):
            raise _validation_error(
                "continued_fraction_nonpositive_term",
                "every partial quotient after the first must be positive",
            )
        if len(quotients) > 1 and quotients[-1] == 1:
            raise _validation_error(
                "continued_fraction_trailing_one",
                "a multi-term simple continued fraction must not end in 1",
            )
        reconstructed = Fraction(quotients[-1])
        for quotient in reversed(quotients[:-1]):
            reconstructed = quotient + Fraction(1, reconstructed)
        if reconstructed != self.value.as_fraction():
            raise _validation_error(
                "continued_fraction_reconstruction",
                "continued fraction terms must reconstruct the retained rational",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalRational,
        terms: tuple[CanonicalInteger, ...],
    ) -> Self:
        return cls.model_construct(value=value, terms=terms)
