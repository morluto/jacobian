"""Typed contracts for bounded direct factorization.

These models own the small isolated-worker factorization envelope used by the
divisor-enumeration and prime-factorization operations.  Certified
factorization and primality certificates have a distinct, larger envelope.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.number_theory._integer_models import PrimePower
from jacobian.math.number_theory._models import (
    BoundedInteger,
    _validation_error,
)

# Twenty decimal digits keep the worker's factorization envelope useful for
# ordinary exact divisor and prime-factor workflows.
MAX_DIRECT_FACTORIZATION_DIGITS = 20
# The largest divisor count below 10**MAX_DIRECT_FACTORIZATION_DIGITS is
# attained by 92005279690628304000. This binds enumeration to the actual source
# envelope instead of rejecting valid 20-digit inputs at an unrelated list cap.
MAX_DIRECT_DIVISORS = 245_760
MAX_DIRECT_FACTOR_ENTRIES = 256

FactorizationInteger = Annotated[
    int,
    DecimalIntegerEncoding(max_digits=MAX_DIRECT_FACTORIZATION_DIGITS),
]


class FactorizationRequest(StrictModel):
    """One bounded integer for direct exact factorization."""

    value: FactorizationInteger


class DivisorListResult(StrictModel):
    """An ordered list of positive divisors of one nonzero integer.

    Retains the canonical source integer and the operation's divisor
    convention. Structural validation keeps the representation safe; the
    producing kernel establishes the exact enumeration.
    The list may be empty: ``proper_divisors(±1)`` has no positive proper
    divisors. Zero is not applicable to the producing operations.
    """

    status: Literal["COMPLETE"] = "COMPLETE"
    value: FactorizationInteger
    divisors: tuple[BoundedInteger, ...] = Field(
        min_length=0,
        max_length=MAX_DIRECT_DIVISORS,
    )
    convention: Literal["ALL_POSITIVE_DIVISORS", "PROPER_DIVISORS"] = (
        "ALL_POSITIVE_DIVISORS"
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: FactorizationInteger,
        divisors: tuple[BoundedInteger, ...],
        convention: Literal["ALL_POSITIVE_DIVISORS", "PROPER_DIVISORS"],
    ) -> Self:
        """Build a complete result after the factorization kernel succeeds."""

        return cls.model_construct(
            value=value,
            divisors=divisors,
            convention=convention,
        )

    @model_validator(mode="after")
    def require_source_enumeration(self) -> Self:
        values = list(self.divisors)
        if any(value < 1 for value in values):
            raise _validation_error(
                "divisors_must_be_positive", "divisors must be positive"
            )
        if values != sorted(values):
            raise _validation_error(
                "divisors_must_be_ascending", "divisors must be ascending"
            )
        if len(set(values)) != len(values):
            raise _validation_error(
                "divisors_must_be_unique", "divisors must be unique"
            )
        return self


class PrimeFactorizationResult(StrictModel):
    """The complete prime-power factorization of one nonzero integer.

    Retains the canonical source integer. Structural validation checks the
    coordinate form; the producing kernel establishes primality and
    completeness.
    The factor list may be empty: ``±1`` has no prime factors.  Zero remains
    not-applicable (handled at the operation layer).
    """

    status: Literal["COMPLETE"] = "COMPLETE"
    value: FactorizationInteger
    factors: tuple[PrimePower, ...] = Field(
        min_length=0,
        max_length=MAX_DIRECT_FACTOR_ENTRIES,
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: FactorizationInteger,
        factors: tuple[PrimePower, ...],
    ) -> Self:
        """Build a complete result after the factorization kernel succeeds."""

        return cls.model_construct(value=value, factors=factors)

    @model_validator(mode="after")
    def require_source_factorization(self) -> Self:
        primes = [factor.prime for factor in self.factors]
        if len(set(primes)) != len(primes):
            raise _validation_error(
                "prime_factors_must_be_unique", "prime factors must be unique"
            )
        previous_prime = 0
        for factor in self.factors:
            prime = factor.prime
            if prime <= previous_prime:
                raise _validation_error(
                    "prime_bases_must_be_strictly_ascending",
                    "prime bases must be strictly ascending",
                )
            if prime < 2:
                raise _validation_error(
                    "factor_prime_domain", "factor primes must be at least 2"
                )
            previous_prime = prime
        return self
