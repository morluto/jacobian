"""Typed contracts and source-bound replay for direct factorization.

These models own the small isolated-worker factorization envelope used by the
divisor-enumeration and prime-factorization operations.  Certified
factorization and primality certificates have a distinct, larger envelope.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._integer_models import PrimePower
from jacobian.math.number_theory._models import (
    BoundedInteger,
    _validation_error,
)

# Twenty decimal digits keep the worker's factorization and bounded replay
# envelope useful for ordinary exact divisor and prime-factor workflows.
MAX_DIRECT_FACTORIZATION_DIGITS = 20
MAX_DIRECT_DIVISORS = 4_096
MAX_DIRECT_FACTOR_ENTRIES = 256

FactorizationInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=MAX_DIRECT_FACTORIZATION_DIGITS,
        strict=True,
    ),
]


class FactorizationRequest(StrictModel):
    """One bounded integer for direct exact factorization."""

    value: FactorizationInteger


class NonzeroFactorizationRequest(FactorizationRequest):
    """One nonzero integer with a finite divisor and prime-factorization set."""

    @model_validator(mode="after")
    def require_nonzero_value(self) -> Self:
        if int(self.value) == 0:
            raise _validation_error(
                "zero_has_no_finite_factorization_or_divisor_enumeration",
                "zero has no finite factorization or divisor enumeration",
            )
        return self


class DivisorListResult(StrictModel):
    """An ordered list of positive divisors of one nonzero integer.

    Retains the canonical source integer and the operation's divisor
    convention. Structural validation keeps the representation safe; the
    explicit owner verifier replays the exact enumeration in a bounded worker.
    The list may be empty:
    ``proper_divisors(±1)`` has no positive proper divisors.  Zero remains
    not-applicable (handled at the operation layer).  The source carries the
    same 20-digit factorization bound as the producing requests, so replay
    never factors outside the operation's admitted work envelope.
    """

    status: Literal["COMPLETE", "UNKNOWN"] = "COMPLETE"
    value: FactorizationInteger
    divisors: tuple[BoundedInteger, ...] = Field(
        min_length=0,
        max_length=MAX_DIRECT_DIVISORS,
    )
    convention: Literal["ALL_POSITIVE_DIVISORS", "PROPER_DIVISORS"] = (
        "ALL_POSITIVE_DIVISORS"
    )
    detail: str | None = None

    @classmethod
    def _unknown(
        cls,
        *,
        value: str,
        convention: Literal["ALL_POSITIVE_DIVISORS", "PROPER_DIVISORS"],
        detail: str,
    ) -> Self:
        return cls(
            status="UNKNOWN",
            value=value,
            divisors=(),
            convention=convention,
            detail=detail,
        )

    @model_validator(mode="after")
    def require_source_enumeration(self) -> Self:
        if self.status == "UNKNOWN":
            if self.divisors or not self.detail:
                raise _validation_error(
                    "unknown_divisor_enumeration_shape",
                    "an unknown divisor enumeration has no divisors and includes a detail",
                )
            return self
        values = [int(divisor) for divisor in self.divisors]
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
        value = int(self.value)
        if value == 0:
            raise _validation_error(
                "zero_has_infinitely_many_divisors", "zero has infinitely many divisors"
            )
        return self


class PrimeFactorizationResult(StrictModel):
    """The complete prime-power factorization of one nonzero integer.

    Retains the canonical source integer. Structural validation checks the
    coordinate form; the explicit owner verifier establishes primality and
    completeness.
    The factor list may be empty: ``±1`` has no prime factors.  Zero remains
    not-applicable (handled at the operation layer).
    """

    status: Literal["COMPLETE", "UNKNOWN"] = "COMPLETE"
    value: FactorizationInteger
    factors: tuple[PrimePower, ...] = Field(
        min_length=0,
        max_length=MAX_DIRECT_FACTOR_ENTRIES,
    )
    detail: str | None = None

    @classmethod
    def _unknown(cls, *, value: str, detail: str) -> Self:
        return cls(status="UNKNOWN", value=value, factors=(), detail=detail)

    @model_validator(mode="after")
    def require_source_factorization(self) -> Self:
        if self.status == "UNKNOWN":
            if self.factors or not self.detail:
                raise _validation_error(
                    "unknown_prime_factorization_shape",
                    "an unknown prime factorization has no factors and includes a detail",
                )
            return self
        primes = [factor.prime for factor in self.factors]
        if len(set(primes)) != len(primes):
            raise _validation_error(
                "prime_factors_must_be_unique", "prime factors must be unique"
            )
        value = int(self.value)
        if value == 0:
            raise _validation_error(
                "zero_has_no_finite_prime_factorization",
                "zero has no finite prime factorization",
            )
        target = abs(value)
        product = 1
        previous_prime = 0
        for factor in self.factors:
            prime = int(factor.prime)
            if prime <= previous_prime:
                raise _validation_error(
                    "prime_bases_must_be_strictly_ascending",
                    "prime bases must be strictly ascending",
                )
            if prime < 2:
                raise _validation_error(
                    "factor_prime_domain", "factor primes must be at least 2"
                )
            power_value = 1
            for _ in range(factor.power):
                power_value *= prime
                if power_value > target:
                    raise _validation_error(
                        "prime_powers_must_multiply_to_abs_value",
                        "prime powers must multiply to abs(value)",
                    )
            product *= power_value
            if product > target:
                raise _validation_error(
                    "prime_powers_must_multiply_to_abs_value",
                    "prime powers must multiply to abs(value)",
                )
            previous_prime = prime
        if product != target:
            raise _validation_error(
                "prime_powers_must_multiply_to_abs_value",
                "prime powers must multiply to abs(value)",
            )
        return self


class SquarefreeResult(StrictModel):
    """A squarefreeness decision, or an explicit non-conclusion."""

    status: Literal["SQUAREFREE", "NOT_SQUAREFREE", "UNKNOWN"]
    n: int = Field(ge=0, le=10_000)
    detail: str | None = None

    @classmethod
    def _unknown(cls, *, n: int, detail: str) -> Self:
        return cls(status="UNKNOWN", n=n, detail=detail)

    @model_validator(mode="after")
    def require_unknown_detail(self) -> Self:
        if self.status == "UNKNOWN" and not self.detail:
            raise _validation_error(
                "unknown_squarefree_shape",
                "an unknown squarefreeness result includes a detail",
            )
        return self


class RadicalResult(StrictModel):
    """The exact radical of one admitted integer, or an explicit non-conclusion."""

    status: Literal["COMPLETE", "UNKNOWN"] = "COMPLETE"
    n: int = Field(ge=0, le=10_000)
    value: BoundedInteger | None = None
    detail: str | None = None

    @classmethod
    def _unknown(cls, *, n: int, detail: str) -> Self:
        return cls(status="UNKNOWN", n=n, detail=detail)

    @model_validator(mode="after")
    def require_complete_value(self) -> Self:
        if self.status == "COMPLETE" and self.value is None:
            raise _validation_error(
                "complete_radical_requires_value",
                "a complete radical result includes its exact value",
            )
        if self.status == "UNKNOWN" and (self.value is not None or not self.detail):
            raise _validation_error(
                "unknown_radical_shape",
                "an unknown radical has no value and includes a detail",
            )
        return self
