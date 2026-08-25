"""Typed contracts and source-bound replay for direct factorization.

These models own the small, synchronous factorization envelope used by the
divisor-enumeration and prime-factorization operations.  Certified
factorization and primality certificates have a distinct, larger envelope.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import (
    BoundedInteger,
    PrimePower,
    _validation_error,
)

# ``factorint`` is used both by the direct kernels and source-bound result
# replay.  Twenty decimal digits keep a hard semiprime within the synchronous
# envelope while still admitting ordinary exact divisor and factor workflows.
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
    convention so validation replays the exact enumeration: the list is
    exactly all positive divisors of ``abs(value)`` (proper ones exclude
    ``abs(value)`` itself) in ascending order.  The list may be empty:
    ``proper_divisors(±1)`` has no positive proper divisors.  Zero remains
    not-applicable (handled at the operation layer).  The source carries the
    same 20-digit factorization bound as the producing requests, so replay
    never factors outside the operation's admitted work envelope.
    """

    value: FactorizationInteger
    divisors: tuple[BoundedInteger, ...] = Field(
        min_length=0,
        max_length=MAX_DIRECT_DIVISORS,
    )
    convention: Literal["ALL_POSITIVE_DIVISORS", "PROPER_DIVISORS"] = (
        "ALL_POSITIVE_DIVISORS"
    )

    @model_validator(mode="after")
    def require_source_enumeration(self) -> Self:
        from jacobian.math.number_theory._factorization_kernels import (
            _replayed_divisors,
        )

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
        if self.divisors != _replayed_divisors(
            value, proper=self.convention == "PROPER_DIVISORS"
        ):
            raise _validation_error(
                "divisor_list_must_enumerate_the_divisors_of_the_source",
                "divisor list must enumerate the divisors of the source",
            )
        return self


class PrimeFactorizationResult(StrictModel):
    """The complete prime-power factorization of one nonzero integer.

    Retains the canonical source integer so validation replays the defining
    invariant: prime bases are strictly increasing proven primes with
    positive exponents whose product reconstructs ``abs(value)`` exactly.
    The factor list may be empty: ``±1`` has no prime factors.  Zero remains
    not-applicable (handled at the operation layer).
    """

    value: BoundedInteger
    factors: tuple[PrimePower, ...] = Field(
        min_length=0,
        max_length=MAX_DIRECT_FACTOR_ENTRIES,
    )

    @model_validator(mode="after")
    def require_source_factorization(self) -> Self:
        from sympy import isprime

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
            if prime < 2 or not isprime(prime):
                raise _validation_error(
                    "f_factor_prime_is_not_prime", f"{factor.prime} is not prime"
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
