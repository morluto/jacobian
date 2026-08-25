"""Typed contracts owned by integer divisibility operations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import BoundedInteger, _validation_error


class IntegerPairRequest(StrictModel):
    """Two canonical integers supplied to a symmetric binary operation."""

    left: BoundedInteger
    right: BoundedInteger


class DivisibilityRequest(StrictModel):
    """A divisor and dividend supplied to a divisibility predicate."""

    divisor: BoundedInteger
    dividend: BoundedInteger

    @model_validator(mode="after")
    def require_nonzero_divisor(self) -> Self:
        if int(self.divisor) == 0:
            raise _validation_error(
                "divisor_must_be_nonzero", "divisor must be nonzero"
            )
        return self


class ValuationRequest(StrictModel):
    """One integer and a prime base supplied to a p-adic valuation."""

    value: BoundedInteger
    prime: BoundedInteger

    @model_validator(mode="after")
    def require_valid_valuation_domain(self) -> Self:
        from sympy import isprime

        if int(self.value) == 0:
            raise _validation_error(
                "valuation_requires_nonzero_value", "valuation requires nonzero value"
            )
        if int(self.prime) < 2 or not isprime(int(self.prime)):
            raise _validation_error(
                "valuation_requires_a_prime_absolute_base_2",
                "valuation requires a prime absolute base >= 2",
            )
        return self


class ExtendedGcdResult(StrictModel):
    """A gcd together with exact Bezout coefficients."""

    gcd: BoundedInteger
    left_coefficient: BoundedInteger
    right_coefficient: BoundedInteger


__all__ = [
    "DivisibilityRequest",
    "ExtendedGcdResult",
    "IntegerPairRequest",
    "ValuationRequest",
]
