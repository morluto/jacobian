"""Shared bounded integer-operation contracts.

The neutral ``_models`` module owns the canonical integer grammar.  These
contracts own the small-integer admission envelope used by the divisibility,
prime, and factorization kernels.
"""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import BoundedInteger

# These operations may invoke a bounded in-process factorization backend.
MAX_SMALL_INTEGER = 10_000


class IntegerValueRequest(StrictModel):
    """One canonical integer supplied to a unary integer operation."""

    value: BoundedInteger


class ArithmeticFunctionRequest(StrictModel):
    """A small nonnegative integer for an exact arithmetic function."""

    n: StrictInt = Field(ge=0, le=MAX_SMALL_INTEGER)


class NonnegativeIntegerRequest(StrictModel):
    """One bounded non-negative integer."""

    n: StrictInt = Field(ge=0, le=MAX_SMALL_INTEGER)


class PositiveIntegerRequest(StrictModel):
    """One bounded positive integer."""

    n: StrictInt = Field(ge=1, le=MAX_SMALL_INTEGER)


class BooleanResult(StrictModel):
    """Truth value of an integer predicate."""

    holds: bool


class PrimePower(StrictModel):
    """One prime base and its exponent in a prime factorization."""

    prime: BoundedInteger
    power: int = Field(ge=1, le=MAX_SMALL_INTEGER)


__all__ = [
    "MAX_SMALL_INTEGER",
    "ArithmeticFunctionRequest",
    "BooleanResult",
    "IntegerValueRequest",
    "NonnegativeIntegerRequest",
    "PositiveIntegerRequest",
    "PrimePower",
]
