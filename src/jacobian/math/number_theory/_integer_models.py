"""Shared bounded integer-operation contracts.

The neutral ``_models`` module owns the canonical integer grammar.  These
contracts own the small-integer admission envelope used by the divisibility,
prime, and factorization kernels.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.number_theory._integer_input import (
    MAX_SAFE_INTEGER,
    IntegerInputEncoding,
)
from jacobian.math.number_theory._models import BoundedInteger

# These operations may invoke a bounded in-process factorization backend.
MAX_SMALL_INTEGER = 10_000


class IntegerValueRequest(StrictModel):
    """One canonical integer supplied to a unary integer operation."""

    value: BoundedInteger


class NonnegativeIntegerRequest(StrictModel):
    """One bounded non-negative integer."""

    n: Annotated[
        int,
        IntegerInputEncoding(
            max_digits=len(str(MAX_SMALL_INTEGER)), minimum=0, maximum=MAX_SMALL_INTEGER
        ),
    ] = Field(
        description="Integer n in [0, 10000], as a canonical decimal string or JSON integer."
    )


class PositiveIntegerRequest(StrictModel):
    """One bounded positive integer."""

    n: Annotated[
        int,
        IntegerInputEncoding(
            max_digits=len(str(MAX_SMALL_INTEGER)), minimum=1, maximum=MAX_SMALL_INTEGER
        ),
    ] = Field(
        description="Integer n in [1, 10000], as a canonical decimal string or JSON integer."
    )


class BooleanResult(StrictModel):
    """Truth value of an integer predicate."""

    holds: bool


class PrimePower(StrictModel):
    """One prime base and its exponent in a prime factorization."""

    prime: BoundedInteger
    power: int = Field(ge=1, le=MAX_SMALL_INTEGER)


__all__ = [
    "MAX_SAFE_INTEGER",
    "MAX_SMALL_INTEGER",
    "BooleanResult",
    "IntegerValueRequest",
    "NonnegativeIntegerRequest",
    "PositiveIntegerRequest",
    "PrimePower",
]
