"""Typed contracts and bounds owned by prime operations."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StrictInt, StringConstraints

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import BoundedInteger

# ``primorial(n)`` carries n(ln n + ln ln n)/ln 10 digits.  The declared
# result budget is ``_MAX_PRIMORIAL_DIGITS`` (3_400), and primorial(1001)
# already has 3397 digits while primorial(1002) has 3401, so the exact
# admitted boundary is n <= 1001.
_MAX_PRIMORIAL_N = 1001
_MAX_PRIMORIAL_DIGITS = 3_400


class PrimalityRequest(StrictModel):
    """One bounded canonical integer for the maintained primality backend."""

    value: BoundedInteger


class PrimorialRequest(StrictModel):
    """One bounded positive integer whose primorial fits the result contract.

    ``primorial(n)`` grows like ``exp(n log n)``: the product of the first
    ``n`` primes carries ``n(log n + log log n) / ln 10`` digits.  The
    shared arithmetic-function bound admits values whose primorial would
    exceed the declared ``_MAX_PRIMORIAL_DIGITS``-digit result, so this
    request derives its own conservative ceiling from the digit bound.
    """

    n: StrictInt = Field(ge=1, le=_MAX_PRIMORIAL_N)


class PreviousPrimeRequest(StrictModel):
    """One bounded integer n >= 3 for previous-prime queries."""

    n: StrictInt = Field(ge=3, le=10_000)


PrimorialInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|[1-9][0-9]*)$",
        max_length=_MAX_PRIMORIAL_DIGITS,
        strict=True,
    ),
]


class PrimorialResult(StrictModel):
    """The primorial (product of the first n primes)."""

    value: PrimorialInteger
