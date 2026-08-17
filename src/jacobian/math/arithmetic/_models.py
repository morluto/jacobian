"""Named Pydantic wire contracts for exact arithmetic operations.

The arithmetic domain owns integer absolute value, sign, decimal digit
sum/count, base expansion, integer nth root, and rational arithmetic/order.
Number-theory models (gcd, lcm, divisibility, primes, modular arithmetic,
integer predicates) live with their owner in ``jacobian.math.number_theory``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

# ---------------------------------------------------------------------------
# Shared bounds
# ---------------------------------------------------------------------------

_MAX_BASE = 10_000
_MAX_NONNEGATIVE = 1_000
MAX_BASE_DIGITS = 1_024

# A positional digit is a small non-negative canonical integer string.  The
# max length of 4 comfortably covers every base up to ``_MAX_BASE`` (10_000).
BaseDigit = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|[1-9][0-9]*)$",
        max_length=4,
        strict=True,
    ),
]


# ---------------------------------------------------------------------------
# Requests — unary integer
# ---------------------------------------------------------------------------


class IntegerValueRequest(StrictModel):
    """One canonical integer supplied to a unary integer operation."""

    value: CanonicalInteger


# ---------------------------------------------------------------------------
# Requests — base expansion
# ---------------------------------------------------------------------------


class IntegerBaseDigitsRequest(StrictModel):
    """Expand one integer's absolute value in a positional base.

    The positional base is named explicitly so this request cannot be confused
    with modular arithmetic.
    """

    value: CanonicalInteger
    base: int = Field(ge=2, le=_MAX_BASE)


# ---------------------------------------------------------------------------
# Requests — nth root
# ---------------------------------------------------------------------------


class IntegerNthRootRequest(StrictModel):
    """One canonical integer and a positive root degree."""

    value: CanonicalInteger
    degree: int = Field(ge=1, le=_MAX_NONNEGATIVE)


# ---------------------------------------------------------------------------
# Structured results — integer
# ---------------------------------------------------------------------------


class IntegerValueResult(StrictModel):
    """One canonical integer produced by a unary integer operation."""

    value: CanonicalInteger


class IntegerSignResult(StrictModel):
    """The sign of one integer as -1, 0, or 1."""

    sign: Literal[-1, 0, 1]


class IntegerNthRootResult(StrictModel):
    """The floor nth root of one integer and whether it is exact."""

    root: CanonicalInteger
    exact: bool


class IntegerBaseDigitsResult(StrictModel):
    """One integer's sign and positional digits in a declared base."""

    sign: Literal[-1, 0, 1]
    base: int = Field(ge=2, le=_MAX_BASE)
    digits: tuple[BaseDigit, ...] = Field(min_length=1, max_length=MAX_BASE_DIGITS)

    @model_validator(mode="after")
    def require_canonical_digits(self) -> Self:
        if any(int(digit) >= self.base for digit in self.digits):
            raise ValueError("every positional digit must be smaller than the base")
        if self.sign == 0 and self.digits != ("0",):
            raise ValueError("zero sign requires the canonical zero digit")
        if self.sign != 0 and self.digits[0] == "0":
            raise ValueError("nonzero positional digits cannot have a leading zero")
        return self
