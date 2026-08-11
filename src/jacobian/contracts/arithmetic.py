"""Named Pydantic wire contracts for exact arithmetic capabilities.

The arithmetic domain owns integer absolute value, sign, decimal digit
sum/count, base expansion, integer nth root, and rational arithmetic/order.
Number-theory contracts (gcd, lcm, divisibility, primes, modular arithmetic,
integer predicates) live in ``contracts/number_theory.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.contracts.exact import CanonicalInteger
from jacobian.contracts.results import ContractModel

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


class IntegerValueRequest(ContractModel):
    """One canonical integer supplied to a unary integer operation."""

    value: CanonicalInteger


# ---------------------------------------------------------------------------
# Requests — base expansion
# ---------------------------------------------------------------------------


class IntegerBaseDigitsRequest(ContractModel):
    """Expand one integer's absolute value in a positional base.

    The positional base is named explicitly so this request cannot be confused
    with modular arithmetic.
    """

    value: CanonicalInteger
    base: int = Field(ge=2, le=_MAX_BASE)


# ---------------------------------------------------------------------------
# Requests — nth root
# ---------------------------------------------------------------------------


class IntegerNthRootRequest(ContractModel):
    """One canonical integer and a positive root degree."""

    value: CanonicalInteger
    degree: int = Field(ge=1, le=_MAX_NONNEGATIVE)


# ---------------------------------------------------------------------------
# Structured results — integer
# ---------------------------------------------------------------------------


class IntegerValueResult(ContractModel):
    """One canonical integer produced by a unary integer operation."""

    value: CanonicalInteger


class IntegerSignResult(ContractModel):
    """The sign of one integer as -1, 0, or 1."""

    sign: Literal[-1, 0, 1]


class IntegerNthRootResult(ContractModel):
    """The floor nth root of one integer and whether it is exact."""

    root: CanonicalInteger
    exact: bool


class IntegerBaseDigitsResult(ContractModel):
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
