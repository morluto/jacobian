"""Typed contracts for p-adic valuation profiles of binomial coefficients."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)

MAX_BINOMIAL_PROFILE_RESULT_BYTES = CanonicalLimits().max_output_bytes
MAX_BINOMIAL_DIGIT_WORK = 2_000_000
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


def _base_digit_count(value: int, base: int) -> int:
    """Return the number of base-``base`` digits in a nonnegative value."""

    if value == 0:
        return 0
    digits = 0
    while value:
        value //= base
        digits += 1
    return digits


def _binomial_result_upper_bound_bytes(n: int, prime: int) -> int:
    """Bound the exact canonical size of a complete valuation profile."""

    valuation_digits = max(1, _base_digit_count(n, prime))
    row_size = strict_json_object_size(
        (
            ("k", len(encode_strict_json(n))),
            ("valuation", len(encode_strict_json(valuation_digits))),
        )
    )
    rows_size = 2 + (n + 1) * row_size + n
    return strict_json_object_size(
        (
            ("n", len(encode_strict_json(n))),
            ("prime", len(encode_strict_json(prime))),
            ("rows", rows_size),
        )
    )


# This cheap pre-check prevents huge Python integers from reaching the exact
# size estimator before the result envelope can reject them.
_MINIMUM_BINOMIAL_ROW_SIZE = strict_json_object_size(
    (
        ("k", len(encode_strict_json(0))),
        ("valuation", len(encode_strict_json(0))),
    )
)
_MAX_BINOMIAL_ROWS_FROM_OUTPUT = MAX_BINOMIAL_PROFILE_RESULT_BYTES // (
    _MINIMUM_BINOMIAL_ROW_SIZE + 1
)


class BinomialValuationProfileRequest(StrictModel):
    """Parameters for computing v_p(C(n,k)) for all k from 0 to n."""

    n: int = Field(ge=0)
    prime: int = Field(ge=2, le=_MAX_SAFE_JSON_INTEGER)

    @model_validator(mode="after")
    def require_prime(self) -> Self:
        """Validate that prime is actually prime."""
        if self.n + 1 > _MAX_BINOMIAL_ROWS_FROM_OUTPUT:
            raise ValueError("valuation profile exceeds the canonical output budget")
        predicted = _binomial_result_upper_bound_bytes(self.n, self.prime)
        if predicted > MAX_BINOMIAL_PROFILE_RESULT_BYTES:
            raise ValueError(
                "valuation profile exceeds the canonical output budget of "
                f"{MAX_BINOMIAL_PROFILE_RESULT_BYTES} bytes"
            )
        digit_work = (self.n + 1) * max(1, _base_digit_count(self.n, self.prime))
        if digit_work > MAX_BINOMIAL_DIGIT_WORK:
            raise ValueError(
                "valuation profile exceeds the digitwise work budget of "
                f"{MAX_BINOMIAL_DIGIT_WORK} steps"
            )
        from sympy import isprime

        if not isprime(self.prime):
            raise ValueError("prime must be a prime number")
        return self


class BinomialValuationProfileRow(StrictModel):
    """One (k, v_p(C(n,k))) pair."""

    k: int = Field(ge=0)
    valuation: int = Field(ge=0)


class BinomialValuationProfileResult(StrictModel):
    """Complete v_p(C(n,k)) profile for k=0..n."""

    n: int
    prime: int
    rows: list[BinomialValuationProfileRow]


__all__ = [
    "MAX_BINOMIAL_DIGIT_WORK",
    "MAX_BINOMIAL_PROFILE_RESULT_BYTES",
    "BinomialValuationProfileRequest",
    "BinomialValuationProfileResult",
    "BinomialValuationProfileRow",
]
