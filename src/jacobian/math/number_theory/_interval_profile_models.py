"""Typed contracts for bounded integer-interval arithmetic-function profiles."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Self

from pydantic import ConfigDict, Field, PrivateAttr, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

# ---------------------------------------------------------------------------
# Admission envelope
# ---------------------------------------------------------------------------
#
# The profiles share one admission shape: a bounded closed interval [L, U]
# with L >= 1 and U >= L.  The key quantities controlling work and output are
# the interval width W = U - L + 1, the base sieve through sqrt(U), and the
# exact result size.  Each operation supplies its own conservative work and
# result-size estimator because the public result shapes have different
# densities.
#
# For squarefree/divisor-count/greatest-prime-factor profiles the kernel is a
# segmented sieve over [L, U] needing primes through floor(sqrt(U)).
#
# For prime-gap profiles the kernel is a segmented prime sieve over [L, U],
# followed by one successor-prime query when the interval contains a prime.

# JSON integers are exactly represented by the public transport through this
# bound.  The actual computational envelope is result- and work-sensitive;
# this is not a mathematical or backend ceiling.
MAX_INTERVAL_UPPER_BOUND: int = (1 << 53) - 1
MAX_INTERVAL_WIDTH: int = 1_000_000
MAX_SIEVE_WORK: int = 20_000_000
MAX_PROFILE_RESULT_BYTES: int = CanonicalLimits().max_output_bytes
_PROFILE_RESULT_OVERHEAD_BYTES: int = 1_024


@dataclass(frozen=True, slots=True)
class IntervalAdmission:
    """The one validated execution envelope for an interval operation."""

    lower_bound: int
    upper_bound: int
    width: int
    estimated_work: int
    estimated_result_bytes: int


def _interval_prime_count_upper_bound(width: int) -> int:
    """Return a conservative bound for primes in an interval of ``width``.

    Brun-Titchmarsh bounds the number of primes in an interval of length
    ``W`` by ``2W / log(W)`` for W > 1.  The one-per-integer bound remains
    sharper for short intervals and protects the small-width edge cases.
    """
    if width <= 1:
        return width
    return min(width, math.ceil(2 * width / math.log(width)) + 1)


def _base_sieve_work(upper_bound: int) -> int:
    """Bound the simple base sieve through ``floor(sqrt(upper_bound))``."""
    root = math.isqrt(upper_bound)
    if root < 2:
        return 0
    return root * (root.bit_length() + 1) + root


def _estimate_squarefree_work(lower_bound: int, upper_bound: int) -> int:
    width = upper_bound - lower_bound + 1
    # Marking p^2 multiples costs at most one hit per interval value in total,
    # plus one first-hit iteration per base prime and one output pass.
    return _base_sieve_work(upper_bound) + width * 2 + math.isqrt(upper_bound)


def _estimate_factor_profile_work(lower_bound: int, upper_bound: int) -> int:
    width = upper_bound - lower_bound + 1
    root = math.isqrt(upper_bound)
    root_bits = max(root.bit_length(), 1)
    value_bits = max(upper_bound.bit_length(), 1)
    # The interval multiple scan is bounded by the harmonic sum over base
    # primes; residual division is bounded by one binary reduction per bit of
    # the input; the final row construction is one more interval pass.
    return _base_sieve_work(upper_bound) + root + width * (root_bits + value_bits + 1)


def _estimate_prime_gap_work(lower_bound: int, upper_bound: int) -> int:
    width = upper_bound - lower_bound + 1
    root = math.isqrt(upper_bound)
    root_bits = max(root.bit_length(), 1)
    # Segmented marking is width-sensitive; the final row scan is charged
    # separately.  The successor query is a single bounded-prime operation.
    return _base_sieve_work(upper_bound) + root + width * (root_bits + 2)


def _integer_digits(value: int) -> int:
    """Return the decimal digit count of a positive integer."""
    return len(str(value))


def _estimate_squarefree_result_bytes(lower_bound: int, upper_bound: int) -> int:
    width = upper_bound - lower_bound + 1
    # Every interval integer occurs exactly once across the two value arrays.
    # One extra byte per value covers its array separator.
    return _PROFILE_RESULT_OVERHEAD_BYTES + width * (_integer_digits(upper_bound) + 1)


def _estimate_divisor_count_result_bytes(lower_bound: int, upper_bound: int) -> int:
    width = upper_bound - lower_bound + 1
    digits = _integer_digits(upper_bound)
    # 24 bytes covers the two field names, punctuation, and an array separator
    # in addition to the two decimal values.
    return _PROFILE_RESULT_OVERHEAD_BYTES + width * (2 * digits + 24)


def _estimate_greatest_prime_factor_result_bytes(
    lower_bound: int, upper_bound: int
) -> int:
    width = upper_bound - lower_bound + 1
    digits = _integer_digits(upper_bound)
    # The longer greatest_prime_factor field needs 32 fixed bytes per row.
    return _PROFILE_RESULT_OVERHEAD_BYTES + width * (2 * digits + 32)


def _estimate_prime_gap_result_bytes(lower_bound: int, upper_bound: int) -> int:
    width = upper_bound - lower_bound + 1
    row_count = _interval_prime_count_upper_bound(width)
    # Bertrand's postulate bounds the successor below 2*U.  39 fixed bytes
    # cover the three field names, punctuation, and an array separator.
    digits = _integer_digits(2 * upper_bound)
    return _PROFILE_RESULT_OVERHEAD_BYTES + row_count * (3 * digits + 39)


_RESULT_ESTIMATOR = Callable[[int, int], int]
_WORK_ESTIMATOR = Callable[[int, int], int]


class IntervalProfileRequest(StrictModel):
    """A bounded closed interval [L, U] with an operation-specific result."""

    lower_bound: StrictInt = Field(ge=1, le=MAX_INTERVAL_UPPER_BOUND)
    upper_bound: StrictInt = Field(ge=1, le=MAX_INTERVAL_UPPER_BOUND)

    _result_estimator: ClassVar[_RESULT_ESTIMATOR] = (
        _estimate_divisor_count_result_bytes
    )
    _work_estimator: ClassVar[_WORK_ESTIMATOR] = _estimate_factor_profile_work
    _admission: IntervalAdmission = PrivateAttr()

    @model_validator(mode="after")
    def require_admitted_interval(self) -> Self:
        if self.upper_bound < self.lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")
        if self.width() > MAX_INTERVAL_WIDTH:
            raise ValueError("interval width exceeds maximum supported width")
        estimated_result_bytes = type(self)._result_estimator(
            self.lower_bound, self.upper_bound
        )
        if estimated_result_bytes > MAX_PROFILE_RESULT_BYTES:
            raise ValueError("interval result exceeds the canonical output budget")
        estimated_work = type(self)._work_estimator(self.lower_bound, self.upper_bound)
        if estimated_work > MAX_SIEVE_WORK:
            raise ValueError(
                "interval exceeds the segmented-sieve work budget of "
                f"{MAX_SIEVE_WORK} steps"
            )
        self._admission = IntervalAdmission(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            width=self.width(),
            estimated_work=estimated_work,
            estimated_result_bytes=estimated_result_bytes,
        )
        return self

    def width(self) -> int:
        return self.upper_bound - self.lower_bound + 1

    @property
    def admission(self) -> IntervalAdmission:
        """Return the admission decision computed during request validation."""
        return self._admission


class SquarefreeProfileRequest(IntervalProfileRequest):
    """Interval request admitted for the two-array squarefree result."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Closed positive integer interval [lower_bound, upper_bound]. "
                f"The transport-safe upper bound is at most {MAX_INTERVAL_UPPER_BOUND:,}, "
                f"the width is at most {MAX_INTERVAL_WIDTH:,}, and the "
                f"segmented-sieve work estimate must fit within {MAX_SIEVE_WORK:,} "
                "steps. The operation-specific worst-case JSON result estimate "
                f"must fit within {MAX_PROFILE_RESULT_BYTES:,} bytes. Squarefree "
                "values occur once across the two returned arrays."
            )
        }
    )
    _result_estimator: ClassVar[_RESULT_ESTIMATOR] = _estimate_squarefree_result_bytes
    _work_estimator: ClassVar[_WORK_ESTIMATOR] = _estimate_squarefree_work


class DivisorCountProfileRequest(IntervalProfileRequest):
    """Interval request admitted for the divisor-count row result."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Closed positive integer interval [lower_bound, upper_bound]. "
                f"The transport-safe upper bound is at most {MAX_INTERVAL_UPPER_BOUND:,}, "
                f"the width is at most {MAX_INTERVAL_WIDTH:,}, and the "
                f"segmented-sieve work estimate must fit within {MAX_SIEVE_WORK:,} "
                "steps. The operation-specific worst-case JSON row estimate must "
                f"fit within {MAX_PROFILE_RESULT_BYTES:,} bytes. The result "
                "contains one (n, divisor_count) row per interval value."
            )
        }
    )


class GreatestPrimeFactorProfileRequest(IntervalProfileRequest):
    """Interval request admitted for the greatest-prime-factor rows."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Closed positive integer interval [lower_bound, upper_bound]. "
                f"The transport-safe upper bound is at most {MAX_INTERVAL_UPPER_BOUND:,}, "
                f"the width is at most {MAX_INTERVAL_WIDTH:,}, and the "
                f"segmented-sieve work estimate must fit within {MAX_SIEVE_WORK:,} "
                "steps. The operation-specific worst-case JSON row estimate must "
                f"fit within {MAX_PROFILE_RESULT_BYTES:,} bytes. The result "
                "contains one (n, greatest_prime_factor) row per interval value."
            )
        }
    )
    _result_estimator: ClassVar[_RESULT_ESTIMATOR] = (
        _estimate_greatest_prime_factor_result_bytes
    )


class PrimeGapProfileRequest(IntervalProfileRequest):
    """Interval request admitted for the sparse consecutive-prime rows."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Closed positive integer interval [lower_bound, upper_bound]. "
                f"The transport-safe upper bound is at most {MAX_INTERVAL_UPPER_BOUND:,}, "
                f"the width is at most {MAX_INTERVAL_WIDTH:,}, and the "
                f"segmented-sieve work estimate must fit within {MAX_SIEVE_WORK:,} "
                "steps. The "
                "operation-specific worst-case JSON estimate for the primes "
                "in the requested interval must fit within "
                f"{MAX_PROFILE_RESULT_BYTES:,} bytes."
            )
        }
    )
    _result_estimator: ClassVar[_RESULT_ESTIMATOR] = _estimate_prime_gap_result_bytes
    _work_estimator: ClassVar[_WORK_ESTIMATOR] = _estimate_prime_gap_work


# ---------------------------------------------------------------------------
# Squarefree profile result
# ---------------------------------------------------------------------------


class SquarefreeProfileResult(StrictModel):
    """Complete exact squarefree/non-squarefree partition of [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    squarefree_values: tuple[StrictInt, ...]
    nonsquarefree_values: tuple[StrictInt, ...]
    squarefree_count: StrictInt
    nonsquarefree_count: StrictInt


# ---------------------------------------------------------------------------
# Divisor-count profile result
# ---------------------------------------------------------------------------


class DivisorCountProfileRow(StrictModel):
    """One (n, tau(n)) pair in a divisor-count profile."""

    n: StrictInt
    divisor_count: StrictInt = Field(ge=1)


class DivisorCountProfileResult(StrictModel):
    """Complete ordered divisor-count table over [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    rows: tuple[DivisorCountProfileRow, ...]


# ---------------------------------------------------------------------------
# Greatest-prime-factor profile result
# ---------------------------------------------------------------------------


class GreatestPrimeFactorProfileRow(StrictModel):
    """One (n, P+(n)) pair in a greatest-prime-factor profile."""

    n: StrictInt
    greatest_prime_factor: StrictInt = Field(ge=1)


class GreatestPrimeFactorProfileResult(StrictModel):
    """Complete ordered greatest-prime-factor table over [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    rows: tuple[GreatestPrimeFactorProfileRow, ...]


# ---------------------------------------------------------------------------
# Prime-gap profile result
# ---------------------------------------------------------------------------


class PrimeGapProfileRow(StrictModel):
    """One consecutive-prime pair (p, q, q - p) in a prime-gap profile."""

    lower_prime: StrictInt = Field(ge=2)
    upper_prime: StrictInt = Field(ge=2)
    gap: StrictInt = Field(ge=1)


class PrimeGapProfileResult(StrictModel):
    """Complete ordered consecutive-prime gap table over [L, U]."""

    lower_bound: StrictInt
    upper_bound: StrictInt
    rows: tuple[PrimeGapProfileRow, ...]


__all__ = [
    "MAX_INTERVAL_UPPER_BOUND",
    "MAX_INTERVAL_WIDTH",
    "MAX_PROFILE_RESULT_BYTES",
    "MAX_SIEVE_WORK",
    "DivisorCountProfileRequest",
    "DivisorCountProfileResult",
    "DivisorCountProfileRow",
    "GreatestPrimeFactorProfileRequest",
    "GreatestPrimeFactorProfileResult",
    "GreatestPrimeFactorProfileRow",
    "IntervalAdmission",
    "IntervalProfileRequest",
    "PrimeGapProfileRequest",
    "PrimeGapProfileResult",
    "PrimeGapProfileRow",
    "SquarefreeProfileRequest",
    "SquarefreeProfileResult",
]
