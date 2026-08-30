"""Typed contracts and bounded admission for exact friable-family enumeration."""

from __future__ import annotations

import math
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

# The enumeration shares the generated-exponent-vector regime with
# ``number_theory.friable.count.compute``.  Materializing a complete ordered
# family requires at most one generated candidate per prime-exponent tuple
# plus sorting/deduplication, so admission reuses the same source-sensitive
# bounds.
MAX_FRIABLE_ENUMERATE_MATERIALIZED_X = 1_000_000
MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF = 10_000
MAX_FRIABLE_ENUMERATE_FAMILY_SIZE = 200_000
_MAX_FRIABLE_ENUMERATE_SOURCE_DIGITS = 256
_MAX_FRIABLE_ENUMERATE_SOURCE_ABS = 10**_MAX_FRIABLE_ENUMERATE_SOURCE_DIGITS
_MAX_FRIABLE_ENUMERATED_BYTES = 3_000_000


def _primes_through(limit: int) -> tuple[int, ...]:
    """Return the primes at most a small admitted cutoff."""

    if limit < 2:
        return ()
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for candidate in range(2, math.isqrt(limit) + 1):
        if not sieve[candidate]:
            continue
        first_composite = candidate * candidate
        composite_count = (limit - first_composite) // candidate + 1
        sieve[first_composite : limit + 1 : candidate] = b"\x00" * composite_count
    return tuple(index for index, is_prime in enumerate(sieve) if is_prime)


def _count_from_exponent_vectors(x: int, primes: tuple[int, ...]) -> int:
    """Count the y-friable integers at most x without materializing them.

    Uses an explicit stack to avoid Python recursion-depth limits when the
    prime list is long (e.g. y = 10_000 admits ~1229 primes).
    """

    if not primes:
        return 1
    total = 0
    stack = [(0, x)]
    while stack:
        prime_index, remaining = stack.pop()
        if prime_index == len(primes):
            total += 1
            continue
        prime = primes[prime_index]
        while True:
            stack.append((prime_index + 1, remaining))
            if remaining < prime:
                break
            remaining //= prime
    return total


def _estimate_serialized_bytes(x: int, family_size: int) -> int:
    """Estimate the serialized byte cost of the result family tuple."""

    if family_size == 0:
        return 0
    max_digits = max(1, len(str(x)))
    return family_size * (max_digits + 3)


class FriableEnumerateRequest(StrictModel):
    """One exact bounded enumeration of positive ``y``-friable integers through ``x``."""

    x: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive source bound. The operation returns "
            "every positive integer at most x whose greatest prime factor is at "
            "most y. Source-sensitive work bounds must fit one exact enumeration "
            "regime."
        ),
        examples=["20"],
    )
    y: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive prime-factor cutoff. Values zero and "
            "one use the convention that only 1 is friable when x is positive."
        ),
        examples=["5"],
    )


class FriableEnumerateResult(StrictModel):
    """An exact friable family bound to its complete source pair."""

    x: BoundedInteger
    y: BoundedInteger
    family: tuple[BoundedInteger, ...]

    @model_validator(mode="after")
    def require_nonempty_or_singleton(self) -> Self:
        if not self.family:
            raise _validation_error(
                "friable_enumerate_family_must_be_nonempty_when_x_is_positive",
                "friable-enumerate family must be nonempty when x is positive",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: FriableEnumerateRequest,
        *,
        family: tuple[int, ...],
    ) -> Self:
        """Build one result after the admitted kernel established the family."""

        return cls.model_construct(
            x=request.x,
            y=request.y,
            family=tuple(str(value) for value in family),
        )


def plan_friable_enumerate(x: int, y: int) -> tuple[str, tuple[int, ...]]:
    """Validate and select one exact friable-enumerate execution regime.

    Returns the regime name and, for the generated regime, the tuple of primes
    at most ``y``.  Raises a validation error when the request cannot fit.
    """

    if x < 0 or y < 0:
        raise _validation_error(
            "friable_enumerate_sources_must_be_nonnegative",
            "friable-enumerate sources must be nonnegative",
        )
    if x >= _MAX_FRIABLE_ENUMERATE_SOURCE_ABS or y >= _MAX_FRIABLE_ENUMERATE_SOURCE_ABS:
        raise _validation_error(
            f"friable_enumerate_sources_must_have_at_most_{_MAX_FRIABLE_ENUMERATE_SOURCE_DIGITS}_decimal_digits",
            f"friable-enumerate sources must have at most {_MAX_FRIABLE_ENUMERATE_SOURCE_DIGITS} decimal digits",
        )

    # Direct regime: x is 0 (empty family is handled by the kernel), or y is so
    # large that every integer 1..x is friable, or y <= 1 (only 1 is friable).
    if x == 0:
        return "DIRECT", ()
    if y <= 1:
        return "DIRECT", ()
    if y >= x:
        return "DIRECT", ()

    # Materialized regime: small enough to scan 1..x directly.
    # Use x as the family-size upper bound (y >= x is handled above as DIRECT,
    # but for y < x the family size is at most x).
    if x <= MAX_FRIABLE_ENUMERATE_MATERIALIZED_X:
        max_family_size = x
        if (
            max_family_size <= MAX_FRIABLE_ENUMERATE_FAMILY_SIZE
            and _estimate_serialized_bytes(x, max_family_size)
            <= _MAX_FRIABLE_ENUMERATED_BYTES
        ):
            return "MATERIALIZED", ()

    # Generated regime: enumerate exponent vectors without materializing 1..x.
    if y > MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF:
        raise _validation_error(
            "generated_friable_enumerate_exceeds_the_admitted_prime_cutoff",
            "generated friable-enumerate exceeds the admitted prime cutoff",
        )

    primes = _primes_through(y)
    family_size = _count_from_exponent_vectors(x, primes)
    if family_size > MAX_FRIABLE_ENUMERATE_FAMILY_SIZE:
        raise _validation_error(
            "friable_enumerate_family_exceeds_the_result_size_budget",
            "friable-enumerate family exceeds the result-size budget",
        )
    if _estimate_serialized_bytes(x, family_size) > _MAX_FRIABLE_ENUMERATED_BYTES:
        raise _validation_error(
            "friable_enumerate_family_exceeds_the_serialized_byte_budget",
            "friable-enumerate family exceeds the serialized-byte budget",
        )
    return "GENERATED", primes


__all__ = [
    "MAX_FRIABLE_ENUMERATE_FAMILY_SIZE",
    "MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF",
    "MAX_FRIABLE_ENUMERATE_MATERIALIZED_X",
    "FriableEnumerateRequest",
    "FriableEnumerateResult",
    "plan_friable_enumerate",
]
