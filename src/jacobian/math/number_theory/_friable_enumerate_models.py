"""Typed contracts and bounded admission for exact friable-family enumeration."""

from __future__ import annotations

import math
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.combinatorics.finite_structures.sets._models import (
    MAX_FINITE_INTEGER_SET_ELEMENTS,
    FiniteIntegerSet,
)
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

# The enumeration shares the generated-exponent-vector regime with
# ``number_theory.friable.count.compute``.  Materializing a complete ordered
# family requires at most one generated candidate per prime-exponent tuple
# plus sorting/deduplication, so admission reuses the same source-sensitive
# bounds.
MAX_FRIABLE_ENUMERATE_MATERIALIZED_X = 1_000_000
MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF = 10_000
MAX_FRIABLE_ENUMERATE_FAMILY_SIZE = MAX_FINITE_INTEGER_SET_ELEMENTS
_MAX_FRIABLE_ENUMERATE_SOURCE_DIGITS = 256
_MAX_FRIABLE_ENUMERATE_SOURCE_ABS = 10**_MAX_FRIABLE_ENUMERATE_SOURCE_DIGITS
# The final operation result is delivered under the canonical output boundary;
# the finite-set carrier itself does not enforce the half-budget constant.
_MAX_FRIABLE_ENUMERATED_BYTES = CanonicalLimits().max_output_bytes
_MAX_FRIABLE_ENUMERATE_COUNT_NODES = 2_000_000


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


def _generate_from_exponent_vectors(x: int, primes: tuple[int, ...]) -> tuple[int, ...]:
    """Generate the y-friable integers at most x once for admission and output.

    Uses an explicit stack to avoid Python recursion-depth limits when the
    prime list is long (e.g. y = 10_000 admits ~1229 primes).
    """

    if not primes:
        return (1,)
    values: list[int] = []
    nodes = 0
    stack = [(0, 1)]
    while stack:
        nodes += 1
        if nodes > _MAX_FRIABLE_ENUMERATE_COUNT_NODES:
            raise _validation_error(
                "friable_enumerate_exceeds_the_search_node_budget",
                "friable-enumerate presolve exceeds the search-node budget",
            )
        prime_index, product = stack.pop()
        if prime_index == len(primes):
            values.append(product)
            if len(values) > MAX_FRIABLE_ENUMERATE_FAMILY_SIZE:
                raise _validation_error(
                    "friable_enumerate_family_exceeds_the_result_size_budget",
                    "friable-enumerate family exceeds the result-size budget",
                )
            continue
        prime = primes[prime_index]
        while True:
            stack.append((prime_index + 1, product))
            if product > x // prime:
                break
            product *= prime
    return tuple(sorted(values))


def _estimate_serialized_bytes(x: int, family_size: int) -> int:
    """Conservatively estimate the serialized byte cost before materialization."""

    if family_size == 0:
        return 0
    max_digits = max(1, len(str(x)))
    return family_size * (max_digits + 3)


def _exact_serialized_bytes(x: int, y: int, family: tuple[int, ...]) -> int:
    """Measure the complete final result in the canonical wire format."""

    payload = {
        "family": {"elements": [str(value) for value in family]},
        "x": str(x),
        "y": str(y),
    }
    return len(
        canonicalize_json(
            payload,
            limits=CanonicalLimits(max_output_bytes=(1 << 63) - 1),
        )
    )


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
    family: FiniteIntegerSet

    @model_validator(mode="after")
    def require_nonempty_or_singleton(self) -> Self:
        x = int(self.x)
        y = int(self.y)
        values = tuple(int(value) for value in self.family.elements)
        if x == 0 and self.family.elements:
            raise _validation_error(
                "friable_enumerate_zero_source_must_be_empty",
                "friable-enumerate family must be empty when x is zero",
            )
        if any(value < 1 or value > x for value in values):
            raise _validation_error(
                "friable_enumerate_family_out_of_range",
                "every friable-enumerate family member must lie in [1, x]",
            )
        if values != tuple(sorted(values)):
            raise _validation_error(
                "friable_enumerate_family_must_be_increasing",
                "friable-enumerate family must be in strictly increasing order",
            )
        if x != 0 and y <= 1 and self.family.elements != ("1",):
            raise _validation_error(
                "friable_enumerate_small_cutoff_is_singleton",
                "positive x with y at most one must have family {1}",
            )
        if not self.family.elements and x != 0:
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
            family=FiniteIntegerSet(
                elements=tuple(str(value) for value in family),
            ),
        )


def _count_friable_bounded(x: int, y: int) -> int:
    """Count y-friable integers in 1..x, stopping at the family-size limit + 1.

    This is a result-sensitive admission aid for the materialized regime:
    it uses the same sieve logic as the kernel but stops counting as soon as
    the family exceeds the result-size budget, avoiding the coarse x upper bound.
    """

    if x == 0:
        return 0
    if y <= 1:
        return 1 if x >= 1 else 0

    is_friable = bytearray(b"\x01") * (x + 1)
    is_friable[0] = 0
    is_prime = bytearray(b"\x01") * (x + 1)
    is_prime[0:2] = b"\x00\x00"

    count = 0
    limit = MAX_FRIABLE_ENUMERATE_FAMILY_SIZE + 1
    for candidate in range(2, x + 1):
        if not is_prime[candidate]:
            continue
        if candidate * candidate <= x:
            first = candidate * candidate
            is_prime[first : x + 1 : candidate] = b"\x00" * (
                (x - first) // candidate + 1
            )
        if candidate > y:
            is_friable[candidate : x + 1 : candidate] = b"\x00" * (x // candidate)

    for value in range(1, x + 1):
        if is_friable[value]:
            count += 1
            if count > limit:
                return count
    return count


def _materialize_friable_bounded(x: int, y: int) -> tuple[int, ...]:
    """Return the y-friable integers in 1..x, or an oversized tuple.

    Reuses the same sieve as _count_friable_bounded but returns the family
    so the kernel can skip the duplicate sieve pass.
    """

    if x == 0:
        return ()
    if y <= 1:
        return (1,) if x >= 1 else ()

    is_friable = bytearray(b"\x01") * (x + 1)
    is_friable[0] = 0
    is_prime = bytearray(b"\x01") * (x + 1)
    is_prime[0:2] = b"\x00\x00"

    for candidate in range(2, x + 1):
        if not is_prime[candidate]:
            continue
        if candidate * candidate <= x:
            first = candidate * candidate
            is_prime[first : x + 1 : candidate] = b"\x00" * (
                (x - first) // candidate + 1
            )
        if candidate > y:
            is_friable[candidate : x + 1 : candidate] = b"\x00" * (x // candidate)

    return tuple(v for v in range(1, x + 1) if is_friable[v])


def plan_friable_enumerate(
    x: int, y: int, *, enforce_transport: bool = True
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """Validate and select one exact friable-enumerate execution regime.

    Returns the regime name, generated prime tuple, and (for the generated
    regime) the materialized family so execution can reuse the admission pass.
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
        return "DIRECT", (), ()
    if y <= 1:
        return "DIRECT", (), (1,)
    if y >= x:
        if x > MAX_FRIABLE_ENUMERATE_FAMILY_SIZE:
            raise _validation_error(
                "friable_enumerate_family_exceeds_the_result_size_budget",
                "friable-enumerate family exceeds the result-size budget",
            )
        if enforce_transport and (
            _exact_serialized_bytes(x, y, tuple(range(1, x + 1)))
            > _MAX_FRIABLE_ENUMERATED_BYTES
        ):
            raise _validation_error(
                "friable_enumerate_family_exceeds_the_serialized_byte_budget",
                "friable-enumerate family exceeds the serialized-byte budget",
            )
        return "DIRECT", (), tuple(range(1, x + 1))

    # Materialized regime: small enough to scan 1..x directly.
    # Count the actual friable family size instead of using the coarse x
    # upper bound, so requests whose result is safely bounded are admitted.
    if x <= MAX_FRIABLE_ENUMERATE_MATERIALIZED_X:
        family = _materialize_friable_bounded(x, y)
        family_size = len(family)
        if family_size > MAX_FRIABLE_ENUMERATE_FAMILY_SIZE:
            raise _validation_error(
                "friable_enumerate_family_exceeds_the_result_size_budget",
                "friable-enumerate family exceeds the result-size budget",
            )
        if (
            enforce_transport
            and _exact_serialized_bytes(x, y, family) > _MAX_FRIABLE_ENUMERATED_BYTES
        ):
            raise _validation_error(
                "friable_enumerate_family_exceeds_the_serialized_byte_budget",
                "friable-enumerate family exceeds the serialized-byte budget",
            )
        return "MATERIALIZED", (), family

    # Generated regime: enumerate exponent vectors without materializing 1..x.
    if y > MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF:
        raise _validation_error(
            "generated_friable_enumerate_exceeds_the_admitted_prime_cutoff",
            "generated friable-enumerate exceeds the admitted prime cutoff",
        )

    primes = _primes_through(y)
    family = _generate_from_exponent_vectors(x, primes)
    family_size = len(family)
    if family_size > MAX_FRIABLE_ENUMERATE_FAMILY_SIZE:
        raise _validation_error(
            "friable_enumerate_family_exceeds_the_result_size_budget",
            "friable-enumerate family exceeds the result-size budget",
        )
    if (
        enforce_transport
        and _exact_serialized_bytes(x, y, family) > _MAX_FRIABLE_ENUMERATED_BYTES
    ):
        raise _validation_error(
            "friable_enumerate_family_exceeds_the_serialized_byte_budget",
            "friable-enumerate family exceeds the serialized-byte budget",
        )
    return "GENERATED", primes, family


__all__ = [
    "MAX_FRIABLE_ENUMERATE_FAMILY_SIZE",
    "MAX_FRIABLE_ENUMERATE_GENERATED_CUTOFF",
    "MAX_FRIABLE_ENUMERATE_MATERIALIZED_X",
    "FriableEnumerateRequest",
    "FriableEnumerateResult",
    "plan_friable_enumerate",
]
