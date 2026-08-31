"""Typed contracts and bounded admission for exact friable-family enumeration."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

# Admission for family enumeration.  Unlike the count operation, the family
# must materialize and retain every y-friable integer, so the envelope is
# tighter.  The materialized regime reuses a local sieve; the generated regime
# enumerates prime-exponent vectors and deduplicates by sorting.

# The materialized sieve covers integers 1..x.  Keep it modest: the family
# itself can have at most x entries, each retained as a canonical string.
MAX_FRIABLE_FAMILY_MATERIALIZED_X = 50_000

# Generated regime: enumerate prime-exponent vectors whose products are at
# most x.  This regime admits large x when the generated candidate count is
# small enough to sort and retain.
MAX_FRIABLE_FAMILY_GENERATED_CUTOFF = 10_000
MAX_FRIABLE_FAMILY_GENERATED_CANDIDATES = 200_000
MAX_FRIABLE_FAMILY_GENERATED_NODES = 1_000_000

# Sources are nonnegative and rejected at 10**_MAX_FRIABLE_FAMILY_SOURCE_DIGITS
# or above, so every admitted value carries at most this many decimal digits.
_MAX_FRIABLE_FAMILY_SOURCE_DIGITS = 256
_MAX_FRIABLE_FAMILY_SOURCE_ABS = 10**_MAX_FRIABLE_FAMILY_SOURCE_DIGITS

# Bound the total number of retained integers and serialized bytes so an
# accepted result always fits the complete envelope.
MAX_FRIABLE_FAMILY_ROWS = 200_000
_MAX_FRIABLE_FAMILY_SERIALIZED_BYTES = 10 * 1024 * 1024

type _FriableFamilyRegime = Literal["DIRECT", "MATERIALIZED", "GENERATED"]


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


def _maximum_exponent(x: int, prime: int) -> int:
    """Return the largest ``exponent`` with ``prime**exponent <= x``."""

    exponent = 0
    remaining = x
    while remaining >= prime:
        remaining //= prime
        exponent += 1
    return exponent


def _estimate_generated_candidates(x: int, primes: tuple[int, ...]) -> int:
    """Upper-bound the number of generated prime-exponent vectors.

    Each vector corresponds to a unique y-friable product, so the number of
    candidates equals the number of y-friable integers in [1, x].  We compute
    this exactly using the same recursive count as the count kernel.
    """

    def visit(index: int, remaining: int) -> int:
        if index == len(primes):
            return 1
        prime = primes[index]
        total = 0
        while True:
            total += visit(index + 1, remaining)
            if remaining < prime:
                return total
            remaining //= prime

    return visit(0, x)


def plan_friable_family(  # noqa: C901
    x: int, y: int
) -> tuple[_FriableFamilyRegime, tuple[int, ...]]:
    """Validate and select one exact friable-family execution regime.

    Returns the regime label and, for the generated regime, the tuple of
    primes at most y used by the enumerator.  Raises a PydanticCustomError on
    any admission failure.
    """

    if x < 0 or y < 0:
        raise _validation_error(
            "friable_family_sources_must_be_nonnegative",
            "friable-family sources must be nonnegative",
        )
    # Constant-size direct regimes are independent of the source magnitude.
    if x == 0 or y <= 1:
        return "DIRECT", ()
    if x >= _MAX_FRIABLE_FAMILY_SOURCE_ABS or y >= _MAX_FRIABLE_FAMILY_SOURCE_ABS:
        raise _validation_error(
            f"friable_family_sources_must_have_at_most_{_MAX_FRIABLE_FAMILY_SOURCE_DIGITS}_decimal_digits",
            f"friable-family sources must have at most {_MAX_FRIABLE_FAMILY_SOURCE_DIGITS} decimal digits",
        )
    # When y >= x every positive integer through x is friable.  This shortcut
    # still materializes the complete family, so admit its rows and wire size.
    if y >= x:
        estimated_bytes = 128 + x * (len(str(x)) + 3)
        if x > MAX_FRIABLE_FAMILY_ROWS:
            raise _validation_error(
                "friable_family_exceeds_the_row_budget",
                "friable family exceeds the row budget",
            )
        if estimated_bytes > _MAX_FRIABLE_FAMILY_SERIALIZED_BYTES:
            raise _validation_error(
                "friable_family_exceeds_the_serialized_budget",
                "friable family exceeds the serialized result budget",
            )
        return "DIRECT", ()

    # Materialized regime: sieve 1..x and collect friable entries.
    if x <= MAX_FRIABLE_FAMILY_MATERIALIZED_X:
        family_size = x  # at most x entries
        if family_size <= MAX_FRIABLE_FAMILY_ROWS:
            return "MATERIALIZED", ()

    if y > MAX_FRIABLE_FAMILY_GENERATED_CUTOFF:
        raise _validation_error(
            "generated_friable_family_exceeds_the_admitted_prime_cutoff",
            "generated friable family exceeds the admitted prime cutoff",
        )

    primes = _primes_through(y)
    nodes_per_pass = 1
    prefix_box = 1
    for prime in primes:
        prefix_box *= _maximum_exponent(x, prime) + 1
        nodes_per_pass += prefix_box
        if nodes_per_pass > MAX_FRIABLE_FAMILY_GENERATED_NODES:
            raise _validation_error(
                "generated_friable_family_exceeds_the_search_node_budget",
                "generated friable family exceeds the search-node budget",
            )

    if nodes_per_pass * 2 > MAX_FRIABLE_FAMILY_GENERATED_NODES:
        raise _validation_error(
            "generated_friable_family_exceeds_the_search_node_budget",
            "generated friable family exceeds the search-node budget",
        )
    candidate_count = _estimate_generated_candidates(x, primes)
    if candidate_count > MAX_FRIABLE_FAMILY_GENERATED_CANDIDATES:
        raise _validation_error(
            "generated_friable_family_exceeds_the_candidate_budget",
            "generated friable family exceeds the candidate budget",
        )
    if candidate_count > MAX_FRIABLE_FAMILY_ROWS:
        raise _validation_error(
            "generated_friable_family_exceeds_the_row_budget",
            "generated friable family exceeds the row budget",
        )
    estimated_bytes = 128 + candidate_count * (len(str(x)) + 3)
    if estimated_bytes > _MAX_FRIABLE_FAMILY_SERIALIZED_BYTES:
        raise _validation_error(
            "generated_friable_family_exceeds_the_serialized_budget",
            "generated friable family exceeds the serialized result budget",
        )

    return "GENERATED", primes


class FriableFamilyRequest(StrictModel):
    """One exact bounded family of positive y-friable integers through x."""

    x: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive source bound. Easy boundary cases "
            "may use up to 256 decimal digits; other cases must fit an exact "
            "enumeration regime selected from the source-sensitive work bounds."
        ),
        examples=["100"],
    )
    y: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive prime-factor cutoff. Values zero "
            "and one use the convention that only 1 is friable when x is positive."
        ),
        examples=["5"],
    )


class FriableFamilyResult(StrictModel):
    """The complete ordered family of positive y-friable integers through x."""

    x: BoundedInteger
    y: BoundedInteger
    family: tuple[BoundedInteger, ...] = Field(max_length=MAX_FRIABLE_FAMILY_ROWS)

    @model_validator(mode="after")
    def require_nonempty_or_consistent(self) -> Self:
        count = parse_canonical_integer(self.x)
        if count < 0:
            raise _validation_error(
                "friable_family_x_must_be_nonnegative",
                "friable family x must be nonnegative",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: FriableFamilyRequest, *, family: list[int]) -> Self:
        """Build one result after the admitted kernel established its family."""

        from jacobian.canonical import format_canonical_integer

        return cls.model_construct(
            x=request.x,
            y=request.y,
            family=tuple(format_canonical_integer(value) for value in family),
        )


__all__ = [
    "MAX_FRIABLE_FAMILY_GENERATED_CANDIDATES",
    "MAX_FRIABLE_FAMILY_GENERATED_CUTOFF",
    "MAX_FRIABLE_FAMILY_MATERIALIZED_X",
    "MAX_FRIABLE_FAMILY_ROWS",
    "_MAX_FRIABLE_FAMILY_SOURCE_ABS",
    "_MAX_FRIABLE_FAMILY_SOURCE_DIGITS",
    "FriableFamilyRequest",
    "FriableFamilyResult",
    "plan_friable_family",
]
