"""Typed contracts and bounded admission for exact friable counting."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

# Friable counting has two exact, result-sensitive execution regimes. The
# materialized regime uses one bytearray for primality and one for friability;
# its work bound covers both the operation and result-validation replay. The
# generated regime enumerates prime-exponent prefixes only when a conservative
# prefix-box bound covers both passes.
MAX_FRIABLE_MATERIALIZED_X = 1_000_000
MAX_FRIABLE_GENERATED_CUTOFF = 10_000
_MAX_FRIABLE_MATERIALIZED_TOTAL_STEPS = 82_000_000
_MAX_FRIABLE_MATERIALIZED_BYTES = 3_000_000
_MAX_FRIABLE_GENERATED_TOTAL_NODES = 1_000_000
# Sources are nonnegative and rejected at ``10**_MAX_FRIABLE_SOURCE_DIGITS`` or
# above, so every admitted value carries at most this many decimal digits.
_MAX_FRIABLE_SOURCE_DIGITS = 256
_MAX_FRIABLE_SOURCE_ABS = 10**_MAX_FRIABLE_SOURCE_DIGITS

type _FriableRegime = Literal["DIRECT", "MATERIALIZED", "GENERATED"]


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


def _plan_friable_count(x: int, y: int) -> tuple[_FriableRegime, tuple[int, ...]]:
    """Validate and select one exact friable-count execution regime."""

    if x < 0 or y < 0:
        raise _validation_error(
            "friable_count_sources_must_be_nonnegative",
            "friable-count sources must be nonnegative",
        )
    if x >= _MAX_FRIABLE_SOURCE_ABS or y >= _MAX_FRIABLE_SOURCE_ABS:
        raise _validation_error(
            f"friable_count_sources_must_have_at_most_{_MAX_FRIABLE_SOURCE_DIGITS}_decimal_digits",
            f"friable-count sources must have at most {_MAX_FRIABLE_SOURCE_DIGITS} decimal digits",
        )
    if x == 0 or y <= 1 or y >= x:
        return "DIRECT", ()

    # Each materialized pass marks at most two harmonic series of multiples,
    # plus one scan. Result validation replays the full exact computation.
    per_pass_steps = x * (2 * x.bit_length() + 1)
    materialized_bytes = 2 * (x + 1) + x // 2
    if (
        x <= MAX_FRIABLE_MATERIALIZED_X
        and 2 * per_pass_steps <= _MAX_FRIABLE_MATERIALIZED_TOTAL_STEPS
        and materialized_bytes <= _MAX_FRIABLE_MATERIALIZED_BYTES
    ):
        return "MATERIALIZED", ()

    if y > MAX_FRIABLE_GENERATED_CUTOFF:
        raise _validation_error(
            "generated_friable_counting_exceeds_the_admitted_prime_cutoff",
            "generated friable counting exceeds the admitted prime cutoff",
        )

    primes = _primes_through(y)
    nodes_per_pass = 1
    prefix_box = 1
    for prime in primes:
        prefix_box *= _maximum_exponent(x, prime) + 1
        nodes_per_pass += prefix_box
        if 2 * nodes_per_pass > _MAX_FRIABLE_GENERATED_TOTAL_NODES:
            raise _validation_error(
                "generated_friable_counting_exceeds_the_search_node_budget",
                "generated friable counting exceeds the search-node budget",
            )
    return "GENERATED", primes


class FriableCountRequest(StrictModel):
    """One exact bounded count of positive ``y``-friable integers through ``x``."""

    x: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive source bound. Easy boundary cases may "
            f"use up to {_MAX_FRIABLE_SOURCE_DIGITS} decimal digits; other cases must fit an exact counting "
            "regime selected from the source-sensitive work bounds."
        ),
        examples=["100"],
    )
    y: BoundedInteger = Field(
        description=(
            "Canonical nonnegative inclusive prime-factor cutoff. Values zero and "
            "one use the convention that only 1 is friable when x is positive."
        ),
        examples=["5"],
    )

    @model_validator(mode="after")
    def require_admitted_exact_count(self) -> Self:
        _plan_friable_count(
            parse_canonical_integer(self.x),
            parse_canonical_integer(self.y),
        )
        return self


class FriableCountResult(StrictModel):
    """An exact friable count bound to its complete source pair."""

    x: BoundedInteger
    y: BoundedInteger
    count: BoundedInteger

    @model_validator(mode="after")
    def bind_exact_count_to_sources(self) -> Self:
        from jacobian.math.number_theory._friable_operations import count_friable

        x = parse_canonical_integer(self.x)
        y = parse_canonical_integer(self.y)
        count = parse_canonical_integer(self.count)
        if count < 0:
            raise _validation_error(
                "friable_count_must_be_nonnegative", "friable count must be nonnegative"
            )
        if count != count_friable(x, y):
            raise _validation_error(
                "friable_count_does_not_match_the_retained_sources",
                "friable count does not match the retained sources",
            )
        return self


__all__ = [
    "MAX_FRIABLE_GENERATED_CUTOFF",
    "MAX_FRIABLE_MATERIALIZED_X",
    "_MAX_FRIABLE_SOURCE_ABS",
    "_MAX_FRIABLE_SOURCE_DIGITS",
    "FriableCountRequest",
    "FriableCountResult",
    "_plan_friable_count",
]
