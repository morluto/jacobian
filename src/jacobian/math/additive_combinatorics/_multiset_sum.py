"""Bounded exact kernel for fixed-arity unordered multiset sums."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from itertools import combinations_with_replacement

MAX_SOURCE_SIZE = 384
MAX_ELEMENT_DIGITS = 64
# Arity is bounded by its decimal representation, not its mathematical
# magnitude: admission rejects every costly request through the candidate-work
# and support preflights, while compact cases (singleton or empty sources)
# stay admissible at any represented arity. The digit bound alone fixes the
# cost of admission arithmetic (binomial preflight, sum products) and of the
# predicted worst-case sum below. The magnitude ceiling is the interoperable
# JSON integer range, because the published schema exposes it as ``maximum``.
MAX_ARITY_DIGITS = 18
MAX_ARITY = (1 << 53) - 1
# The operation enumerates once and source-bound result validation replays once,
# so an accepted call performs at most twice this many coordinate steps.
MAX_ENUMERATION_WORK = 20_000_000

# A row containing a signed sum of at most MAX_RESULT_DIGITS digits and an
# eight-digit multiplicity (the work preflight caps candidates below 10^8)
# stays under 128 canonical JSON bytes. Reserving 64 KiB for the maximum
# source, the optional window, and scalar fields keeps every admitted exact
# result below an 8 MiB owner-local budget and the transport's 10 MiB limit.
RESULT_BUDGET_BYTES = 8 * 1024 * 1024
_RESULT_RESERVE_BYTES = 64 * 1024
_ENTRY_WIRE_BYTES = 128
MAX_SUPPORT_SIZE = (RESULT_BUDGET_BYTES - _RESULT_RESERVE_BYTES) // _ENTRY_WIRE_BYTES
MAX_RESULT_DIGITS = MAX_ELEMENT_DIGITS + MAX_ARITY_DIGITS
MAX_INTEGER_LENGTH = MAX_RESULT_DIGITS + 1


def candidate_count(source_size: int, arity: int) -> int:
    if arity == 0:
        return 1
    if source_size == 0:
        return 0
    # C(n+k-1, n-1) takes at most MAX_SOURCE_SIZE-1 multiplicative steps for an
    # admitted source, even when arity is large; preflight never expands it.
    return math.comb(source_size + arity - 1, source_size - 1)


def enumeration_work(
    values: tuple[int, ...],
    arity: int,
    bounds: tuple[int, int] | None,
    candidates: int,
) -> int:
    if candidates == 0 or arity == 0:
        return 0
    # A window missing every attainable sum [arity*values[0], arity*values[-1]]
    # has an exactly empty profile; count_sums returns before inspecting any
    # candidate, so admission charges zero enumeration work for that scope.
    if bounds is not None and (
        arity * values[0] > bounds[1] or arity * values[-1] < bounds[0]
    ):
        return 0
    return candidates * min(len(values), arity)


def support_bound(
    values: tuple[int, ...],
    arity: int,
    bounds: tuple[int, int] | None,
    candidates: int,
) -> int:
    if candidates == 0:
        return 0
    if arity == 0:
        minimum_sum = maximum_sum = 0
    else:
        minimum_sum = arity * values[0]
        maximum_sum = arity * values[-1]
    if bounds is None:
        intersection_lower = minimum_sum
        intersection_upper = maximum_sum
    else:
        intersection_lower = max(bounds[0], minimum_sum)
        intersection_upper = min(bounds[1], maximum_sum)
    if intersection_lower > intersection_upper:
        return 0
    return min(candidates, intersection_upper - intersection_lower + 1)


def _bar_position_tuples(pool_size: int, bars: int) -> Iterator[tuple[int, ...]]:
    # Lazy equivalent of combinations(range(pool_size), bars) that never
    # materializes the slot pool: iteration keeps only the O(bars) working
    # state, so an admitted request's intermediate memory is bounded by its
    # source size rather than by its arity.
    if bars == 0:
        yield ()
        return
    if bars > pool_size:
        return
    positions = list(range(bars))
    yield tuple(positions)
    while True:
        for index in range(bars - 1, -1, -1):
            if positions[index] != index + pool_size - bars:
                break
        else:
            return
        positions[index] += 1
        for follower in range(index + 1, bars):
            positions[follower] = positions[follower - 1] + 1
        yield tuple(positions)


def count_sums(
    values: tuple[int, ...],
    arity: int,
    bounds: tuple[int, int] | None,
) -> Counter[int]:
    def in_scope(value: int) -> bool:
        return bounds is None or bounds[0] <= value <= bounds[1]

    if arity == 0:
        return Counter({0: 1}) if in_scope(0) else Counter()
    if not values:
        return Counter()
    if len(values) == 1:
        value = arity * values[0]
        return Counter({value: 1}) if in_scope(value) else Counter()

    # Every candidate sum lies in [arity*values[0], arity*values[-1]], so a
    # window missing that whole interval has an exactly empty profile without
    # inspecting any candidate.
    if bounds is not None and (
        arity * values[0] > bounds[1] or arity * values[-1] < bounds[0]
    ):
        return Counter()

    counts: Counter[int] = Counter()
    if arity <= len(values):
        for terms in combinations_with_replacement(values, arity):
            value = sum(terms)
            if in_scope(value):
                counts[value] += 1
        return counts

    # When n < k, stars-and-bars emits the same C(n+k-1, k) multisets
    # while materializing n multiplicities instead of k repeated values; the
    # bar positions are generated lazily in lexicographic order without
    # snapshotting the slot range.
    slots = arity + len(values) - 1
    for bars in _bar_position_tuples(slots, len(values) - 1):
        value = 0
        previous = -1
        for index, bar in enumerate(bars):
            value += (bar - previous - 1) * values[index]
            previous = bar
        value += (slots - previous - 1) * values[-1]
        if in_scope(value):
            counts[value] += 1
    return counts
