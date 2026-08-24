"""Private exact kernel for one indexed subset-sum target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _WitnessNode:
    previous: _WitnessNode | None
    index: int


def _witness_indices(witness: _WitnessNode | None) -> tuple[int, ...]:
    indices: list[int] = []
    while witness is not None:
        indices.append(witness.index)
        witness = witness.previous
    indices.reverse()
    return tuple(indices)


def _attained_sum_interval(
    values: tuple[int, ...],
    *,
    allow_empty_subset: bool,
) -> tuple[int, int]:
    """Return the closed interval containing every admissible subset sum.

    With the empty subset admissible this is
    ``[sum(negatives), sum(positives)]``.  Without it the empty subset's zero
    is not a witness, so a source with no negative items attains nothing below
    ``min(values)`` and a source with no positive items attains nothing above
    ``max(values)``; a strictly one-signed source therefore resolves a zero
    target immediately.
    """

    if allow_empty_subset:
        return (
            sum(value for value in values if value < 0),
            sum(value for value in values if value > 0),
        )
    if not values:
        # No admissible subset exists at all; only zero can still reach the
        # loop below, which resolves it as unattained.
        return 0, 0
    negative_items = [value for value in values if value < 0]
    positive_items = [value for value in values if value > 0]
    lower = sum(negative_items) if negative_items else min(values)
    upper = sum(positive_items) if positive_items else max(values)
    return lower, upper


def _solve_subset_sum_target(
    values: tuple[int, ...],
    target: int,
    *,
    allow_empty_subset: bool,
) -> tuple[int, ...] | None:
    """Return the canonical index subset summing to ``target``, if one exists.

    A target outside the attained interval misses every admissible subset
    sum, so it is resolved exactly without building any state; admission
    resolves the same requests before expansion.  With the empty subset
    inadmissible that interval excludes the empty subset's zero, so strictly
    one-signed sources resolve a zero target immediately.

    Each reachable sum retains the smallest binary incidence mask, with source
    index zero as the least-significant bit.  This gives repeated values and
    zeros a stable, deterministic witness without treating them as a set.
    Every insertion returns immediately when it attains ``target``, so the
    search stops at the first prefix that resolves it; witnesses inside that
    prefix carry masks strictly below any witness of the later expansion.
    """

    lower, upper = _attained_sum_interval(
        values,
        allow_empty_subset=allow_empty_subset,
    )
    if not lower <= target <= upper:
        # Outside the attained interval no admissible subset sum can equal
        # the target.
        return None
    states: dict[int, _WitnessNode | None] = {0: None} if allow_empty_subset else {}
    if target in states:
        # The retained empty witness is already the globally smallest mask,
        # and insert-only updates can never improve it later.
        return ()
    for index, value in enumerate(values):
        previous = tuple(states.items())

        # Every retained witness uses only earlier indices, so its incidence
        # mask is smaller than any witness containing this index.  For a fixed
        # item, distinct prior sums also produce distinct candidate sums.
        # Insert-only updates therefore retain the globally smallest mask
        # without constructing exponentially wide bit masks.
        if not allow_empty_subset and value not in states:
            if value == target:
                return (index,)
            states[value] = _WitnessNode(previous=None, index=index)

        for subtotal, witness in previous:
            candidate_sum = subtotal + value
            if candidate_sum == target:
                return _witness_indices(_WitnessNode(previous=witness, index=index))
            if candidate_sum not in states:
                states[candidate_sum] = _WitnessNode(
                    previous=witness,
                    index=index,
                )

    # Every attaining insertion above returned early, so exhausting the loop
    # establishes exact non-attainment across the whole source.
    return None
