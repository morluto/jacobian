"""Private exact kernel for one indexed subset-sum target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _WitnessNode:
    previous: _WitnessNode | None
    index: int


def _solve_subset_sum_target(
    values: tuple[int, ...],
    target: int,
    *,
    allow_empty_subset: bool,
) -> tuple[int, ...] | None:
    """Return the canonical index subset summing to ``target``, if one exists.

    Each reachable sum retains the smallest binary incidence mask, with source
    index zero as the least-significant bit.  This gives repeated values and
    zeros a stable, deterministic witness without treating them as a set.
    """

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
            states[value] = _WitnessNode(previous=None, index=index)

        for subtotal, witness in previous:
            candidate_sum = subtotal + value
            if candidate_sum not in states:
                states[candidate_sum] = _WitnessNode(
                    previous=witness,
                    index=index,
                )

    if target not in states:
        return None

    indices: list[int] = []
    witness = states[target]
    while witness is not None:
        indices.append(witness.index)
        witness = witness.previous
    indices.reverse()
    return tuple(indices)
