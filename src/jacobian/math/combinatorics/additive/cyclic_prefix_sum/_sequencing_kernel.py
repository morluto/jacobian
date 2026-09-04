"""Private deterministic kernel for forbidden-prefix cyclic sequencing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SequencingKernelResult:
    """Unbound search outcome used by the trusted result factory."""

    status: Literal["FOUND", "EXHAUSTED", "UNKNOWN"]
    ordering_indices: tuple[int, ...] = ()
    states_explored: int = 0


def search_forbidden_prefix_sequencing(
    elements: tuple[tuple[int, ...], ...],
    moduli: tuple[int, ...],
    forbidden_values: tuple[tuple[int, ...], ...],
    first_index: int | None,
    search_node_limit: int,
) -> SequencingKernelResult:
    """Exhaust the source-index permutation tree in deterministic order.

    A state is one partial or complete source-index sequence. At each state,
    available indices are tried in increasing order. The root and terminal
    states count toward the node budget; every arithmetic update adds one
    coordinate per group axis. ``UNKNOWN`` is returned as soon as the node
    allowance is exhausted, so it can never be promoted to a negative
    conclusion.
    """

    element_count = len(elements)
    if element_count == 0:
        return SequencingKernelResult(status="EXHAUSTED", states_explored=1)
    if first_index is None:
        initial_indices = tuple(range(element_count))
    else:
        initial_indices = (first_index,)

    forbidden = set(forbidden_values)
    zero = tuple(0 for _ in moduli)
    visited = 0
    ordering: list[int] = []
    seen_prefixes: set[tuple[int, ...]] = set()

    def add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            (x + y) % modulus for x, y, modulus in zip(left, right, moduli, strict=True)
        )

    def visit(
        current_sum: tuple[int, ...],
        used: frozenset[int],
    ) -> SequencingKernelResult:
        nonlocal visited
        if visited >= search_node_limit:
            return SequencingKernelResult(status="UNKNOWN", states_explored=visited)
        visited += 1

        if len(ordering) == element_count:
            # A terminal sum is not a proper prefix sum. In particular, a
            # zero-sum source returns to zero without a collision.
            return SequencingKernelResult(
                status="FOUND",
                ordering_indices=tuple(ordering),
                states_explored=visited,
            )

        remaining = element_count - len(ordering) - 1
        for index in initial_indices if not ordering else range(element_count):
            if index in used:
                continue
            next_sum = add(current_sum, elements[index])
            # Only proper prefixes carry sequencing constraints; a terminal
            # sum is accepted even when it is zero or forbidden.
            if remaining > 0 and (
                next_sum == zero or next_sum in forbidden or next_sum in seen_prefixes
            ):
                continue
            ordering.append(index)
            if remaining > 0:
                seen_prefixes.add(next_sum)
            result = visit(next_sum, used | {index})
            if remaining > 0:
                seen_prefixes.remove(next_sum)
            ordering.pop()
            if result.status != "EXHAUSTED":
                return result
        return SequencingKernelResult(status="EXHAUSTED", states_explored=visited)

    return visit(zero, frozenset())


__all__ = ["SequencingKernelResult", "search_forbidden_prefix_sequencing"]
