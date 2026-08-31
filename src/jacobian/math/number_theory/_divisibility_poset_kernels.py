"""Exact kernel for finite divisibility poset construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DivisibilityPosetData:
    """Private canonical data returned by the poset kernel."""

    strict_order_pairs: tuple[tuple[str, str], ...]


def construct_divisibility_poset(values: tuple[str, ...]) -> DivisibilityPosetData:
    """Return the canonical proper-divisibility poset of a finite set.

    The poset has a < b exactly when a divides b and a != b.
    """
    int_values = [int(v) for v in values]
    n = len(int_values)

    pairs: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if int_values[j] % int_values[i] == 0:
                pairs.append((values[i], values[j]))

    return DivisibilityPosetData(strict_order_pairs=tuple(pairs))
