"""Homogeneous progression set system constructor."""

from __future__ import annotations

from jacobian.math.combinatorics.discrepancy.homogeneous_progression._models import (
    HomogeneousProgressionResult,
)

__all__ = ["construct_homogeneous_progression_set_system"]


def construct_homogeneous_progression_set_system(
    n: int,
) -> HomogeneousProgressionResult:
    """Construct the homogeneous progression set system on [n].

    The ground set is indexed by 0..n-1 (representing 1..n). The sets are
    the zero-based images of homogeneous progressions {d, 2d, ..., kd}
    for every d, k >= 1 with dk <= n, in canonical order.
    """
    sets: list[tuple[int, ...]] = []
    for d in range(1, n + 1):
        k = 1
        while d * k <= n:
            progression = tuple(d * i - 1 for i in range(1, k + 1))
            sets.append(progression)
            k += 1

    return HomogeneousProgressionResult(
        n=n,
        ground_set_size=n,
        sets=tuple(sets),
    )
