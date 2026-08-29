"""Homogeneous progression set system kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.discrepancy._models import (
    MAX_GROUND_SET,
    FiniteSetSystem,
)
from jacobian.math.combinatorics.discrepancy.homogeneous_progression._models import (
    HomogeneousProgressionResult,
)

__all__ = ["construct_homogeneous_progression_set_system"]


def construct_homogeneous_progression_set_system(
    n: int,
) -> HomogeneousProgressionResult:
    """Construct the set system of all homogeneous arithmetic progressions in [n].

    For every d,k >= 1 with dk <= n, the set is (d-1, 2d-1, ..., kd-1)
    (zero-based indices representing 1..n).
    The sets are returned in canonical order: by d, then by k.
    """
    if not 0 <= n <= MAX_GROUND_SET:
        raise OperationDomainValidationError(
            location=("n",),
            code="discrepancy.homogeneous_progression_ground_set_size",
            message=(
                "homogeneous progression systems support ground-set sizes "
                f"from 0 through {MAX_GROUND_SET}"
            ),
        )
    sets: list[tuple[int, ...]] = []

    for d in range(1, n + 1):
        k = 1
        while d * k <= n:
            # Zero-based: values d, 2d, ..., kd -> 0-based: d-1, 2d-1, ..., kd-1
            subset = tuple(d * j - 1 for j in range(1, k + 1))
            sets.append(subset)
            k += 1

    set_system = FiniteSetSystem(
        ground_set_size=n,
        sets=tuple(sets),
    )

    return HomogeneousProgressionResult(
        n=n,
        set_system=set_system,
    )
