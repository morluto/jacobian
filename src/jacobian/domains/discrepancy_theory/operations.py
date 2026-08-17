"""Domain adapter for finite set-system discrepancy operations."""

from __future__ import annotations

import itertools

from jacobian.contracts.discrepancy_theory import (
    DiscrepancyEvalRequest,
    DiscrepancyEvalResult,
    DiscrepancyOptimumRequest,
    DiscrepancyOptimumResult,
)


def _max_absolute_imbalance(
    signed_sums: tuple[int, ...],
) -> int:
    """Return the maximum absolute value among signed sums (0 for empty)."""

    return max((abs(value) for value in signed_sums), default=0)


def compute_discrepancy(request: DiscrepancyEvalRequest) -> DiscrepancyEvalResult:
    """Compute the signed sum on every set and the maximum absolute imbalance.

    For a coloring ``c`` and a set ``S`` the signed sum is
    ``sum(c[i] for i in S)``. The maximum absolute imbalance is the maximum
    of the absolute signed sums across all sets; it is zero when the family
    is empty.
    """
    signed_sums = tuple(
        sum(request.coloring[element] for element in subset)
        for subset in request.set_system.sets
    )
    return DiscrepancyEvalResult(
        signed_sums=signed_sums,
        max_absolute_imbalance=_max_absolute_imbalance(signed_sums),
    )


def compute_optimal_discrepancy(
    request: DiscrepancyOptimumRequest,
) -> DiscrepancyOptimumResult:
    """Search over all 2^n colorings for the minimum maximum discrepancy.

    The ground set size is bounded by ``MAX_GROUND_SET`` so the exhaustive
    search over ``itertools.product`` stays a bounded combinatorial
    computation. When the ground set is empty there is exactly one coloring
    (the empty coloring) with discrepancy zero.
    """
    n = request.set_system.ground_set_size
    sets = request.set_system.sets

    if n == 0:
        return DiscrepancyOptimumResult(
            optimal_coloring=(),
            optimal_discrepancy=0,
            exhaustive=True,
        )

    best_coloring: tuple[int, ...] | None = None
    best_discrepancy: int | None = None
    for values in itertools.product((-1, 1), repeat=n):
        coloring = values
        max_imbalance = 0
        for subset in sets:
            signed_sum = sum(coloring[element] for element in subset)
            absolute = -signed_sum if signed_sum < 0 else signed_sum
            if absolute > max_imbalance:
                max_imbalance = absolute
                if best_discrepancy is not None and max_imbalance >= best_discrepancy:
                    break
        else:
            if best_discrepancy is None or max_imbalance < best_discrepancy:
                best_discrepancy = max_imbalance
                best_coloring = coloring
                if best_discrepancy == 0:
                    break

    assert best_coloring is not None
    assert best_discrepancy is not None
    return DiscrepancyOptimumResult(
        optimal_coloring=best_coloring,
        optimal_discrepancy=best_discrepancy,
        exhaustive=True,
    )
