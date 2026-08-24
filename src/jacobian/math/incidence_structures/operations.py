"""Native exact incidence-profile and finite-trade operations."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from jacobian.math.incidence_structures._models import (
    ContainmentProfileResult,
    IncidenceMomentComparison,
    IncidenceMultiplicityDifference,
    IncidenceStructure,
    IncidenceTradeResult,
    _require_containment_profile_admitted,
    _require_incidence_trade_admitted,
)

type _SubsetProfile = tuple[tuple[tuple[str, ...], int], ...]
type _Histogram = tuple[tuple[int, int], ...]
type _ContainmentProfileData = tuple[
    _SubsetProfile,
    _Histogram,
    int,
    int,
    int,
    bool,
    int | None,
]


def _containment_profile_data(
    incidence: IncidenceStructure,
    order: int,
) -> _ContainmentProfileData:
    """Return one complete fixed-order multiplicity profile."""

    points = incidence.points
    counts: Counter[tuple[str, ...]] = Counter()
    for block in incidence.blocks:
        block_members = set(block)
        ordered_block = tuple(point for point in points if point in block_members)
        counts.update(combinations(ordered_block, order))

    subsets = tuple(combinations(points, order))
    subset_profile = tuple((subset, counts[subset]) for subset in subsets)
    histogram_counts = Counter(count for _, count in subset_profile)
    histogram = tuple(sorted(histogram_counts.items()))
    multiplicities = tuple(count for _, count in subset_profile)
    total_multiplicity = sum(multiplicities)
    min_multiplicity = min(multiplicities, default=0)
    max_multiplicity = max(multiplicities, default=0)
    is_constant = min_multiplicity == max_multiplicity
    constant_lambda = min_multiplicity if is_constant else None
    return (
        subset_profile,
        histogram,
        total_multiplicity,
        min_multiplicity,
        max_multiplicity,
        is_constant,
        constant_lambda,
    )


def containment_profile(
    incidence: IncidenceStructure,
    order: int,
) -> ContainmentProfileResult:
    """Return every fixed-order subset containment multiplicity exactly.

    The ordered point axis determines canonical subset order. Distinct block
    IDs count separately even when their member sets are equal. The result is
    complete, including zero-multiplicity subsets, and retains its source.
    """

    if not isinstance(incidence, IncidenceStructure):
        raise TypeError("incidence must be an IncidenceStructure")
    if type(order) is not int:
        raise TypeError("containment-profile order must be an integer")
    _require_containment_profile_admitted(incidence, order)
    (
        subset_profile,
        histogram,
        total_multiplicity,
        min_multiplicity,
        max_multiplicity,
        is_constant,
        constant_lambda,
    ) = _containment_profile_data(incidence, order)
    return ContainmentProfileResult(
        incidence=incidence,
        t=order,
        subset_profile=subset_profile,
        histogram=histogram,
        total_multiplicity=total_multiplicity,
        min_multiplicity=min_multiplicity,
        max_multiplicity=max_multiplicity,
        is_constant=is_constant,
        constant_lambda=constant_lambda,
    )


def _incidence_trade_data(
    left: IncidenceStructure,
    right: IncidenceStructure,
    max_order: int,
) -> tuple[int, tuple[IncidenceMomentComparison, ...], bool]:
    comparisons: list[IncidenceMomentComparison] = []
    for order in range(1, max_order + 1):
        left_profile = _containment_profile_data(left, order)
        right_profile = _containment_profile_data(right, order)
        left_subsets = left_profile[0]
        right_subsets = right_profile[0]
        differences = tuple(
            IncidenceMultiplicityDifference(
                subset=left_entry[0],
                left_multiplicity=left_entry[1],
                right_multiplicity=right_entry[1],
            )
            for left_entry, right_entry in zip(
                left_subsets,
                right_subsets,
                strict=True,
            )
            if left_entry[1] != right_entry[1]
        )
        comparisons.append(
            IncidenceMomentComparison(
                left=left,
                right=right,
                points=left.points,
                order=order,
                left_total=left_profile[2],
                right_total=right_profile[2],
                differences=differences,
                equal=not differences,
            )
        )

    comparison_tuple = tuple(comparisons)
    return (
        len(left.blocks) - len(right.blocks),
        comparison_tuple,
        all(comparison.equal for comparison in comparison_tuple),
    )


def check_incidence_trade(
    left: IncidenceStructure,
    right: IncidenceStructure,
    max_order: int,
) -> IncidenceTradeResult:
    """Compare two indexed block families through a positive subset order.

    The families must have exactly the same ordered point axis. The result
    contains complete sparse difference profiles for orders 1 through
    ``max_order``: an omitted subset has equal multiplicity on both sides.
    The zeroth block-count difference is reported separately.
    """

    if not isinstance(left, IncidenceStructure) or not isinstance(
        right, IncidenceStructure
    ):
        raise TypeError("trade sides must be IncidenceStructure values")
    if type(max_order) is not int:
        raise TypeError("trade comparison order must be an integer")
    _require_incidence_trade_admitted(left, right, max_order)
    zeroth_difference, comparisons, positive_moments_equal = _incidence_trade_data(
        left,
        right,
        max_order,
    )
    return IncidenceTradeResult(
        left=left,
        right=right,
        max_order=max_order,
        zeroth_difference=zeroth_difference,
        comparisons=comparisons,
        positive_moments_equal=positive_moments_equal,
    )


__all__ = ["check_incidence_trade", "containment_profile"]
