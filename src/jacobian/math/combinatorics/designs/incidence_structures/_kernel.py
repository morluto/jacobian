"""Exact containment-profile kernels shared by native and catalog operations."""

from collections import Counter
from itertools import combinations

from jacobian.math.combinatorics.designs.incidence_structures._models import (
    IncidenceMomentComparison,
    IncidenceMultiplicityDifference,
    IncidenceStructure,
)

type _SubsetProfile = tuple[tuple[tuple[str, ...], int], ...]
type _Histogram = tuple[tuple[int, int], ...]
type ContainmentProfileData = tuple[
    _SubsetProfile, _Histogram, int, int, int, bool, int | None
]


def containment_profile_data(
    incidence: IncidenceStructure, order: int
) -> ContainmentProfileData:
    """Return one complete fixed-order multiplicity profile."""

    points = incidence.points
    counts: Counter[tuple[str, ...]] = Counter()
    for block in incidence.blocks:
        block_members = set(block)
        counts.update(
            combinations(
                tuple(point for point in points if point in block_members), order
            )
        )
    subset_profile = tuple(
        (subset, counts[subset]) for subset in combinations(points, order)
    )
    histogram = tuple(sorted(Counter(count for _, count in subset_profile).items()))
    multiplicities = tuple(count for _, count in subset_profile)
    minimum = min(multiplicities, default=0)
    maximum = max(multiplicities, default=0)
    return (
        subset_profile,
        histogram,
        sum(multiplicities),
        minimum,
        maximum,
        minimum == maximum,
        minimum if minimum == maximum else None,
    )


def incidence_trade_data(
    left: IncidenceStructure, right: IncidenceStructure, max_order: int
) -> tuple[int, tuple[IncidenceMomentComparison, ...], bool]:
    comparisons: list[IncidenceMomentComparison] = []
    for order in range(1, max_order + 1):
        left_profile = containment_profile_data(left, order)
        right_profile = containment_profile_data(right, order)
        differences = tuple(
            IncidenceMultiplicityDifference(
                subset=left_entry[0],
                left_multiplicity=left_entry[1],
                right_multiplicity=right_entry[1],
            )
            for left_entry, right_entry in zip(
                left_profile[0], right_profile[0], strict=True
            )
            if left_entry[1] != right_entry[1]
        )
        comparisons.append(
            IncidenceMomentComparison._from_kernel(
                left, right, order, left_profile[2], right_profile[2], differences
            )
        )
    comparison_tuple = tuple(comparisons)
    return (
        len(left.blocks) - len(right.blocks),
        comparison_tuple,
        all(comparison.equal for comparison in comparison_tuple),
    )


__all__ = ["ContainmentProfileData", "containment_profile_data", "incidence_trade_data"]
