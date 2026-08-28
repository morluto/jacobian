"""Native exact incidence-profile and finite-trade operations."""

from __future__ import annotations

from jacobian.math.combinatorics.designs.incidence_structures._models import (
    ContainmentProfileResult,
    IncidenceMomentComparison,
    IncidenceStructure,
    IncidenceTradeResult,
    _require_containment_profile_admitted,
    _require_incidence_trade_admitted,
)
from jacobian.math.combinatorics.designs.incidence_structures._operations import (
    _containment_profile_data,
    _incidence_trade_data,
)


def containment_profile(
    incidence: IncidenceStructure, order: int
) -> ContainmentProfileResult:
    """Return every fixed-order subset containment multiplicity exactly."""

    if not isinstance(incidence, IncidenceStructure):
        raise TypeError("incidence must be an IncidenceStructure")
    if type(order) is not int:
        raise TypeError("containment-profile order must be an integer")
    _require_containment_profile_admitted(incidence, order)
    return ContainmentProfileResult._from_kernel(
        incidence, order, _containment_profile_data(incidence, order)
    )


def check_incidence_trade(
    left: IncidenceStructure, right: IncidenceStructure, max_order: int
) -> IncidenceTradeResult:
    """Compare two indexed block families through a positive subset order."""

    if not isinstance(left, IncidenceStructure) or not isinstance(
        right, IncidenceStructure
    ):
        raise TypeError("trade sides must be IncidenceStructure values")
    if type(max_order) is not int:
        raise TypeError("trade comparison order must be an integer")
    _require_incidence_trade_admitted(left, right, max_order)
    zeroth_difference, comparisons, _positive_moments_equal = _incidence_trade_data(
        left, right, max_order
    )
    return IncidenceTradeResult._from_kernel(
        left, right, max_order, zeroth_difference, comparisons
    )


def verify_incidence_moment_comparison(
    comparison: IncidenceMomentComparison,
) -> bool:
    """Verify an externally supplied moment comparison within its admission."""

    try:
        _require_containment_profile_admitted(comparison.left, comparison.order)
        _require_containment_profile_admitted(comparison.right, comparison.order)
    except ValueError:
        return False
    left_profile = _containment_profile_data(comparison.left, comparison.order)
    right_profile = _containment_profile_data(comparison.right, comparison.order)
    expected_differences = tuple(
        (left_entry[0], left_entry[1], right_entry[1])
        for left_entry, right_entry in zip(
            left_profile[0], right_profile[0], strict=True
        )
        if left_entry[1] != right_entry[1]
    )
    actual_differences = tuple(
        (difference.subset, difference.left_multiplicity, difference.right_multiplicity)
        for difference in comparison.differences
    )
    return (
        comparison.left_total == left_profile[2]
        and comparison.right_total == right_profile[2]
        and actual_differences == expected_differences
    )


__all__ = [
    "check_incidence_trade",
    "containment_profile",
    "verify_incidence_moment_comparison",
]
