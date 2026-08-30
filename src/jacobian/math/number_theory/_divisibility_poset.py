"""Finite divisibility poset construction and declaration."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.math.number_theory._divisibility_poset_kernels import (
    construct_divisibility_poset,
)
from jacobian.math.number_theory._divisibility_poset_models import (
    DivisibilityPosetRequest,
    DivisibilityPosetResult,
)
from jacobian.math.number_theory._support import number_theory_operation


def compute_divisibility_poset(
    request: DivisibilityPosetRequest,
) -> DivisibilityPosetResult:
    """Return the canonical proper-divisibility poset of a finite set of integers."""
    data = construct_divisibility_poset(request.values)
    return DivisibilityPosetResult(
        values=request.values,
        strict_order_pairs=data.strict_order_pairs,
    )


DIVISIBILITY_POSET_OPERATION = number_theory_operation(
    "integer.divisibility_poset.compute",
    "Construct finite divisibility poset",
    "Given a finite set of positive integers, return the canonical "
    "proper-divisibility poset where a < b exactly when a divides b "
    "and a != b. The result is a source-labelled directed relation.",
    DivisibilityPosetRequest,
    DivisibilityPosetResult,
    compute_divisibility_poset,
    "number-theory",
    "divisibility",
    "poset",
    "exact",
    examples=(
        example(
            "divisibility_236",
            "For {2,3,6}, the proper-divisibility poset has 2<6 and 3<6; "
            "values must be positive canonical decimal integers.",
            {"values": ["2", "3", "6"]},
        ),
    ),
)


__all__ = ["DIVISIBILITY_POSET_OPERATION", "compute_divisibility_poset"]
