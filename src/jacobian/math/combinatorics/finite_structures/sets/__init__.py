"""Exact finite canonical-integer sets and operations."""

from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteIntegerSet,
    FiniteSetCoverageResult,
)
from jacobian.math.combinatorics.finite_structures.sets.operations import (
    are_disjoint,
    cardinality,
    exact_cover,
    intersection_cardinality,
    is_proper_subset,
    is_subset,
    set_difference,
    set_intersection,
    set_symmetric_difference,
    set_union,
    union_cardinality,
)

__all__ = [
    "FiniteIntegerSet",
    "FiniteSetCoverageResult",
    "are_disjoint",
    "cardinality",
    "exact_cover",
    "intersection_cardinality",
    "is_proper_subset",
    "is_subset",
    "set_difference",
    "set_intersection",
    "set_symmetric_difference",
    "set_union",
    "union_cardinality",
]
