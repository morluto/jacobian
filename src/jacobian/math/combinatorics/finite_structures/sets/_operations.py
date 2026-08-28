"""Exact finite-set operations over canonical integers."""

from __future__ import annotations

from jacobian.math.combinatorics.finite_structures.sets import operations as native
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteSetBooleanResult,
    FiniteSetCardinalityResult,
    FiniteSetCoverageRequest,
    FiniteSetCoverageResult,
    FiniteSetElementListResult,
    FiniteSetPairRequest,
)


def decide_exact_cover(request: FiniteSetCoverageRequest) -> FiniteSetCoverageResult:
    """Decide whether the supplied values contain every scope element exactly once."""

    return native.exact_cover(request.scope, request.values)


def set_union(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_union(request.left, request.right).elements
    )


def set_intersection(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_intersection(request.left, request.right).elements
    )


def set_difference(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_difference(request.left, request.right).elements
    )


def set_symmetric_difference(
    request: FiniteSetPairRequest,
) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_symmetric_difference(request.left, request.right).elements
    )


def decide_subset(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    return FiniteSetBooleanResult(holds=native.is_subset(request.left, request.right))


def decide_proper_subset(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    return FiniteSetBooleanResult(
        holds=native.is_proper_subset(request.left, request.right)
    )


def decide_disjoint(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    return FiniteSetBooleanResult(
        holds=native.are_disjoint(request.left, request.right)
    )


def left_cardinality(request: FiniteSetPairRequest) -> FiniteSetCardinalityResult:
    return FiniteSetCardinalityResult(cardinality=native.cardinality(request.left))


def intersection_cardinality(
    request: FiniteSetPairRequest,
) -> FiniteSetCardinalityResult:
    return FiniteSetCardinalityResult(
        cardinality=native.intersection_cardinality(request.left, request.right)
    )


def union_cardinality(request: FiniteSetPairRequest) -> FiniteSetCardinalityResult:
    return FiniteSetCardinalityResult(
        cardinality=native.union_cardinality(request.left, request.right)
    )
