"""Exact finite-set operations over canonical integers."""

from __future__ import annotations

from collections import Counter

from jacobian.contracts.finite_sets import (
    FiniteSetBooleanResult,
    FiniteSetCardinalityResult,
    FiniteSetCoverageRequest,
    FiniteSetCoverageResult,
    FiniteSetElementListResult,
    FiniteSetPairRequest,
)


def decide_exact_cover(request: FiniteSetCoverageRequest) -> FiniteSetCoverageResult:
    """Decide whether the supplied values contain every scope element exactly once."""

    scope = {int(element) for element in request.scope.elements}
    counts = Counter(int(element) for element in request.values)
    missing = scope - counts.keys()
    duplicates = {value for value, count in counts.items() if count > 1}
    outside = counts.keys() - scope
    return FiniteSetCoverageResult(
        holds=not (missing or duplicates or outside),
        missing=tuple(str(value) for value in sorted(missing)),
        duplicates=tuple(str(value) for value in sorted(duplicates)),
        outside=tuple(str(value) for value in sorted(outside)),
    )


def _pair(request: FiniteSetPairRequest) -> tuple[set[int], set[int]]:
    pair = request
    return (
        {int(element) for element in pair.left.elements},
        {int(element) for element in pair.right.elements},
    )


def _element_list(values: set[int]) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=tuple(str(value) for value in sorted(values)),
    )


def set_union(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    left, right = _pair(request)
    return _element_list(left | right)


def set_intersection(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    left, right = _pair(request)
    return _element_list(left & right)


def set_difference(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    left, right = _pair(request)
    return _element_list(left - right)


def set_symmetric_difference(
    request: FiniteSetPairRequest,
) -> FiniteSetElementListResult:
    left, right = _pair(request)
    return _element_list(left ^ right)


def decide_subset(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    left, right = _pair(request)
    return FiniteSetBooleanResult(holds=left <= right)


def decide_proper_subset(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    left, right = _pair(request)
    return FiniteSetBooleanResult(holds=left < right)


def decide_disjoint(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    left, right = _pair(request)
    return FiniteSetBooleanResult(holds=left.isdisjoint(right))


def left_cardinality(request: FiniteSetPairRequest) -> FiniteSetCardinalityResult:
    left, _ = _pair(request)
    return FiniteSetCardinalityResult(cardinality=len(left))


def intersection_cardinality(
    request: FiniteSetPairRequest,
) -> FiniteSetCardinalityResult:
    left, right = _pair(request)
    return FiniteSetCardinalityResult(cardinality=len(left & right))


def union_cardinality(request: FiniteSetPairRequest) -> FiniteSetCardinalityResult:
    left, right = _pair(request)
    return FiniteSetCardinalityResult(cardinality=len(left | right))
