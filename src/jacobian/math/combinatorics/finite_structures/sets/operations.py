"""Exact native operations on finite canonical-integer sets."""

from __future__ import annotations

from collections import Counter

from jacobian._exact import ExactInteger
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import (
    MAX_FINITE_INTEGER_SET_ELEMENTS,
    MAX_FINITE_SET_COVERAGE_VALUES,
    FiniteIntegerSet,
    FiniteSetCoverageResult,
)

MAX_FINITE_SET_TOTAL_DIGITS = 10_000_000


def _parse_integers(
    elements: tuple[ExactInteger, ...], *, location: tuple[str, ...]
) -> tuple[int, ...]:
    def digits(element: ExactInteger) -> int:
        return len(format_canonical_integer(element))

    if any(
        digits(element) > CanonicalLimits().max_integer_digits for element in elements
    ):
        raise OperationDomainValidationError(
            location=location,
            code="finite_set.integer_digit_bound",
            message="finite-set elements exceed the canonical integer digit bound",
        )
    total_digits = sum(digits(element) for element in elements)
    if total_digits > MAX_FINITE_SET_TOTAL_DIGITS:
        raise OperationDomainValidationError(
            location=location,
            code="finite_set.aggregate_digit_bound",
            message="finite-set aggregate digits exceed the admitted parsing budget",
        )
    return elements


def _integers(value: FiniteIntegerSet) -> set[int]:
    return set(_parse_integers(value.elements, location=("value",)))


def _canonical(values: set[int]) -> tuple[ExactInteger, ...]:
    return tuple(sorted(values))


def _require_bounded_result(values: set[int]) -> None:
    if len(values) > MAX_FINITE_INTEGER_SET_ELEMENTS:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="finite_set.result_size_exceeded",
            message=(
                "finite-set result exceeds the "
                f"{MAX_FINITE_INTEGER_SET_ELEMENTS}-element exact-result bound"
            ),
        )


def _bounded_result(values: set[int]) -> FiniteIntegerSet:
    _require_bounded_result(values)
    return FiniteIntegerSet(elements=_canonical(values))


def exact_cover(
    scope: FiniteIntegerSet,
    values: tuple[ExactInteger, ...],
) -> FiniteSetCoverageResult:
    """Diagnose whether ``values`` covers every scope element exactly once."""

    if len(values) > MAX_FINITE_SET_COVERAGE_VALUES:
        raise OperationDomainValidationError(
            location=("values",),
            code="finite_set.coverage_values_exceeded",
            message=(
                "exact-cover input exceeds the "
                f"{MAX_FINITE_SET_COVERAGE_VALUES}-value bound"
            ),
        )
    scope_values = _integers(scope)
    counts = Counter(_parse_integers(values, location=("values",)))
    missing = scope_values - counts.keys()
    duplicates = {value for value, count in counts.items() if count > 1}
    outside = counts.keys() - scope_values
    return FiniteSetCoverageResult(
        holds=not (missing or duplicates or outside),
        missing=_canonical(missing),
        duplicates=_canonical(duplicates),
        outside=_canonical(outside),
    )


def set_union(left: FiniteIntegerSet, right: FiniteIntegerSet) -> FiniteIntegerSet:
    """Return the union of two finite integer sets."""

    return _bounded_result(_integers(left) | _integers(right))


def set_intersection(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> FiniteIntegerSet:
    """Return the intersection of two finite integer sets."""

    return _bounded_result(_integers(left) & _integers(right))


def set_difference(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> FiniteIntegerSet:
    """Return the elements of ``left`` outside ``right``."""

    return _bounded_result(_integers(left) - _integers(right))


def set_symmetric_difference(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> FiniteIntegerSet:
    """Return the elements occurring in exactly one input set."""

    return _bounded_result(_integers(left) ^ _integers(right))


def is_subset(left: FiniteIntegerSet, right: FiniteIntegerSet) -> bool:
    """Decide whether ``left`` is a subset of ``right``."""

    return _integers(left) <= _integers(right)


def is_proper_subset(left: FiniteIntegerSet, right: FiniteIntegerSet) -> bool:
    """Decide whether ``left`` is a strict subset of ``right``."""

    return _integers(left) < _integers(right)


def are_disjoint(left: FiniteIntegerSet, right: FiniteIntegerSet) -> bool:
    """Decide whether two finite sets are disjoint."""

    return _integers(left).isdisjoint(_integers(right))


def cardinality(value: FiniteIntegerSet) -> int:
    """Return the cardinality of one canonical finite set."""

    return len(value.elements)


def intersection_cardinality(left: FiniteIntegerSet, right: FiniteIntegerSet) -> int:
    """Return the cardinality of the intersection."""

    return len(_integers(left) & _integers(right))


def union_cardinality(left: FiniteIntegerSet, right: FiniteIntegerSet) -> int:
    """Return the cardinality of the union."""

    union = _integers(left) | _integers(right)
    _require_bounded_result(union)
    return len(union)


__all__ = [
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
