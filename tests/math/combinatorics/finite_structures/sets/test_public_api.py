"""Public API and publication contract for finite integer sets."""

import pytest

from jacobian.canonical import CanonicalLimits, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures import sets
from jacobian.math.combinatorics.finite_structures.sets._tools import TOOLS


def test_exact_public_api_symbols() -> None:
    expected = (
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
    )

    assert tuple(sets.__all__) == expected
    assert all(hasattr(sets, name) for name in sets.__all__)


def test_all_eleven_finite_set_operations_are_published() -> None:
    assert len(TOOLS) == 11
    assert len({operation.operation_id for operation in TOOLS}) == 11


def test_native_set_operations_compose_through_canonical_values() -> None:
    left = sets.FiniteIntegerSet(elements=(3, 1))
    right = sets.FiniteIntegerSet(elements=(2, 3))

    union = sets.set_union(left, right)
    assert union == sets.FiniteIntegerSet(elements=(1, 2, 3))
    assert sets.cardinality(union) == 3
    assert sets.set_intersection(left, right).elements == (3,)
    assert sets.set_difference(left, right).elements == (1,)
    assert sets.set_symmetric_difference(left, right).elements == (1, 2)
    assert sets.is_subset(left, union) is True
    assert sets.is_proper_subset(left, union) is True
    assert sets.are_disjoint(left, sets.FiniteIntegerSet(elements=(4,))) is True
    assert sets.intersection_cardinality(left, right) == 1
    assert sets.union_cardinality(left, right) == 3


def test_native_exact_cover_retains_complete_diagnostics() -> None:
    scope = sets.FiniteIntegerSet(elements=(1, 2))

    result = sets.exact_cover(scope, (1, 1, 3))

    assert result.holds is False
    assert result.missing == (2,)
    assert result.duplicates == (1,)
    assert result.outside == (3,)


def test_native_exact_cover_bounds_values_before_integer_parsing() -> None:
    oversized = "9" * (CanonicalLimits().max_integer_digits + 1)

    with pytest.raises(OperationDomainValidationError) as caught:
        sets.exact_cover(
            sets.FiniteIntegerSet(elements=()),
            (parse_canonical_integer(oversized),) * 2,
        )

    assert caught.value.errors()[0]["loc"] == ("values",)
    assert caught.value.errors()[0]["type"] == "finite_set.integer_digit_bound"
