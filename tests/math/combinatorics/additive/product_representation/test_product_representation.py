from __future__ import annotations

import json
from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.product_representation.operations import (
    compute_product_representation_profile,
)
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet


def _set(elements: Iterable[int]) -> FiniteIntegerSet:
    return FiniteIntegerSet(elements=tuple(elements))


def test_simple_product() -> None:
    left = _set([1, 2])
    right = _set([3, 4])
    result = compute_product_representation_profile(left, right)
    assert result.support_cardinality == 4
    entries = {e.product: e.multiplicity for e in result.entries}
    assert entries == {3: 1, 4: 1, 6: 1, 8: 1}


def test_empty_left() -> None:
    left = _set([])
    right = _set([1, 2])
    result = compute_product_representation_profile(left, right)
    assert result.support_cardinality == 0
    assert result.entries == ()


def test_duplicates_in_product() -> None:
    left = _set([2, 3])
    right = _set([6, 4])
    result = compute_product_representation_profile(left, right)
    entries = {e.product: e.multiplicity for e in result.entries}
    assert entries[12] == 2
    assert entries[8] == 1
    assert entries[18] == 1


def test_sorted_output() -> None:
    left = _set([5, 3, 7])
    right = _set([2, 6])
    result = compute_product_representation_profile(left, right)
    products = [e.product for e in result.entries]
    assert products == [6, 10, 14, 18, 30, 42]


def test_negative_integers() -> None:
    left = _set([-1, 2])
    right = _set([3, -4])
    result = compute_product_representation_profile(left, right)
    entries = {e.product: e.multiplicity for e in result.entries}
    # -1*3=-3, -1*-4=4, 2*3=6, 2*-4=-8
    assert entries == {-3: 1, 4: 1, 6: 1, -8: 1}


def test_result_preserves_source() -> None:
    left = _set([1])
    right = _set([1])
    result = compute_product_representation_profile(left, right)
    assert result.left == left
    assert result.right == right
    assert result.support_cardinality == 1
    assert result.entries[0].product == 1
    assert result.entries[0].multiplicity == 1


def test_large_canonical_products_remain_exact_and_deliverable() -> None:
    large = 10**5_000 - 1
    result = compute_product_representation_profile(_set([large]), _set([1]))

    assert result.entries[0].product == large
    assert result.model_validate_json(result.model_dump_json()) == result


def test_cartesian_work_is_rejected_before_enumeration() -> None:
    left = _set(range(317))
    right = _set(range(316))

    with pytest.raises(OperationDomainValidationError, match="pair work bound"):
        compute_product_representation_profile(left, right)


def test_worst_case_digit_work_is_rejected_before_product_construction() -> None:
    prefix = (10**32_765 - 1) * 1_000
    left = _set(prefix + index for index in range(150))
    right = _set(prefix + index for index in range(3))

    with pytest.raises(OperationDomainValidationError, match="digit work bound"):
        compute_product_representation_profile(left, right)


def test_large_operands_are_rejected_before_integer_parsing() -> None:
    operand = "9" * 600_001

    with pytest.raises(ValidationError):
        FiniteIntegerSet.model_validate_json(json.dumps({"elements": [operand]}))
