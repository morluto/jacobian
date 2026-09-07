"""Tests for integer.divisibility_poset.compute."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    FinitePoset,
)
from jacobian.math.number_theory._divisibility_poset import (
    compute_divisibility_poset,
    divisibility_poset,
)
from jacobian.math.number_theory._divisibility_poset_models import (
    MAX_DIVISIBILITY_SET_SIZE,
    DivisibilityPosetRequest,
)

MAX_ADMITTED_ELEMENTS = min(MAX_DIVISIBILITY_SET_SIZE, MAX_POSET_ELEMENTS)


def _compute(elements: list[int]) -> FinitePoset:
    request = DivisibilityPosetRequest.model_validate(
        {"values": {"elements": elements}}
    )
    return compute_divisibility_poset(request)


def test_divisors_of_12() -> None:
    result = _compute([1, 2, 3, 4, 6, 12])
    assert result.elements == ("1", "12", "2", "3", "4", "6")
    pairs = {(pair.lower, pair.upper) for pair in result.strict_order_pairs}
    assert ("1", "2") in pairs
    assert ("1", "12") in pairs
    assert ("2", "4") in pairs
    assert ("2", "6") in pairs
    assert ("2", "12") in pairs
    assert ("3", "6") in pairs
    assert ("3", "12") in pairs
    assert ("4", "12") in pairs
    assert ("6", "12") in pairs
    assert ("2", "1") not in pairs
    assert ("2", "3") not in pairs
    assert ("3", "4") not in pairs


def test_single_element() -> None:
    result = _compute([7])
    assert result.elements == ("7",)
    assert result.strict_order_pairs == ()


def test_empty_set() -> None:
    result = _compute([])
    assert result.elements == ()
    assert result.strict_order_pairs == ()


def test_antichain_coprime_pair() -> None:
    result = _compute([2, 3])
    assert result.elements == ("2", "3")
    assert result.strict_order_pairs == ()
    assert len(result.incomparable_pairs) == 1


def test_chain() -> None:
    result = _compute([2, 4, 8, 16])
    pairs = {(pair.lower, pair.upper) for pair in result.strict_order_pairs}
    assert pairs == {
        ("2", "4"),
        ("2", "8"),
        ("2", "16"),
        ("4", "8"),
        ("4", "16"),
        ("8", "16"),
    }


def test_unsorted_input_is_canonical() -> None:
    result_a = _compute([3, 1, 12, 6, 4, 2])
    result_b = _compute([1, 2, 3, 4, 6, 12])
    assert result_a.poset_digest == result_b.poset_digest
    assert result_a.elements == result_b.elements


def test_native_path_rejects_nonpositive_source_values() -> None:
    source = FiniteIntegerSet(elements=(-1, 1))
    with pytest.raises(OperationDomainValidationError, match="positive integers"):
        divisibility_poset(source)


def test_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        DivisibilityPosetRequest.model_validate(
            {"values": {"elements": ["2", "2", "3"]}}
        )


def test_operation_element_limit_is_not_request_structure() -> None:
    elements = list(range(1, MAX_ADMITTED_ELEMENTS + 2))
    request = DivisibilityPosetRequest.model_validate(
        {"values": {"elements": elements}}
    )

    with pytest.raises(OperationDomainValidationError, match="between 0 and"):
        compute_divisibility_poset(request)


def test_exactly_max_elements() -> None:
    elements = list(range(1, MAX_ADMITTED_ELEMENTS + 1))
    result = _compute(elements)
    assert len(result.elements) == MAX_ADMITTED_ELEMENTS


def test_poset_is_valid_finite_poset() -> None:
    result = _compute([1, 2, 4, 8])
    assert result.poset_digest.startswith("sha256:")
    assert result.graded is True
    assert result.ranks is not None
    assert len(result.ranks) == 4


def test_minimal_and_maximal_elements() -> None:
    result = _compute([1, 2, 4])
    assert result.minimal_elements == ("1",)
    assert result.maximal_elements == ("4",)


def test_transitive_closure_completeness() -> None:
    result = _compute([2, 4, 8])
    pairs = {(pair.lower, pair.upper) for pair in result.strict_order_pairs}
    assert pairs == {("2", "4"), ("4", "8"), ("2", "8")}


def test_catalog_discovery() -> None:
    ids = [tool.operation_id for tool in BUILTIN_TOOLS]
    assert "integer.divisibility_poset.compute" in ids
    assert "number_theory.divisibility_poset.compute" not in ids


def test_power_of_two_chain() -> None:
    result = _compute([1, 2, 4, 8, 16, 32])
    count = len(result.elements)
    assert len(result.strict_order_pairs) == count * (count - 1) // 2
    assert result.incomparable_pairs == ()
