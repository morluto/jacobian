"""Tests for number_theory.divisibility_poset.compute."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.number_theory.divisibility_poset import compute_divisibility_poset
from jacobian.math.number_theory.divisibility_poset._models import (
    MAX_DIVISIBILITY_POSET_ELEMENTS,
    DivisibilityPosetRequest,
    IntegerDivisibilityPosetResult,
)


def _compute(elements: list[str]) -> IntegerDivisibilityPosetResult:
    request = DivisibilityPosetRequest.model_validate(
        {"source_set": {"elements": elements}}
    )
    return compute_divisibility_poset(request.source_set)


def _source_map(result: IntegerDivisibilityPosetResult) -> dict[str, str]:
    return {e.label: e.source_integer for e in result.element_sources}


def test_divisors_of_12() -> None:
    result = _compute(["1", "2", "3", "4", "6", "12"])
    sources = _source_map(result)
    assert len(result.poset.elements) == 6
    assert set(sources.values()) == {"1", "2", "3", "4", "6", "12"}
    assert tuple(result.poset.elements) == ("e0", "e1", "e2", "e3", "e4", "e5")
    pairs = {(p.lower, p.upper) for p in result.poset.strict_order_pairs}
    one_label = next(label for label, src in sources.items() if src == "1")
    for other_src in ("2", "3", "4", "6", "12"):
        other_label = next(label for label, src in sources.items() if src == other_src)
        assert (one_label, other_label) in pairs
    two_label = next(label for label, src in sources.items() if src == "2")
    for other_src in ("4", "6", "12"):
        other_label = next(label for label, src in sources.items() if src == other_src)
        assert (two_label, other_label) in pairs
    three_label = next(label for label, src in sources.items() if src == "3")
    for other_src in ("6", "12"):
        other_label = next(label for label, src in sources.items() if src == other_src)
        assert (three_label, other_label) in pairs
    four_label = next(label for label, src in sources.items() if src == "4")
    twelve_label = next(label for label, src in sources.items() if src == "12")
    assert (four_label, twelve_label) in pairs
    six_label = next(label for label, src in sources.items() if src == "6")
    assert (six_label, twelve_label) in pairs
    assert (two_label, one_label) not in pairs
    assert (twelve_label, one_label) not in pairs
    assert (one_label, four_label) in pairs
    assert (two_label, twelve_label) in pairs
    assert (two_label, three_label) not in pairs
    assert (three_label, two_label) not in pairs
    assert (three_label, four_label) not in pairs
    assert (four_label, three_label) not in pairs


def test_single_element() -> None:
    result = _compute(["7"])
    assert len(result.poset.elements) == 1
    assert result.poset.strict_order_pairs == ()
    assert _source_map(result) == {"e0": "7"}


def test_empty_set() -> None:
    result = _compute([])
    assert result.poset.elements == ()
    assert result.poset.strict_order_pairs == ()
    assert result.element_sources == ()


def test_antichain_coprime_pair() -> None:
    result = _compute(["2", "3"])
    assert len(result.poset.elements) == 2
    assert result.poset.strict_order_pairs == ()
    assert len(result.poset.incomparable_pairs) == 1


def test_chain() -> None:
    result = _compute(["2", "4", "8", "16"])
    pairs = {(p.lower, p.upper) for p in result.poset.strict_order_pairs}
    assert len(pairs) == 6
    sources = _source_map(result)

    def label(src: str) -> str:
        return next(label for label, s in sources.items() if s == src)

    l2, l4, l8, l16 = label("2"), label("4"), label("8"), label("16")
    for lower, upper in [
        (l2, l4),
        (l2, l8),
        (l2, l16),
        (l4, l8),
        (l4, l16),
        (l8, l16),
    ]:
        assert (lower, upper) in pairs


def test_unsorted_input_is_canonical() -> None:
    result_a = _compute(["3", "1", "12", "6", "4", "2"])
    result_b = _compute(["1", "2", "3", "4", "6", "12"])
    assert result_a.poset.poset_digest == result_b.poset.poset_digest
    assert result_a.poset.elements == result_b.poset.elements
    assert _source_map(result_a) == _source_map(result_b)


def test_source_set_preserved() -> None:
    elements = ["1", "2", "3", "4", "6", "12"]
    result = _compute(elements)
    assert tuple(result.source_set.elements) == tuple(elements)


def test_large_digit_integers_divisibility() -> None:
    big1 = "1" + "0" * 100
    big2 = "1" + "0" * 200
    result = _compute([big1, big2])
    sources = _source_map(result)
    pairs = {(p.lower, p.upper) for p in result.poset.strict_order_pairs}
    big1_label = next(label for label, s in sources.items() if s == big1)
    big2_label = next(label for label, s in sources.items() if s == big2)
    assert (big1_label, big2_label) in pairs
    assert (big2_label, big1_label) not in pairs


def test_non_positive_rejected() -> None:
    with pytest.raises(ValidationError):
        DivisibilityPosetRequest.model_validate(
            {"source_set": {"elements": ["0", "1", "2"]}}
        )
    with pytest.raises(ValidationError):
        DivisibilityPosetRequest.model_validate(
            {"source_set": {"elements": ["-1", "1", "2"]}}
        )


def test_native_path_rejects_nonpositive_source_values() -> None:
    source = FiniteIntegerSet(elements=("-1", "1"))
    with pytest.raises(OperationDomainValidationError, match="positive integers"):
        compute_divisibility_poset(source)


def test_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        DivisibilityPosetRequest.model_validate(
            {"source_set": {"elements": ["2", "2", "3"]}}
        )


def test_too_many_elements_rejected() -> None:
    elements = [str(i) for i in range(1, MAX_DIVISIBILITY_POSET_ELEMENTS + 2)]
    with pytest.raises(ValidationError):
        DivisibilityPosetRequest.model_validate({"source_set": {"elements": elements}})


def test_exactly_max_elements() -> None:
    elements = [str(i) for i in range(1, MAX_DIVISIBILITY_POSET_ELEMENTS + 1)]
    result = _compute(elements)
    assert len(result.poset.elements) == MAX_DIVISIBILITY_POSET_ELEMENTS


def test_poset_is_valid_finite_poset() -> None:
    result = _compute(["1", "2", "4", "8"])
    assert result.poset.poset_digest.startswith("sha256:")
    assert result.poset.graded is True
    assert result.poset.ranks is not None
    assert len(result.poset.ranks) == 4


def test_minimal_and_maximal_elements() -> None:
    result = _compute(["1", "2", "4"])
    sources = _source_map(result)
    one_label = next(label for label, s in sources.items() if s == "1")
    four_label = next(label for label, s in sources.items() if s == "4")
    assert result.poset.minimal_elements == (one_label,)
    assert result.poset.maximal_elements == (four_label,)


def test_transitive_closure_completeness() -> None:
    result = _compute(["2", "4", "8"])
    sources = _source_map(result)
    two_label = next(label for label, s in sources.items() if s == "2")
    four_label = next(label for label, s in sources.items() if s == "4")
    eight_label = next(label for label, s in sources.items() if s == "8")
    pairs = {(p.lower, p.upper) for p in result.poset.strict_order_pairs}
    assert (two_label, four_label) in pairs
    assert (four_label, eight_label) in pairs
    assert (two_label, eight_label) in pairs


def test_catalog_discovery() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = [t.operation_id for t in BUILTIN_TOOLS]
    assert "number_theory.divisibility_poset.compute" in ids


def test_element_sources_covers_all_elements() -> None:
    result = _compute(["1", "2", "4"])
    poset_element_set = set(result.poset.elements)
    source_labels = {e.label for e in result.element_sources}
    assert poset_element_set == source_labels


def test_source_set_in_result_is_canonical() -> None:
    elements = ["6", "12", "1", "3"]
    result = _compute(elements)
    assert tuple(result.source_set.elements) == tuple(elements)


def test_power_of_two_chain() -> None:
    result = _compute(["1", "2", "4", "8", "16", "32"])
    n = len(result.poset.elements)
    assert len(result.poset.strict_order_pairs) == n * (n - 1) // 2
    assert len(result.poset.incomparable_pairs) == 0
