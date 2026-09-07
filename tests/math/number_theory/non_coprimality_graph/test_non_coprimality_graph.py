from __future__ import annotations

from math import gcd

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.number_theory.non_coprimality_graph._models import (
    NonCoprimalityGraphRequest,
)
from jacobian.math.number_theory.non_coprimality_graph.operations import (
    construct_non_coprimality_graph,
    verify_non_coprimality_graph,
)


def test_fixture() -> None:
    """For {2,3,4,6}, edges are 2-4, 2-6, 3-6, 4-6."""
    result = construct_non_coprimality_graph((2, 3, 4, 6))
    edges = set(result.graph.edges)
    assert ("2", "4") in edges
    assert ("2", "6") in edges
    assert ("3", "6") in edges
    assert ("4", "6") in edges
    assert len(result.graph.edges) == 4


def test_all_coprime() -> None:
    """Set of pairwise coprime integers has no edges."""
    result = construct_non_coprimality_graph((2, 3, 5, 7))
    assert len(result.graph.edges) == 0


def test_single_vertex() -> None:
    """Single integer has no edges."""
    result = construct_non_coprimality_graph((7,))
    assert len(result.graph.vertices) == 1
    assert len(result.graph.edges) == 0


def test_lexical_graph_edges_follow_the_retained_source_axis() -> None:
    """Numeric pair ordering must not replace the source's vertex ordering."""
    result = construct_non_coprimality_graph((10, 2))
    assert result.integers.elements == (10, 2)
    assert result.graph.vertices == ("10", "2")
    assert result.graph.edges == (("10", "2"),)
    assert verify_non_coprimality_graph(
        type(result).model_validate_json(result.model_dump_json())
    )


def test_replay_gcd() -> None:
    """Every edge satisfies gcd > 1, every non-edge has gcd = 1."""
    ints = (6, 10, 15, 7, 35)
    result = construct_non_coprimality_graph(ints)
    edge_set = set(result.graph.edges)
    for i in range(len(ints)):
        for j in range(i + 1, len(ints)):
            a, b = str(ints[i]), str(ints[j])
            expected = (min(a, b), max(a, b))
            should_have = gcd(ints[i], ints[j]) > 1
            assert (expected in edge_set) == should_have


def test_rejects_non_positive() -> None:
    with pytest.raises(OperationDomainValidationError):
        construct_non_coprimality_graph((0, 2))


def test_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        NonCoprimalityGraphRequest(integers=FiniteIntegerSet(elements=(2, 2)))


def test_request_uses_canonical_integer_strings() -> None:
    request = NonCoprimalityGraphRequest(
        integers=FiniteIntegerSet(elements=(2, 3, 4, 6))
    )
    assert request.integers.elements == (2, 3, 4, 6)


def test_vertex_labels_preserve_integers() -> None:
    """Vertex labels are the canonical integer strings."""
    result = construct_non_coprimality_graph((11, 13, 17))
    assert set(result.graph.vertices) == {"11", "13", "17"}


def test_result_preserves_source() -> None:
    """Result retains the source integers."""
    ints = (3, 5, 7)
    result = construct_non_coprimality_graph(ints)
    assert result.integers.elements == ints


def test_canonical_integer_strings_preserve_large_values() -> None:
    result = construct_non_coprimality_graph((2**53 + 1, 2**53 + 3))
    assert result.integers.elements == (2**53 + 1, 2**53 + 3)


def test_serialized_graph_claim_verifies_and_rejects_forgery() -> None:
    result = construct_non_coprimality_graph((2, 3, 4, 6))
    restored = type(result).model_validate_json(result.model_dump_json())
    assert verify_non_coprimality_graph(restored)
    forged = restored.model_dump(mode="json")
    forged["graph"]["edges"].pop()
    assert not verify_non_coprimality_graph(
        type(result).model_validate_json(__import__("json").dumps(forged))
    )


def test_native_rejects_oversized_integer_before_gcd() -> None:
    with pytest.raises(ValueError, match="digit bound"):
        construct_non_coprimality_graph((10**256,))
