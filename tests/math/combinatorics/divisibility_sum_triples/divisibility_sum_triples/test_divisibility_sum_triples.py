from __future__ import annotations

from jacobian.math.combinatorics.divisibility_sum_triples.operations import (
    construct_divisibility_sum_triples,
)


def test_small_interval() -> None:
    result = construct_divisibility_sum_triples(1, 5)
    assert result.lower == 1
    assert result.upper == 5
    assert len(result.hypergraph.vertices) == 5


def test_single_element() -> None:
    result = construct_divisibility_sum_triples(3, 3)
    assert len(result.hypergraph.edges) == 0


def test_two_elements() -> None:
    result = construct_divisibility_sum_triples(1, 2)
    assert len(result.hypergraph.edges) == 0


def test_all_edges_are_triples() -> None:
    result = construct_divisibility_sum_triples(1, 10)
    for _, members in result.hypergraph.edges:
        assert len(members) == 3


def test_divisibility_condition() -> None:
    """Verify each edge satisfies a | (b+c) where a is the smallest member."""
    result = construct_divisibility_sum_triples(2, 20)
    for _, members in result.hypergraph.edges:
        a, b, c = sorted(int(m) for m in members)
        assert a < b < c
        assert (b + c) % a == 0


def test_result_preserves_bounds() -> None:
    result = construct_divisibility_sum_triples(5, 15)
    assert result.lower == 5
    assert result.upper == 15
