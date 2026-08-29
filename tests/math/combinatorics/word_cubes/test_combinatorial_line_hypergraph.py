from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.word_cubes._models import (
    CombinatorialLineHypergraphRequest,
)
from jacobian.math.combinatorics.word_cubes.operations import (
    construct_combinatorial_line_hypergraph,
)


def _edges(result):
    return list(result.hypergraph.edges)


def test_q2_d1() -> None:
    """[2]^1 has 2 vertices and 1 combinatorial line."""
    result = construct_combinatorial_line_hypergraph(2, 1)
    assert len(result.hypergraph.vertices) == 2
    assert len(result.hypergraph.edges) == 1
    _, members = _edges(result)[0]
    assert set(members) == {"0", "1"}


def test_q3_d1() -> None:
    """[3]^1 has 3 vertices and 1 combinatorial line."""
    result = construct_combinatorial_line_hypergraph(3, 1)
    assert len(result.hypergraph.vertices) == 3
    assert len(result.hypergraph.edges) == 1
    _, members = _edges(result)[0]
    assert set(members) == {"0", "1", "2"}


def test_q2_d2_count() -> None:
    """[2]^2 has 4 vertices and 5 combinatorial lines ((3^2-2^2=5)."""
    result = construct_combinatorial_line_hypergraph(2, 2)
    assert len(result.hypergraph.vertices) == 4
    assert len(result.hypergraph.edges) == 5


def test_q2_d2_fixture() -> None:
    """[2]^2 includes one-wildcard coordinate lines and the all-wildcard line."""
    result = construct_combinatorial_line_hypergraph(2, 2)
    edge_member_sets = {frozenset(members) for _, members in result.hypergraph.edges}
    assert frozenset({"00", "10"}) in edge_member_sets
    assert frozenset({"01", "11"}) in edge_member_sets
    assert frozenset({"00", "01"}) in edge_member_sets
    assert frozenset({"10", "11"}) in edge_member_sets
    assert frozenset({"00", "11"}) in edge_member_sets


def test_q3_d2_count() -> None:
    """[3]^2 has 9 vertices and (3+1)^2 - 3^2 = 16-9 = 7 combinatorial lines."""
    result = construct_combinatorial_line_hypergraph(3, 2)
    assert len(result.hypergraph.vertices) == 9
    assert len(result.hypergraph.edges) == 7


def test_q3_d2_fixture() -> None:
    """For q=3,d=2: 6 one-wildcard coordinate lines plus 3 fully-wildcard diagonal lines."""
    result = construct_combinatorial_line_hypergraph(3, 2)
    edges = result.hypergraph.edges
    assert len(edges) == 7
    for _, members in edges:
        assert len(members) == 3


def test_count_identity() -> None:
    """Vertex count = q^d and edge count = (q+1)^d - q^d."""
    for q, d in [(2, 1), (2, 2), (3, 1), (3, 2), (2, 3), (4, 1)]:
        result = construct_combinatorial_line_hypergraph(q, d)
        assert len(result.hypergraph.vertices) == q**d
        expected_edges = (q + 1) ** d - q**d
        assert len(result.hypergraph.edges) == expected_edges, (
            f"q={q}, d={d}: got {len(result.hypergraph.edges)}, expected {expected_edges}"
        )


def test_vertex_labels_are_words() -> None:
    """Vertex labels are canonical word representations."""
    result = construct_combinatorial_line_hypergraph(2, 2)
    assert set(result.hypergraph.vertices) == {"00", "01", "10", "11"}


def test_edge_size_equals_q() -> None:
    """Each edge has exactly q vertices."""
    result = construct_combinatorial_line_hypergraph(3, 2)
    for _, members in result.hypergraph.edges:
        assert len(members) == 3


def test_q4_example() -> None:
    """q=4,d=1: 4 vertices, 1 edge."""
    result = construct_combinatorial_line_hypergraph(4, 1)
    assert len(result.hypergraph.vertices) == 4
    assert len(result.hypergraph.edges) == 1


def test_rejects_q_too_small() -> None:
    with pytest.raises(ValidationError):
        CombinatorialLineHypergraphRequest(alphabet_size=1, dimension=2)


def test_rejects_vertex_count_exceeds_bound() -> None:
    with pytest.raises(ValidationError):
        CombinatorialLineHypergraphRequest(alphabet_size=10, dimension=5)


def test_independent_wildcard_enumeration() -> None:
    """Cross-check by independently enumerating all wildcard patterns."""
    q, d = 3, 2
    result = construct_combinatorial_line_hypergraph(q, d)
    actual_edges = {frozenset(members) for _, members in result.hypergraph.edges}

    expected_edges = set()
    for mask in range(1, 1 << d):
        wildcard_positions = [i for i in range(d) if mask & (1 << i)]
        fixed_positions = [i for i in range(d) if not (mask & (1 << i))]
        for fixed_values in product(range(q), repeat=len(fixed_positions)):
            edge = []
            for wildcard_val in range(q):
                word = [0] * d
                for i, pos in enumerate(fixed_positions):
                    word[pos] = fixed_values[i]
                for pos in wildcard_positions:
                    word[pos] = wildcard_val
                edge.append("".join(str(x) for x in word))
            expected_edges.add(frozenset(edge))

    assert actual_edges == expected_edges
