"""Tests for 3-term progression hypergraph construction."""

from jacobian.math.combinatorics._progression_hypergraph_models import (
    ProgressionHypergraphRequest,
)
from jacobian.math.combinatorics._progression_hypergraph_operations import (
    construct_3term_progression_hypergraph,
)


def test_z3() -> None:
    """Z/3Z has 3 vertices. The only 3-AP is {0,1,2}."""
    result = construct_3term_progression_hypergraph(
        ProgressionHypergraphRequest(group_order=3)
    )
    assert len(result.hypergraph.vertices) == 3
    assert len(result.hypergraph.edges) == 1


def test_z5() -> None:
    """Z/5Z has 5 vertices and 10 edges (C(5,3) = 10, all triples are 3-APs)."""
    result = construct_3term_progression_hypergraph(
        ProgressionHypergraphRequest(group_order=5)
    )
    assert len(result.hypergraph.vertices) == 5
    assert len(result.hypergraph.edges) == 10


def test_z7_count() -> None:
    """Z/7Z: n*(n-1)/2 = 21 edges."""
    result = construct_3term_progression_hypergraph(
        ProgressionHypergraphRequest(group_order=7)
    )
    assert len(result.hypergraph.edges) == 21


def test_all_edges_are_3_uniform() -> None:
    """Every edge should have exactly 3 distinct vertices."""
    result = construct_3term_progression_hypergraph(
        ProgressionHypergraphRequest(group_order=7)
    )
    for _, members in result.hypergraph.edges:
        assert len(members) == 3
        assert len(set(members)) == 3
