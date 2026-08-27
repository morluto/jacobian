"""Tests for divisibility-incidence graph construction."""

from jacobian.math.number_theory._divisibility_graph_models import DivisibilityIncidenceGraphRequest
from jacobian.math.number_theory._divisibility_graph_operations import compute_divisibility_incidence_graph


def test_basic() -> None:
    result = compute_divisibility_incidence_graph(
        DivisibilityIncidenceGraphRequest(left_family=["2", "3"], right_family=["6", "12", "5"])
    )
    edges = set(tuple(e) for e in result.graph.edges)
    assert ("L0", "R0") in edges
    assert ("L0", "R1") in edges
    assert ("L1", "R0") in edges
    assert ("L1", "R1") in edges
    assert len(result.graph.edges) == 4


def test_no_edges() -> None:
    result = compute_divisibility_incidence_graph(
        DivisibilityIncidenceGraphRequest(left_family=["7"], right_family=["3"])
    )
    assert len(result.graph.edges) == 0


def test_bipartite() -> None:
    result = compute_divisibility_incidence_graph(
        DivisibilityIncidenceGraphRequest(left_family=["1", "2", "3"], right_family=["2", "3", "6"])
    )
    for e in result.graph.edges:
        assert e[0].startswith("L") and e[1].startswith("R")
