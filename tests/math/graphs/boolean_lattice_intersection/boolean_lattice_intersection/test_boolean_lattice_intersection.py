from __future__ import annotations

from jacobian.math.graphs.boolean_lattice_intersection.operations import (
    construct_boolean_lattice_intersection_graph,
)


def test_n0() -> None:
    result = construct_boolean_lattice_intersection_graph(0, 0, "INTERSECTION_EQ_THRESHOLD")
    assert len(result.graph.vertices) == 1  # Only the empty set


def test_n1() -> None:
    result = construct_boolean_lattice_intersection_graph(1, 0, "INTERSECTION_EQ_THRESHOLD")
    assert len(result.graph.vertices) == 2  # {}, {0}


def test_n2() -> None:
    result = construct_boolean_lattice_intersection_graph(2, 1, "INTERSECTION_EQ_THRESHOLD")
    assert len(result.graph.vertices) == 4  # {}, {0}, {1}, {0,1}


def test_result_preserves_params() -> None:
    result = construct_boolean_lattice_intersection_graph(3, 2, "INTERSECTION_EQ_THRESHOLD")
    assert result.n == 3
    assert result.intersection_cardinality == 2
