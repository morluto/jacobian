from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.path_decomposition.operations import (
    compute_minimum_path_decomposition,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_p3_path_number_1() -> None:
    """P3 has path number 1."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 1


def test_k3_path_number_2() -> None:
    """K3 has path number 2: one length-2 path plus one remaining edge."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 2


def test_single_edge() -> None:
    """A single edge has path number 1."""
    g = _graph(["a", "b"], [("a", "b")])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 1


def test_edgeless_graph() -> None:
    """An edgeless graph has path number 0."""
    g = _graph(["a", "b"], [])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 0


def test_path_replay() -> None:
    """Every source edge appears in exactly one returned path."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")])
    result = compute_minimum_path_decomposition(g)
    all_edges = set()
    for path in result.paths:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            edge = (min(a, b), max(a, b))
            assert edge not in all_edges
            all_edges.add(edge)
    assert all_edges == set(g.edges)


def test_result_preserves_source() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    result = compute_minimum_path_decomposition(g)
    assert result.graph == g


def test_k5_search_is_rejected_before_exhaustive_cover() -> None:
    """The admitted graph shape must fit the exact cover work envelope."""
    vertices = ["a", "b", "c", "d", "e"]
    g = _graph(
        vertices,
        [(vertices[i], vertices[j]) for i in range(5) for j in range(i + 1, 5)],
    )
    with pytest.raises(OperationDomainValidationError, match="work envelope"):
        compute_minimum_path_decomposition(g)
