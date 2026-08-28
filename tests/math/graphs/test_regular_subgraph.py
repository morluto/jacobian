from __future__ import annotations

from jacobian.math.graphs.regular_subgraph.operations import (
    find_k_regular_subgraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: list[str], edges: list[tuple[str, str]]) -> SimpleUndirectedGraph:
    canonical_edges = tuple(
        (left, right) if left <= right else (right, left) for left, right in edges
    )
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=canonical_edges,
    )


def test_c4_is_2_regular() -> None:
    """C4 contains a 2-regular subgraph (itself)."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    result = find_k_regular_subgraph(g, 2)
    assert result.found
    assert len(result.vertices) == 4
    # Every used vertex must have degree 2.
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 2 for d in degree.values())


def test_p3_no_2_regular_subgraph() -> None:
    """P3 has no nonempty 2-regular subgraph."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = find_k_regular_subgraph(g, 2)
    assert not result.found
    assert result.vertices == ()
    assert result.edges == ()


def test_k0_returns_single_vertex() -> None:
    """k=0 returns a single vertex with no edges."""
    g = _graph(["a", "b"], [("a", "b")])
    result = find_k_regular_subgraph(g, 0)
    assert result.found
    assert len(result.vertices) == 1
    assert len(result.edges) == 0


def test_k1_matching() -> None:
    """k=1 subgraph is a matching: any edge."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = find_k_regular_subgraph(g, 1)
    assert result.found
    assert len(result.edges) == 1
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 1 for d in degree.values())


def test_triangle_is_2_regular() -> None:
    """K3 is 2-regular."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
    result = find_k_regular_subgraph(g, 2)
    assert result.found
    assert len(result.vertices) == 3
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 2 for d in degree.values())


def test_k3_is_3_regular_in_k4() -> None:
    """K4 contains a K4 (3-regular subgraph)."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")],
    )
    result = find_k_regular_subgraph(g, 3)
    assert result.found
    assert len(result.vertices) == 4
    degree: dict[str, int] = {}
    for u, v in result.edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert all(d == 3 for d in degree.values())
