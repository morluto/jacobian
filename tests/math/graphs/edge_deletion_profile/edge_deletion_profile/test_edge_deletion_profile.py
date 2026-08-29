from __future__ import annotations

from jacobian.math.graphs.edge_deletion_profile.operations import (
    compute_edge_deletion_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(tuple(e) for e in edges),
    )


def test_triangle_d0() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_edge_deletion_profile(graph, 0)
    assert result.source_chromatic_number == 3
    assert len(result.entries) == 1
    assert result.entries[0].chromatic_number == 3


def test_triangle_d1() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_edge_deletion_profile(graph, 1)
    assert len(result.entries) == 4
    for entry in result.entries[1:]:
        assert entry.chromatic_number == 2


def test_edgeless_graph() -> None:
    graph = _graph(["a", "b", "c"], [])
    result = compute_edge_deletion_profile(graph, 0)
    assert result.source_chromatic_number == 1
    assert len(result.entries) == 1


def test_path_d0() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_edge_deletion_profile(graph, 0)
    assert result.source_chromatic_number == 2


def test_path_d1() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_edge_deletion_profile(graph, 1)
    assert len(result.entries) == 3
    # After deleting any edge from a path of 3, we get an edge + isolated vertex
    for entry in result.entries[1:]:
        assert entry.chromatic_number == 2


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_edge_deletion_profile(graph, 0)
    assert result.graph == graph
