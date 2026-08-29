from __future__ import annotations

from jacobian.math.graphs.neighborhood.open_neighborhood.operations import (
    compute_open_neighborhood,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(tuple(e) for e in edges),
    )


def test_empty_selected() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_open_neighborhood(graph, ())
    assert result.neighborhood == ()


def test_edgeless_graph() -> None:
    graph = _graph(["a", "b", "c"], [])
    result = compute_open_neighborhood(graph, ("a",))
    assert result.neighborhood == ()


def test_single_vertex_neighborhood() -> None:
    graph = _graph(["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]])
    result = compute_open_neighborhood(graph, ("b",))
    assert result.neighborhood == ("a", "c")


def test_all_vertices_selected() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_open_neighborhood(graph, ("a", "b", "c"))
    assert result.neighborhood == ()


def test_overlapping_neighborhoods() -> None:
    graph = _graph(
        ["a", "b", "c", "d", "e"],
        [["a", "b"], ["a", "c"], ["b", "d"], ["c", "e"]],
    )
    result = compute_open_neighborhood(graph, ("a", "b"))
    assert result.neighborhood == ("c", "d")


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_open_neighborhood(graph, ("a",))
    assert result.graph == graph
    assert result.selected_vertices == ("a",)
    assert result.neighborhood == ("b",)


def test_sorted_output() -> None:
    graph = _graph(
        ["v0", "v1", "v2", "v3"],
        [["v0", "v2"], ["v0", "v1"], ["v0", "v3"]],
    )
    result = compute_open_neighborhood(graph, ("v0",))
    assert result.neighborhood == ("v1", "v2", "v3")
    assert list(result.neighborhood) == sorted(result.neighborhood)
