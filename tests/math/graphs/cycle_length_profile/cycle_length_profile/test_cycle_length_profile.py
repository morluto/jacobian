from __future__ import annotations

from jacobian.math.graphs.cycle_length_profile.operations import (
    compute_cycle_length_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(tuple(e) for e in edges),
    )


def test_triangle() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_cycle_length_profile(graph)
    assert result.cycle_lengths == (3,)


def test_edgeless_graph() -> None:
    graph = _graph(["a", "b", "c"], [])
    result = compute_cycle_length_profile(graph)
    assert result.cycle_lengths == ()


def test_path_graph() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_cycle_length_profile(graph)
    assert result.cycle_lengths == ()


def test_k4() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
    )
    result = compute_cycle_length_profile(graph)
    assert set(result.cycle_lengths) == {3, 4}


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_cycle_length_profile(graph)
    assert result.graph == graph
