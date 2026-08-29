from __future__ import annotations

from jacobian.math.graphs.coloring.equitable_k_coloring.operations import (
    decide_equitable_k_coloring,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(tuple(e) for e in edges),
    )


def test_k4_equitable() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
    )
    result = decide_equitable_k_coloring(graph, 4)
    assert result.colorable


def test_path_equitable() -> None:
    graph = _graph(["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]])
    result = decide_equitable_k_coloring(graph, 2)
    assert result.colorable


def test_k3_not_2_colorable() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = decide_equitable_k_coloring(graph, 2)
    assert not result.colorable


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = decide_equitable_k_coloring(graph, 2)
    assert result.graph == graph
    assert result.k == 2
