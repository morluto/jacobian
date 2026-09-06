from __future__ import annotations

from collections.abc import Sequence

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring.equitable_k_coloring.operations import (
    decide_equitable_k_coloring,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
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


def test_nonpositive_palette_is_rejected_before_division() -> None:
    with pytest.raises(OperationDomainValidationError, match="positive palette"):
        decide_equitable_k_coloring(_graph(["a"], []), 0)


def test_large_palette_uses_the_direct_singleton_class_construction() -> None:
    graph = _graph([str(index) for index in range(64)], [])
    result = decide_equitable_k_coloring(graph, 64)

    assert result.colorable
    assert result.coloring is not None
    assert result.coloring.coloring == tuple(range(64))


def test_edgeless_graph_uses_direct_balanced_class_construction() -> None:
    graph = _graph([str(index) for index in range(20)], [])
    result = decide_equitable_k_coloring(graph, 2)

    assert result.colorable
    assert result.coloring is not None
    assert result.coloring.coloring.count(0) == 10
    assert result.coloring.coloring.count(1) == 10


def test_exponential_search_is_rejected_before_backtracking() -> None:
    graph = _graph([str(index) for index in range(20)], [["0", "1"]])

    with pytest.raises(OperationDomainValidationError, match="search bound"):
        decide_equitable_k_coloring(graph, 2)
