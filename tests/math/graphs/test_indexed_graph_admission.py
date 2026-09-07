"""Boundary separation between indexed and label-based graph carriers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.coloring._models import _require_indexed_coloring_graph
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)


def test_indexed_graph_accepts_line_graph_vertex_bound() -> None:
    graph = IndexedSimpleUndirectedGraph(vertex_count=1024, edges=())

    assert graph.vertex_count == 1024


def test_indexed_graph_rejects_vertices_above_line_graph_bound() -> None:
    with pytest.raises(ValidationError):
        IndexedSimpleUndirectedGraph(vertex_count=1025, edges=())


def test_simple_graph_retains_256_vertex_bound() -> None:
    SimpleUndirectedGraph(vertices=tuple(str(i) for i in range(256)), edges=())

    with pytest.raises(ValidationError):
        SimpleUndirectedGraph(vertices=tuple(str(i) for i in range(257)), edges=())


def test_coloring_admission_retains_256_vertex_bound() -> None:
    graph = IndexedSimpleUndirectedGraph(vertex_count=257, edges=())

    with pytest.raises(ValueError, match="at most 256 vertices"):
        _require_indexed_coloring_graph(graph)
