from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.monochromatic_path_hypergraph._models import (
    MonochromaticPathRequest,
)
from jacobian.math.graphs.monochromatic_path_hypergraph.operations import (
    construct_monochromatic_path_hypergraphs,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def test_two_color_path() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )
    colored = ColoredUndirectedGraph(
        graph=graph,
        edge_colors=("red", "blue"),
    )
    result = construct_monochromatic_path_hypergraphs(colored)
    assert len(result.per_color) == 2  # red and blue


def test_single_color() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )
    colored = ColoredUndirectedGraph(
        graph=graph,
        edge_colors=("red", "red"),
    )
    result = construct_monochromatic_path_hypergraphs(colored)
    assert len(result.per_color) == 1
    red_hg = result.per_color[0].hypergraph
    # Should have 3 singletons + path support {a,b,c}
    assert len(red_hg.edges) >= 4


def test_result_preserves_source() -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    colored = ColoredUndirectedGraph(graph=graph, edge_colors=("red",))
    result = construct_monochromatic_path_hypergraphs(colored)
    assert result.graph == colored


def test_complete_graph_support_family_is_rejected_before_enumeration() -> None:
    vertices = tuple(f"v{i:02d}" for i in range(14))
    edges = tuple(
        (left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    colored = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=tuple("red" for _ in edges),
    )

    with pytest.raises(OperationDomainValidationError, match="result bounds"):
        construct_monochromatic_path_hypergraphs(colored)
    with pytest.raises(ValidationError, match="result bounds"):
        MonochromaticPathRequest(graph=colored)
