from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.graph_transforms import (
    GraphEdge,
    GraphTransformRequest,
    SimpleGraph,
    SubgraphRequest,
)
from jacobian.domains.graph_transforms.operations import (
    compute_complement,
    compute_graph_power,
    compute_induced_subgraph,
    compute_line_graph,
)


def _graph(vc: int, edges: list[tuple[int, int]]) -> SimpleGraph:
    return SimpleGraph(
        vertex_count=vc,
        edges=tuple(GraphEdge(source=s, target=t) for s, t in edges),
    )


def _result_edges(result) -> set[tuple[int, int]]:
    return frozenset(
        (e.source, e.target) if e.source < e.target else (e.target, e.source)
        for e in result.edges
    )


def test_complement_of_path_3() -> None:
    """Complement of path 0-1-2 is the single edge (0,2)."""
    g = _graph(3, [(0, 1), (1, 2)])
    result = compute_complement(GraphTransformRequest(graph=g))
    assert result.vertex_count == 3
    assert _result_edges(result) == {(0, 2)}


def test_complement_of_complete_graph_is_empty() -> None:
    """Complement of K3 (complete graph) is empty."""
    g = _graph(3, [(0, 1), (1, 2), (0, 2)])
    result = compute_complement(GraphTransformRequest(graph=g))
    assert result.vertex_count == 3
    assert len(result.edges) == 0


def test_line_graph_of_path() -> None:
    """Line graph of path 0-1-2 is a single edge between the two edges."""
    g = _graph(3, [(0, 1), (1, 2)])
    result = compute_line_graph(GraphTransformRequest(graph=g))
    assert result.vertex_count == 2  # two edges in original
    assert len(result.edges) == 1  # they share a vertex


def test_line_graph_of_triangle_is_triangle() -> None:
    """Line graph of K3 (triangle) is K3."""
    g = _graph(3, [(0, 1), (1, 2), (0, 2)])
    result = compute_line_graph(GraphTransformRequest(graph=g))
    assert result.vertex_count == 3
    assert len(result.edges) == 3


def test_graph_power_path_2() -> None:
    """Square of path 0-1-2 adds edge (0,2)."""
    g = _graph(3, [(0, 1), (1, 2)])
    result = compute_graph_power(GraphTransformRequest(graph=g), 2)
    assert result.vertex_count == 3
    assert _result_edges(result) == {(0, 1), (1, 2), (0, 2)}


def test_graph_power_complete_graph() -> None:
    """Square of complete graph is itself."""
    g = _graph(3, [(0, 1), (1, 2), (0, 2)])
    result = compute_graph_power(GraphTransformRequest(graph=g), 2)
    assert result.vertex_count == 3
    assert len(result.edges) == 3


def test_induced_subgraph_path() -> None:
    """Induced subgraph of path 0-1-2 on {0, 2} has no edges."""
    g = _graph(3, [(0, 1), (1, 2)])
    result = compute_induced_subgraph(SubgraphRequest(graph=g, vertices=(0, 2)))
    assert result.vertex_count == 2
    assert len(result.edges) == 0


def test_induced_subgraph_triangle() -> None:
    """Induced subgraph of K3 on {0, 1} has one edge."""
    g = _graph(3, [(0, 1), (1, 2), (0, 2)])
    result = compute_induced_subgraph(SubgraphRequest(graph=g, vertices=(0, 1)))
    assert result.vertex_count == 2
    assert len(result.edges) == 1


def test_contract_rejects_self_loop() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        GraphEdge(source=0, target=0)


def test_contract_rejects_duplicate_edges() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SimpleGraph(
            vertex_count=3,
            edges=(
                GraphEdge(source=0, target=1),
                GraphEdge(source=1, target=0),  # same edge reversed
            ),
        )


def test_contract_rejects_out_of_range_vertices() -> None:
    with pytest.raises(ValidationError, match="vertex_count"):
        SimpleGraph(
            vertex_count=2,
            edges=(GraphEdge(source=0, target=5),),
        )
