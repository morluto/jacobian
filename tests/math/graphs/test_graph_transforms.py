from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.transforms._models import (
    GraphEdge,
    GraphTransformRequest,
    ResultGraphEdge,
    SimpleGraph,
    SubgraphRequest,
)
from jacobian.math.graphs.transforms._operations import (
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


def _result_edges(result) -> frozenset[tuple[int, int]]:
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
    result = compute_graph_power(GraphTransformRequest(graph=g))
    assert result.vertex_count == 3
    assert _result_edges(result) == {(0, 1), (1, 2), (0, 2)}


def test_graph_power_complete_graph() -> None:
    """Square of complete graph is itself."""
    g = _graph(3, [(0, 1), (1, 2), (0, 2)])
    result = compute_graph_power(GraphTransformRequest(graph=g))
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


def test_complement_of_edgeless_graph_within_output_bounds() -> None:
    """Complement of a 64-vertex edgeless graph produces 2016 edges."""
    g = _graph(64, [])
    result = compute_complement(GraphTransformRequest(graph=g))
    assert result.vertex_count == 64
    assert len(result.edges) == 2016  # C(64, 2)


def test_line_graph_of_edgeless_graph_is_empty() -> None:
    """Line graph of an edgeless graph has zero vertices (empty result allowed)."""
    g = _graph(3, [])
    result = compute_line_graph(GraphTransformRequest(graph=g))
    assert result.vertex_count == 0
    assert len(result.edges) == 0


def test_induced_subgraph_empty_vertex_set() -> None:
    """Induced subgraph on empty vertex set has zero vertices."""
    g = _graph(3, [(0, 1), (1, 2)])
    result = compute_induced_subgraph(SubgraphRequest(graph=g, vertices=()))
    assert result.vertex_count == 0
    assert len(result.edges) == 0


def test_contract_rejects_duplicate_subgraph_vertices() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SubgraphRequest(
            graph=_graph(3, [(0, 1)]),
            vertices=(0, 0),
        )


def test_line_graph_reindexes_endpoints_above_input_vertex_bound() -> None:
    """A line graph of more than 64 input edges reindexes endpoints above 63."""
    edges = [(0, i) for i in range(1, 64)] + [(1, 2), (1, 3)]
    g = _graph(64, edges)
    result = compute_line_graph(GraphTransformRequest(graph=g))
    assert result.vertex_count == 65
    endpoints = {
        endpoint for edge in result.edges for endpoint in (edge.source, edge.target)
    }
    assert max(endpoints) >= 64


def test_input_edge_rejects_endpoint_at_or_above_vertex_bound() -> None:
    with pytest.raises(ValidationError):
        GraphEdge(source=0, target=64)


def test_result_edge_allows_endpoint_up_to_input_edge_bound() -> None:
    edge = ResultGraphEdge(source=1023, target=1022)
    assert edge.source == 1023
    assert edge.target == 1022


def test_result_edge_rejects_endpoint_above_input_edge_bound() -> None:
    with pytest.raises(ValidationError):
        ResultGraphEdge(source=1024, target=0)
