from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.transforms._models import (
    GraphTransformRequest,
    LineGraphRequest,
    SubgraphRequest,
)
from jacobian.math.graphs.transforms._tools import (
    compute_complement,
    compute_graph_power,
    compute_induced_subgraph,
    compute_line_graph,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


def _graph(vc: int, edges: list[tuple[int, int]]) -> IndexedSimpleUndirectedGraph:
    return IndexedSimpleUndirectedGraph(
        vertex_count=vc,
        edges=tuple(edges),
    )


def _result_edges(result: IndexedSimpleUndirectedGraph) -> frozenset[tuple[int, int]]:
    return frozenset(result.edges)


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
    result = compute_line_graph(LineGraphRequest(graph=g))
    assert result.vertex_count == 2  # two edges in original
    assert len(result.edges) == 1  # they share a vertex


def test_line_graph_of_triangle_is_triangle() -> None:
    """Line graph of K3 (triangle) is K3."""
    g = _graph(3, [(0, 1), (1, 2), (0, 2)])
    result = compute_line_graph(LineGraphRequest(graph=g))
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


def test_complement_of_edgeless_graph_within_output_bounds() -> None:
    """Complement of a 64-vertex edgeless graph produces 2016 edges."""
    g = _graph(64, [])
    result = compute_complement(GraphTransformRequest(graph=g))
    assert result.vertex_count == 64
    assert len(result.edges) == 2016  # C(64, 2)


def test_line_graph_of_edgeless_graph_is_empty() -> None:
    """Line graph of an edgeless graph has zero vertices (empty result allowed)."""
    g = _graph(3, [])
    result = compute_line_graph(LineGraphRequest(graph=g))
    assert result.vertex_count == 0
    assert len(result.edges) == 0


def test_induced_subgraph_empty_vertex_set() -> None:
    """Induced subgraph on empty vertex set has zero vertices."""
    g = _graph(3, [(0, 1), (1, 2)])
    result = compute_induced_subgraph(SubgraphRequest(graph=g, vertices=()))
    assert result.vertex_count == 0
    assert len(result.edges) == 0


def test_contract_rejects_duplicate_subgraph_vertices() -> None:
    with pytest.raises(ValidationError):
        SubgraphRequest(
            graph=_graph(3, [(0, 1)]),
            vertices=(0, 0),
        )


def test_line_graph_reindexes_endpoints_above_input_vertex_bound() -> None:
    """A line graph of more than 64 input edges reindexes endpoints above 63."""
    edges = [(0, i) for i in range(1, 64)] + [(1, 2), (1, 3)]
    g = _graph(64, edges)
    result = compute_line_graph(LineGraphRequest(graph=g))
    assert result.vertex_count == 65
    endpoints = {endpoint for edge in result.edges for endpoint in edge}
    assert max(endpoints) >= 64


def test_input_edge_rejects_endpoint_at_or_above_vertex_bound() -> None:
    with pytest.raises(ValidationError):
        _graph(64, [(0, 64)])


def test_line_graph_retains_maximum_input_edge_boundary() -> None:
    edges = [(left, right) for left in range(32) for right in range(32, 64)]
    result = compute_line_graph(LineGraphRequest(graph=_graph(64, edges)))
    assert type(result) is IndexedSimpleUndirectedGraph
    assert result.vertex_count == 1024
    assert len(result.edges) == 32 * 32 * 62 // 2
    assert (
        IndexedSimpleUndirectedGraph.model_validate_json(result.model_dump_json())
        == result
    )


def test_line_graph_preserves_its_expansion_admission() -> None:
    edges = [(left, right) for left in range(32) for right in range(32, 64)]
    edges.append((0, 1))
    with pytest.raises(ValidationError, match="1024 input edges"):
        LineGraphRequest(graph=_graph(64, edges))


def test_reversed_induced_selection_has_canonical_edges() -> None:
    result = compute_induced_subgraph(
        SubgraphRequest(graph=_graph(2, [(0, 1)]), vertices=(1, 0))
    )
    assert result.edges == ((0, 1),)
    assert (
        IndexedSimpleUndirectedGraph.model_validate_json(result.model_dump_json())
        == result
    )


def test_dense_complement_round_trip_needs_no_representation_adapter() -> None:
    original = _graph(64, [])
    complete = compute_complement(GraphTransformRequest(graph=original))
    request = GraphTransformRequest.model_validate_json(
        '{"graph":' + complete.model_dump_json() + "}"
    )
    assert compute_complement(request) == original


def test_empty_transform_results_compose_unchanged() -> None:
    empty = _graph(0, [])
    assert compute_complement(GraphTransformRequest(graph=empty)) == empty
    assert compute_graph_power(GraphTransformRequest(graph=empty)) == empty
    assert compute_line_graph(LineGraphRequest(graph=empty)) == empty
    assert compute_induced_subgraph(SubgraphRequest(graph=empty, vertices=())) == empty
