"""Domain adapter for graph transform operations."""

from __future__ import annotations

from jacobian.contracts.graph_transforms import (
    GraphEdge,
    GraphResult,
    GraphTransformRequest,
    SimpleGraph,
    SubgraphRequest,
)
from jacobian.math.graph_transforms import (
    complement,
    graph_power,
    induced_subgraph,
    line_graph,
)


def _edges(graph: SimpleGraph) -> list[tuple[int, int]]:
    return [(e.source, e.target) for e in graph.edges]


def _result(vc: int, edges: list[tuple[int, int]]) -> GraphResult:
    return GraphResult(
        vertex_count=vc,
        edges=tuple(GraphEdge(source=s, target=t) for s, t in edges),
    )


def compute_complement(request: GraphTransformRequest) -> GraphResult:
    g = request.graph
    vc, edges = complement(g.vertex_count, _edges(g))
    return _result(vc, edges)


def compute_line_graph(request: GraphTransformRequest) -> GraphResult:
    g = request.graph
    vc, edges = line_graph(g.vertex_count, _edges(g))
    return _result(vc, edges)


def compute_graph_power(request: GraphTransformRequest, power: int) -> GraphResult:
    g = request.graph
    vc, edges = graph_power(g.vertex_count, _edges(g), power)
    return _result(vc, edges)


def compute_induced_subgraph(request: SubgraphRequest) -> GraphResult:
    g = request.graph
    vc, edges = induced_subgraph(g.vertex_count, _edges(g), list(request.vertices))
    return _result(vc, edges)
