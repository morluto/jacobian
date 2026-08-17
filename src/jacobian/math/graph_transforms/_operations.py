"""Domain-owned graph transform operations."""

from __future__ import annotations

from jacobian.math.graph_transforms import (
    complement,
    graph_power,
    induced_subgraph,
    line_graph,
)
from jacobian.math.graph_transforms._models import (
    GraphResult,
    GraphTransformRequest,
    ResultGraphEdge,
    SimpleGraph,
    SubgraphRequest,
)

SQUARE_POWER = 2


def _edges(graph: SimpleGraph) -> list[tuple[int, int]]:
    return [(edge.source, edge.target) for edge in graph.edges]


def _result(vertex_count: int, edges: list[tuple[int, int]]) -> GraphResult:
    return GraphResult(
        vertex_count=vertex_count,
        edges=tuple(
            ResultGraphEdge(source=source, target=target) for source, target in edges
        ),
    )


def compute_complement(request: GraphTransformRequest) -> GraphResult:
    graph = request.graph
    vertex_count, edges = complement(graph.vertex_count, _edges(graph))
    return _result(vertex_count, edges)


def compute_line_graph(request: GraphTransformRequest) -> GraphResult:
    graph = request.graph
    vertex_count, edges = line_graph(graph.vertex_count, _edges(graph))
    return _result(vertex_count, edges)


def compute_graph_power(request: GraphTransformRequest) -> GraphResult:
    graph = request.graph
    vertex_count, edges = graph_power(
        graph.vertex_count,
        _edges(graph),
        SQUARE_POWER,
    )
    return _result(vertex_count, edges)


def compute_induced_subgraph(request: SubgraphRequest) -> GraphResult:
    graph = request.graph
    vertex_count, edges = induced_subgraph(
        graph.vertex_count,
        _edges(graph),
        list(request.vertices),
    )
    return _result(vertex_count, edges)
