"""Graph transform operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.transforms import (
    complement,
    graph_power,
    induced_subgraph,
    line_graph,
    path_profile,
)
from jacobian.math.graphs.transforms._models import (
    GraphResult,
    GraphTransformRequest,
    ResultGraphEdge,
    SubgraphRequest,
)
from jacobian.math.graphs.transforms._path_profile_models import (
    PathProfileRequest,
    PathProfileResult,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

SQUARE_POWER = 2


def _edges(graph: IndexedSimpleUndirectedGraph) -> list[tuple[int, int]]:
    return list(graph.edges)


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


def compute_path_profile(request: PathProfileRequest) -> PathProfileResult:
    return path_profile(request.graph, request.path_length)


_GRAPH_EXAMPLE = {
    "graph": {
        "vertex_count": 3,
        "edges": [
            [0, 1],
            [1, 2],
        ],
    }
}


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.complement.compute",
        title="Compute the complement of a graph",
        description="Compute the exact complement of a simple undirected graph. The complement has the same vertex set, with edges exactly where the original has no edge.",
        request_type=GraphTransformRequest,
        result_type=GraphResult,
        run=compute_complement,
        tags=("graph", "complement", "exact"),
        examples=(
            OperationExample(
                name="path_p2",
                description="Complement of a path graph P2.",
                input=_GRAPH_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.line_graph.compute",
        title="Compute the line graph of a graph",
        description="Compute the exact line graph L(G), whose vertices are edges of G and whose edges join pairs of incident edges.",
        request_type=GraphTransformRequest,
        result_type=GraphResult,
        run=compute_line_graph,
        tags=("graph", "line-graph", "exact"),
        examples=(
            OperationExample(
                name="path_p2",
                description="Line graph of a path graph P2.",
                input=_GRAPH_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.power.compute",
        title="Compute the graph square",
        description="Compute the exact square G^2 of a simple undirected graph, joining vertices at distance at most two.",
        request_type=GraphTransformRequest,
        result_type=GraphResult,
        run=compute_graph_power,
        tags=("graph", "graph-power", "exact"),
        examples=(
            OperationExample(
                name="path_p2",
                description="Square of a path graph P2.",
                input=_GRAPH_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.induced_subgraph.compute",
        title="Extract an induced subgraph on a vertex subset",
        description="Compute the exact induced subgraph G[V'] and reindex its selected vertices from zero.",
        request_type=SubgraphRequest,
        result_type=GraphResult,
        run=compute_induced_subgraph,
        tags=("graph", "induced-subgraph", "exact"),
        examples=(
            OperationExample(
                name="path_p2_vertices_0_2",
                description="Induced subgraph of P2 on vertices {0, 2}.",
                input={"graph": _GRAPH_EXAMPLE["graph"], "vertices": [0, 2]},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.path_profile.compute",
        title="Profile fixed-length simple paths by endpoint",
        description="For each ordered pair of vertices, count simple paths of the given length; the request is bounded by a degree-sensitive search budget.",
        request_type=PathProfileRequest,
        result_type=PathProfileResult,
        run=compute_path_profile,
        tags=("graph", "path", "profile"),
        examples=(
            OperationExample(
                name="path_profile_p3_len1",
                description="Count length-1 paths in a path graph P3; path_length must be at most 10.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    },
                    "path_length": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
