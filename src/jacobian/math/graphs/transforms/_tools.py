"""Graph transform operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def gt_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    gt_operation(
        "graph.complement.compute",
        "Compute the complement of a graph",
        "Compute the exact complement of a simple undirected graph. The complement has the same vertex set, with edges exactly where the original has no edge.",
        GraphTransformRequest,
        GraphResult,
        compute_complement,
        "graph",
        "complement",
        "exact",
        examples=(
            example("path_p2", "Complement of a path graph P2.", _GRAPH_EXAMPLE),
        ),
    ),
    gt_operation(
        "graph.line_graph.compute",
        "Compute the line graph of a graph",
        "Compute the exact line graph L(G), whose vertices are edges of G and whose edges join pairs of incident edges.",
        GraphTransformRequest,
        GraphResult,
        compute_line_graph,
        "graph",
        "line-graph",
        "exact",
        examples=(
            example("path_p2", "Line graph of a path graph P2.", _GRAPH_EXAMPLE),
        ),
    ),
    gt_operation(
        "graph.power.compute",
        "Compute the graph square",
        "Compute the exact square G^2 of a simple undirected graph, joining vertices at distance at most two.",
        GraphTransformRequest,
        GraphResult,
        compute_graph_power,
        "graph",
        "graph-power",
        "exact",
        examples=(example("path_p2", "Square of a path graph P2.", _GRAPH_EXAMPLE),),
    ),
    gt_operation(
        "graph.induced_subgraph.compute",
        "Extract an induced subgraph on a vertex subset",
        "Compute the exact induced subgraph G[V'] and reindex its selected vertices from zero.",
        SubgraphRequest,
        GraphResult,
        compute_induced_subgraph,
        "graph",
        "induced-subgraph",
        "exact",
        examples=(
            example(
                "path_p2_vertices_0_2",
                "Induced subgraph of P2 on vertices {0, 2}.",
                {"graph": _GRAPH_EXAMPLE["graph"], "vertices": [0, 2]},
            ),
        ),
    ),
    gt_operation(
        "graph.path_profile.compute",
        "Profile fixed-length simple paths by endpoint",
        "For each ordered pair of vertices, count simple paths of the given length; the request is bounded by a degree-sensitive search budget.",
        PathProfileRequest,
        PathProfileResult,
        compute_path_profile,
        "graph",
        "path",
        "profile",
        examples=(
            example(
                "path_profile_p3_len1",
                "Count length-1 paths in a path graph P3; path_length must be at most 10.",
                {
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
