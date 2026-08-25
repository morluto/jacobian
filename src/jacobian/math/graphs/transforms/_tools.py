"""Graph transform operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.transforms._models import (
    GraphResult,
    GraphTransformRequest,
    SubgraphRequest,
)
from jacobian.math.graphs.transforms._operations import (
    compute_complement,
    compute_graph_power,
    compute_induced_subgraph,
    compute_line_graph,
)


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
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ],
    }
}


TOOLS: MathTools = (
    gt_operation(
        "graph.complement.compute",
        "Compute the complement of a graph",
        "Compute the exact complement of a simple undirected graph using "
        "NetworkX. The complement has the same vertex set, with edges "
        "exactly where the original has no edge.",
        GraphTransformRequest,
        GraphResult,
        compute_complement,
        "graph",
        "complement",
        "exact",
        examples=(
            example(
                "path_p2",
                "Complement of a path graph P2.",
                _GRAPH_EXAMPLE,
            ),
        ),
    ),
    gt_operation(
        "graph.line_graph.compute",
        "Compute the line graph of a graph",
        "Compute the exact line graph L(G) where vertices are edges of G "
        "and two vertices in L(G) are adjacent if they share an endpoint in G.",
        GraphTransformRequest,
        GraphResult,
        compute_line_graph,
        "graph",
        "line-graph",
        "exact",
        examples=(
            example(
                "path_p2",
                "Line graph of a path graph P2.",
                _GRAPH_EXAMPLE,
            ),
        ),
    ),
    gt_operation(
        "graph.power.compute",
        "Compute the graph power (square) of a graph",
        "Compute the exact square G^2 where two vertices are adjacent if "
        "their distance in G is at most 2.",
        GraphTransformRequest,
        GraphResult,
        compute_graph_power,
        "graph",
        "graph-power",
        "exact",
        examples=(
            example(
                "path_p2",
                "Square of a path graph P2.",
                _GRAPH_EXAMPLE,
            ),
        ),
    ),
    gt_operation(
        "graph.induced_subgraph.compute",
        "Extract an induced subgraph on a vertex subset",
        "Compute the exact induced subgraph G[V'] where V' is a subset of "
        "the vertices. Vertices are reindexed 0..|V'|-1.",
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
                {
                    "graph": _GRAPH_EXAMPLE["graph"],
                    "vertices": [0, 2],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
