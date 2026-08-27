"""Exact directed graph operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.directed._models import (
    AcyclicOrderRequest,
    AcyclicOrderResult,
    CondensationRequest,
    CondensationResult,
    DagLongestPathRequest,
    DagLongestPathResult,
    ReachabilityRequest,
    ReachabilityResult,
    StronglyConnectedComponentsRequest,
    StronglyConnectedComponentsResult,
)
from jacobian.math.graphs.directed._operations import (
    compute_acyclic_order,
    compute_condensation,
    compute_dag_longest_path,
    compute_reachability,
    compute_strongly_connected_components,
)


def directed_graph_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
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


DIRECTED_GRAPH_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    directed_graph_operation(
        "graph.directed.reachability.compute",
        "Compute reachable vertices from a source in a directed graph",
        "Determine which vertices are reachable from a given source vertex in a "
        "simple directed graph using NetworkX. Returns the reachable and "
        "unreachable vertex sets.",
        ReachabilityRequest,
        ReachabilityResult,
        compute_reachability,
        "graph",
        "directed",
        "reachability",
        "exact",
        examples=(
            example(
                "simple_reachability",
                "Compute reachability from vertex 0 in a small graph.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                    "source": 0,
                },
            ),
        ),
    ),
    directed_graph_operation(
        "graph.directed.scc.compute",
        "Compute strongly connected components of a directed graph",
        "Partition a simple directed graph into strongly connected components "
        "using NetworkX. Returns the number of components and the vertex list of "
        "each component.",
        StronglyConnectedComponentsRequest,
        StronglyConnectedComponentsResult,
        compute_strongly_connected_components,
        "graph",
        "directed",
        "scc",
        "exact",
        examples=(
            example(
                "simple_cycle_scc",
                "Compute SCCs of a graph containing a simple cycle.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 0], [2, 3]],
                    },
                },
            ),
        ),
    ),
    directed_graph_operation(
        "graph.directed.condensation.compute",
        "Compute the condensation of a directed graph",
        "Compute the condensation DAG of a simple directed graph using NetworkX. "
        "The condensation's vertices are the strongly connected components of the "
        "original graph.",
        CondensationRequest,
        CondensationResult,
        compute_condensation,
        "graph",
        "directed",
        "condensation",
        "exact",
        examples=(
            example(
                "simple_condensation",
                "Compute the condensation of a graph with one cycle.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 0], [2, 3]],
                    },
                },
            ),
        ),
    ),
    directed_graph_operation(
        "graph.directed.acyclic_order.compute",
        "Compute a topological order of a directed acyclic graph",
        "Compute a topological ordering of a simple directed graph using "
        "NetworkX. Reports acyclic=false and an empty order when the graph "
        "contains a cycle.",
        AcyclicOrderRequest,
        AcyclicOrderResult,
        compute_acyclic_order,
        "graph",
        "directed",
        "topological-sort",
        "exact",
        examples=(
            example(
                "simple_dag_topological_order",
                "Compute a topological order of a small DAG.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [0, 2], [1, 3], [2, 3]],
                    },
                },
            ),
        ),
    ),
    directed_graph_operation(
        "graph.directed.dag_longest_path.compute",
        "Compute the longest directed path in a DAG",
        "Compute the exact maximum directed simple-path length (number of "
        "edges) and a canonical path witness in a simple directed acyclic "
        "graph. Reports NOT_APPLICABLE when the graph contains a cycle.",
        DagLongestPathRequest,
        DagLongestPathResult,
        compute_dag_longest_path,
        "graph",
        "directed",
        "dag",
        "longest-path",
        "exact",
        examples=(
            example(
                "simple_dag_longest_path",
                "Compute the longest path in a small diamond DAG.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [0, 2], [1, 3], [2, 3]],
                    },
                },
            ),
        ),
    ),
)

TOOLS = DIRECTED_GRAPH_OPERATIONS

__all__ = ["TOOLS"]
