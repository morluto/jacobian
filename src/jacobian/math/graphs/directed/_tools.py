"""Exact directed graph operation declarations."""

from typing import Any

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
from jacobian.math.graphs.directed.operations import (
    acyclic_order,
    condensation,
    dag_longest_path,
    reachability,
    strongly_connected_components,
)


def _reachability(request: ReachabilityRequest) -> ReachabilityResult:
    return reachability(request.graph, request.source)


def _components(
    request: StronglyConnectedComponentsRequest,
) -> StronglyConnectedComponentsResult:
    return strongly_connected_components(request.graph)


def _condensation(request: CondensationRequest) -> CondensationResult:
    return condensation(request.graph)


def _acyclic_order(request: AcyclicOrderRequest) -> AcyclicOrderResult:
    return acyclic_order(request.graph)


def _dag_longest_path(request: DagLongestPathRequest) -> DagLongestPathResult:
    return dag_longest_path(request.graph)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.directed.reachability.compute",
        title="Compute reachable vertices from a source in a directed graph",
        description="Determine which vertices are reachable from a given source vertex in a "
        "simple directed graph using NetworkX. Returns the reachable and "
        "unreachable vertex sets.",
        request_type=ReachabilityRequest,
        result_type=ReachabilityResult,
        run=_reachability,
        tags=("graph", "directed", "reachability", "exact"),
        examples=(
            OperationExample(
                name="simple_reachability",
                description="Compute reachability from vertex 0 in a small graph.",
                input={
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                    "source": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.directed.scc.compute",
        title="Compute strongly connected components of a directed graph",
        description="Partition a simple directed graph into strongly connected components. Returns the number of components and each component's sorted vertex list.",
        request_type=StronglyConnectedComponentsRequest,
        result_type=StronglyConnectedComponentsResult,
        run=_components,
        tags=("graph", "directed", "scc", "exact"),
        examples=(
            OperationExample(
                name="simple_cycle_scc",
                description="Compute SCCs of a graph containing a simple cycle.",
                input={
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 0], [2, 3]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.directed.condensation.compute",
        title="Compute the condensation of a directed graph",
        description="Compute the condensation DAG of a simple directed graph using NetworkX. "
        "The condensation's vertices are the strongly connected components of the "
        "original graph.",
        request_type=CondensationRequest,
        result_type=CondensationResult,
        run=_condensation,
        tags=("graph", "directed", "condensation", "exact"),
        examples=(
            OperationExample(
                name="simple_condensation",
                description="Compute the condensation of a graph with one cycle.",
                input={
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 0], [2, 3]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.directed.acyclic_order.compute",
        title="Compute a topological order of a directed acyclic graph",
        description="Compute a topological ordering of a simple directed graph using "
        "NetworkX. Reports acyclic=false and an empty order when the graph "
        "contains a cycle.",
        request_type=AcyclicOrderRequest,
        result_type=AcyclicOrderResult,
        run=_acyclic_order,
        tags=("graph", "directed", "topological-sort", "exact"),
        examples=(
            OperationExample(
                name="simple_dag_topological_order",
                description="Compute a topological order of a small DAG.",
                input={
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [0, 2], [1, 3], [2, 3]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.directed.dag_longest_path.compute",
        title="Compute the longest directed path in a DAG",
        description="Compute the exact maximum directed simple-path length (number of "
        "edges) and a canonical path witness in a simple directed acyclic "
        "graph. Reports NOT_APPLICABLE when the graph contains a cycle.",
        request_type=DagLongestPathRequest,
        result_type=DagLongestPathResult,
        run=_dag_longest_path,
        tags=("graph", "directed", "dag", "longest-path", "exact"),
        examples=(
            OperationExample(
                name="simple_dag_longest_path",
                description="Compute the longest path in a small diamond DAG.",
                input={
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [0, 2], [1, 3], [2, 3]],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
