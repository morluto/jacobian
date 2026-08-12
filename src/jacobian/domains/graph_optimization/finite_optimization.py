"""Five bounded exact finite-graph optimization operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.contracts.graph_coloring import ChromaticGraph
from jacobian.contracts.graph_optimization import (
    GraphDominationMinimumOutput,
    GraphInducedBipartiteMaximumOutput,
    GraphInducedForestMaximumOutput,
    GraphInducedTreeMaximumOutput,
    GraphMinimumMaximalMatchingOutput,
    GraphOptimizationBudget,
    GraphOptimizationRequest,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.domains.graph_optimization.exact_search import (
    solve_domination,
    solve_induced_bipartite,
    solve_induced_forest,
    solve_induced_tree,
    solve_minimum_maximal_matching,
)
from jacobian.domains.graph_optimization.operations import build_simple_graph
from jacobian.operation_bindings import InstalledOperation, inline_operation
from jacobian.operations import OperationAbortError, OperationSpec


class _HasStatus(Protocol):
    status: str
    termination_reason: str


def _valid_witness(graph: Any, result: ContractModel) -> bool:
    import networkx as nx

    graph_vertices = set(graph)
    if isinstance(result, GraphDominationMinimumOutput):
        return set(result.witness_vertices) <= graph_vertices and nx.is_dominating_set(
            graph,
            result.witness_vertices,
        )
    if isinstance(result, GraphMinimumMaximalMatchingOutput):
        edges = set(result.witness_edges)
        return (
            all(
                left in graph_vertices and right in graph_vertices
                for left, right in edges
            )
            and nx.is_matching(graph, edges)
            and nx.is_maximal_matching(graph, edges)
        )
    if isinstance(result, GraphInducedForestMaximumOutput):
        if not set(result.witness_vertices) <= graph_vertices:
            return False
        induced = graph.subgraph(result.witness_vertices)
        return induced.number_of_nodes() == 0 or nx.is_forest(induced)
    if isinstance(result, GraphInducedTreeMaximumOutput):
        if not set(result.witness_vertices) <= graph_vertices:
            return False
        induced = graph.subgraph(result.witness_vertices)
        return (graph.number_of_nodes() == 0 and induced.number_of_nodes() == 0) or (
            induced.number_of_nodes() > 0
            and nx.is_connected(induced)
            and nx.is_forest(induced)
        )
    if isinstance(result, GraphInducedBipartiteMaximumOutput):
        return set(result.witness_vertices) <= graph_vertices and nx.is_bipartite(
            graph.subgraph(result.witness_vertices)
        )
    return False


_INVALID_GRAPH_OPTIMIZATION_REQUEST = CapabilityDiagnostic(
    code="INVALID_GRAPH_OPTIMIZATION_REQUEST",
    stage="graph_optimization_input_validation",
    message="Input does not satisfy the bounded finite-graph optimization contract.",
    hint=(
        "Supply a canonical finite simple graph within max_order and explicit "
        "wall-clock and solver-call budgets."
    ),
)


def _execute[ResultT: ContractModel](
    request: GraphOptimizationRequest,
    solve: Callable[
        [Any, ChromaticGraph, GraphOptimizationBudget],
        ResultT,
    ],
) -> ResultT:
    graph = cast(Any, build_simple_graph(request.graph))
    result = solve(graph, request.graph, request.resource_budget)
    state = cast(_HasStatus, result)
    if not _valid_witness(graph, result):
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            CapabilityDiagnostic(
                code="GRAPH_OPTIMIZATION_WITNESS_INVALID",
                stage="graph_optimization_postcondition",
                message=(
                    "The solver returned an incumbent that does not satisfy "
                    "the declared graph predicate."
                ),
            ),
        )
    if state.status == "EXACT":
        return result
    if state.termination_reason in {"WALL_TIME", "SOLVER_UNKNOWN"}:
        raise OperationAbortError(
            ExecutionStatus.TIMEOUT,
            CapabilityDiagnostic(
                code="GRAPH_OPTIMIZATION_TIMEOUT",
                stage="graph_optimization_search",
                message=(
                    "The graph optimization search exhausted its wall-clock "
                    "budget before establishing optimality."
                ),
            ),
        )
    return result


def _operation[ResultT: ContractModel](
    operation_id: str,
    title: str,
    description: str,
    result_type: type[ResultT],
    solve: Callable[[Any, ChromaticGraph, GraphOptimizationBudget], ResultT],
    *tags: str,
) -> InstalledOperation[GraphOptimizationRequest, ResultT]:
    return inline_operation(
        OperationSpec(
            operation_id=operation_id,
            version="1",
            title=title,
            description=description,
            request_type=GraphOptimizationRequest,
            result_type=result_type,
            execute=lambda request: _execute(request, solve),
            tags=("graph", *tags, "bounded", "z3"),
            invalid_request=_INVALID_GRAPH_OPTIMIZATION_REQUEST,
        )
    )


DOMINATION_MINIMUM_CAPABILITY = _operation(
    "graph.domination.minimum.compute",
    "Minimum dominating set",
    "Compute the domination number and an attaining set within explicit budgets.",
    GraphDominationMinimumOutput,
    lambda graph, contract, budget: solve_domination(graph, contract, budget),
    "domination",
    "minimum",
)

MINIMUM_MAXIMAL_MATCHING_CAPABILITY = _operation(
    "graph.matching.maximal.minimum.compute",
    "Minimum maximal matching",
    "Compute the saturation number and an attaining maximal matching within explicit budgets.",
    GraphMinimumMaximalMatchingOutput,
    lambda graph, contract, budget: solve_minimum_maximal_matching(
        graph, contract, budget
    ),
    "matching",
    "saturation_number",
    "minimum",
)

INDUCED_FOREST_MAXIMUM_CAPABILITY = _operation(
    "graph.induced_forest.maximum.compute",
    "Maximum induced forest",
    "Compute a maximum-order induced forest and vertex witness within explicit budgets.",
    GraphInducedForestMaximumOutput,
    lambda graph, contract, budget: solve_induced_forest(graph, contract, budget),
    "induced_forest",
    "maximum",
)

INDUCED_TREE_MAXIMUM_CAPABILITY = _operation(
    "graph.induced_tree.maximum.compute",
    "Maximum induced tree",
    "Compute a maximum-order induced tree and vertex witness within explicit budgets.",
    GraphInducedTreeMaximumOutput,
    lambda graph, contract, budget: solve_induced_tree(graph, contract, budget),
    "induced_tree",
    "maximum",
)

INDUCED_BIPARTITE_MAXIMUM_CAPABILITY = _operation(
    "graph.induced_bipartite.maximum.compute",
    "Maximum induced bipartite subgraph",
    "Compute a maximum-order induced bipartite subgraph within explicit budgets.",
    GraphInducedBipartiteMaximumOutput,
    lambda graph, contract, budget: solve_induced_bipartite(graph, contract, budget),
    "induced_bipartite",
    "maximum",
)

FINITE_GRAPH_OPTIMIZATION_CAPABILITIES = (
    DOMINATION_MINIMUM_CAPABILITY,
    MINIMUM_MAXIMAL_MATCHING_CAPABILITY,
    INDUCED_FOREST_MAXIMUM_CAPABILITY,
    INDUCED_TREE_MAXIMUM_CAPABILITY,
    INDUCED_BIPARTITE_MAXIMUM_CAPABILITY,
)
