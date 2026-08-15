"""Five bounded exact finite-graph optimization operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from jacobian.contracts.base import ContractModel
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
from jacobian.domains.graph_optimization.exact_search import (
    solve_domination,
    solve_induced_bipartite,
    solve_induced_forest,
    solve_induced_tree,
    solve_minimum_maximal_matching,
)
from jacobian.domains.graph_optimization.operations import build_simple_graph
from jacobian.math_tools import MathTool

_WitnessValidator = Callable[[Any, ContractModel], bool]


def _validate_domination(graph: Any, result: ContractModel) -> bool:
    import networkx as nx

    result = cast(GraphDominationMinimumOutput, result)
    graph_vertices = set(graph)
    return set(result.witness_vertices) <= graph_vertices and nx.is_dominating_set(
        graph,
        result.witness_vertices,
    )


def _validate_matching(graph: Any, result: ContractModel) -> bool:
    import networkx as nx

    result = cast(GraphMinimumMaximalMatchingOutput, result)
    graph_vertices = set(graph)
    edges = set(result.witness_edges)
    return (
        all(left in graph_vertices and right in graph_vertices for left, right in edges)
        and nx.is_matching(graph, edges)
        and nx.is_maximal_matching(graph, edges)
    )


def _validate_induced_forest(graph: Any, result: ContractModel) -> bool:
    import networkx as nx

    result = cast(GraphInducedForestMaximumOutput, result)
    if not set(result.witness_vertices) <= set(graph):
        return False
    induced = graph.subgraph(result.witness_vertices)
    return induced.number_of_nodes() == 0 or nx.is_forest(induced)


def _validate_induced_tree(graph: Any, result: ContractModel) -> bool:
    import networkx as nx

    result = cast(GraphInducedTreeMaximumOutput, result)
    if not set(result.witness_vertices) <= set(graph):
        return False
    induced = graph.subgraph(result.witness_vertices)
    return (graph.number_of_nodes() == 0 and induced.number_of_nodes() == 0) or (
        induced.number_of_nodes() > 0
        and nx.is_connected(induced)
        and nx.is_forest(induced)
    )


def _validate_induced_bipartite(graph: Any, result: ContractModel) -> bool:
    import networkx as nx

    result = cast(GraphInducedBipartiteMaximumOutput, result)
    return set(result.witness_vertices) <= set(graph) and nx.is_bipartite(
        graph.subgraph(result.witness_vertices)
    )


_WITNESS_VALIDATORS: dict[type[ContractModel], _WitnessValidator] = {
    GraphDominationMinimumOutput: _validate_domination,
    GraphMinimumMaximalMatchingOutput: _validate_matching,
    GraphInducedForestMaximumOutput: _validate_induced_forest,
    GraphInducedTreeMaximumOutput: _validate_induced_tree,
    GraphInducedBipartiteMaximumOutput: _validate_induced_bipartite,
}


def _valid_witness(graph: Any, result: ContractModel) -> bool:
    validate = _WITNESS_VALIDATORS.get(type(result))
    return validate(graph, result) if validate else False


def _execute[ResultT: ContractModel](
    request: GraphOptimizationRequest,
    solve: Callable[
        [Any, ChromaticGraph, GraphOptimizationBudget],
        ResultT,
    ],
) -> ResultT:
    graph = cast(Any, build_simple_graph(request.graph))
    result = solve(graph, request.graph, request.resource_budget)
    if not _valid_witness(graph, result):
        raise RuntimeError("graph optimization backend returned an invalid witness")
    return result


def _operation[ResultT: ContractModel](
    operation_id: str,
    title: str,
    description: str,
    result_type: type[ResultT],
    solve: Callable[[Any, ChromaticGraph, GraphOptimizationBudget], ResultT],
    *tags: str,
) -> MathTool[GraphOptimizationRequest, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="1",
        title=title,
        description=description,
        request_type=GraphOptimizationRequest,
        result_type=result_type,
        run=lambda request: _execute(request, solve),
        tags=("graph", *tags, "bounded", "z3"),
    )


DOMINATION_MINIMUM_OPERATION = _operation(
    "graph.domination.minimum.compute",
    "Minimum dominating set",
    "Compute the domination number and an attaining set within explicit budgets.",
    GraphDominationMinimumOutput,
    lambda graph, contract, budget: solve_domination(graph, contract, budget),
    "domination",
    "minimum",
)

MINIMUM_MAXIMAL_MATCHING_OPERATION = _operation(
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

INDUCED_FOREST_MAXIMUM_OPERATION = _operation(
    "graph.induced_forest.maximum.compute",
    "Maximum induced forest",
    "Compute a maximum-order induced forest and vertex witness within explicit budgets.",
    GraphInducedForestMaximumOutput,
    lambda graph, contract, budget: solve_induced_forest(graph, contract, budget),
    "induced_forest",
    "maximum",
)

INDUCED_TREE_MAXIMUM_OPERATION = _operation(
    "graph.induced_tree.maximum.compute",
    "Maximum induced tree",
    "Compute a maximum-order induced tree and vertex witness within explicit budgets.",
    GraphInducedTreeMaximumOutput,
    lambda graph, contract, budget: solve_induced_tree(graph, contract, budget),
    "induced_tree",
    "maximum",
)

INDUCED_BIPARTITE_MAXIMUM_OPERATION = _operation(
    "graph.induced_bipartite.maximum.compute",
    "Maximum induced bipartite subgraph",
    "Compute a maximum-order induced bipartite subgraph within explicit budgets.",
    GraphInducedBipartiteMaximumOutput,
    lambda graph, contract, budget: solve_induced_bipartite(graph, contract, budget),
    "induced_bipartite",
    "maximum",
)

FINITE_GRAPH_OPTIMIZATION_OPERATIONS = (
    DOMINATION_MINIMUM_OPERATION,
    MINIMUM_MAXIMAL_MATCHING_OPERATION,
    INDUCED_FOREST_MAXIMUM_OPERATION,
    INDUCED_TREE_MAXIMUM_OPERATION,
    INDUCED_BIPARTITE_MAXIMUM_OPERATION,
)
