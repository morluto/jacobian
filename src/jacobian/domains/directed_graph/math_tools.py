"""Exact directed-graph operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.directed_graph import (
    AcyclicOrderRequest,
    AcyclicOrderResult,
    DegreeProfileRequest,
    DegreeProfileResult,
    ReachabilityRequest,
    ReachabilityResult,
    StrongComponentsRequest,
    StrongComponentsResult,
    TransitiveClosureRequest,
    TransitiveClosureResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.directed_graph.operations import (
    compute_acyclic_order,
    compute_degree_profile,
    compute_reachability,
    compute_strong_components,
    compute_transitive_closure,
)
from jacobian.math_tools import MathTool


def directed_graph_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
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
        "digraph.reachability.compute",
        "Compute directed reachability from a source vertex",
        "Compute reachable/unreachable vertex sets, minimum directed distances, predecessor witnesses, and optionally a shortest path to a target vertex.",
        ReachabilityRequest,
        ReachabilityResult,
        compute_reachability,
        "directed",
        "graph",
        "reachability",
        "exact",
        examples=(
            example(
                "simple_reachability",
                "Compute reachability from vertex 0 in a simple chain.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 3]],
                    },
                    "source": 0,
                },
            ),
            example(
                "reachability_with_target",
                "Compute reachability from vertex 0 and check target vertex 3.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 3]],
                    },
                    "source": 0,
                    "target": 3,
                },
            ),
        ),
    ),
    directed_graph_operation(
        "digraph.strong_components.compute",
        "Compute strongly connected components and condensation DAG",
        "Compute the SCC partition, component IDs, condensation DAG, source/sink components, and strong connectivity using NetworkX.",
        StrongComponentsRequest,
        StrongComponentsResult,
        compute_strong_components,
        "directed",
        "graph",
        "scc",
        "condensation",
        "exact",
        examples=(
            example(
                "simple_scc",
                "Compute SCCs in a graph with a 3-cycle and one extra vertex.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 0]],
                    },
                },
            ),
        ),
    ),
    directed_graph_operation(
        "digraph.acyclic_order.compute",
        "Compute a topological order or detect a directed cycle",
        "Compute a deterministic topological order with positions for a DAG, or return a concrete directed cycle witness if the graph is cyclic.",
        AcyclicOrderRequest,
        AcyclicOrderResult,
        compute_acyclic_order,
        "directed",
        "graph",
        "topological",
        "acyclic",
        "cycle",
        "exact",
        examples=(
            example(
                "acyclic_chain",
                "Compute the topological order of a simple 4-vertex chain.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 3]],
                    },
                },
            ),
            example(
                "cyclic_graph",
                "Detect a directed cycle in a 3-vertex cycle graph.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 0]],
                    },
                },
            ),
        ),
    ),
    directed_graph_operation(
        "digraph.transitive_closure.compute",
        "Compute the transitive closure of a directed graph",
        "Compute the complete reachable ordered-pair relation with an explicit reflexive convention.",
        TransitiveClosureRequest,
        TransitiveClosureResult,
        compute_transitive_closure,
        "directed",
        "graph",
        "transitive-closure",
        "exact",
        examples=(
            example(
                "chain_closure",
                "Compute transitive closure of a 4-vertex chain (irreflexive).",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 3]],
                    },
                    "reflexive": False,
                },
            ),
            example(
                "chain_closure_reflexive",
                "Compute reflexive transitive closure of a 4-vertex chain.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 3]],
                    },
                    "reflexive": True,
                },
            ),
        ),
    ),
    directed_graph_operation(
        "digraph.degree_profile.compute",
        "Compute in-degree and out-degree profile for every vertex",
        "Compute exact in-degree and out-degree for every vertex plus sources, sinks/dead ends, and isolated vertices.",
        DegreeProfileRequest,
        DegreeProfileResult,
        compute_degree_profile,
        "directed",
        "graph",
        "degree",
        "exact",
        examples=(
            example(
                "simple_degree_profile",
                "Compute degree profile of a simple 4-vertex chain.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "arcs": [[0, 1], [1, 2], [2, 3]],
                    },
                },
            ),
        ),
    ),
)
