"""Five bounded exact finite-graph optimization operations."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.graphs.optimization._chromatic_kernel import build_simple_graph
from jacobian.math.graphs.optimization._exact_search import (
    solve_domination,
    solve_induced_bipartite,
    solve_induced_forest,
    solve_induced_tree,
    solve_minimum_maximal_matching,
)
from jacobian.math.graphs.optimization._models import (
    GraphDominationMinimumOutput,
    GraphInducedBipartiteMaximumOutput,
    GraphInducedForestMaximumOutput,
    GraphInducedTreeMaximumOutput,
    GraphMinimumMaximalMatchingOutput,
    GraphOptimizationRequest,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_WitnessValidator = Callable[[Any, StrictModel], bool]
_OPTIMIZATION_WORKER = Path(__file__).with_name("_finite_optimization_worker.py")
_WORKER_OUTPUT_BYTES = 64 * 1024
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _validate_domination(graph: Any, result: StrictModel) -> bool:
    import networkx as nx

    result = cast(GraphDominationMinimumOutput, result)
    graph_vertices = set(graph)
    return set(result.witness_vertices) <= graph_vertices and nx.is_dominating_set(
        graph,
        result.witness_vertices,
    )


def _validate_matching(graph: Any, result: StrictModel) -> bool:
    import networkx as nx

    result = cast(GraphMinimumMaximalMatchingOutput, result)
    graph_vertices = set(graph)
    edges = set(result.witness_edges)
    return (
        all(left in graph_vertices and right in graph_vertices for left, right in edges)
        and nx.is_matching(graph, edges)
        and nx.is_maximal_matching(graph, edges)
    )


def _validate_induced_forest(graph: Any, result: StrictModel) -> bool:
    import networkx as nx

    result = cast(GraphInducedForestMaximumOutput, result)
    if not set(result.witness_vertices) <= set(graph):
        return False
    induced = graph.subgraph(result.witness_vertices)
    return induced.number_of_nodes() == 0 or nx.is_forest(induced)


def _validate_induced_tree(graph: Any, result: StrictModel) -> bool:
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


def _validate_induced_bipartite(graph: Any, result: StrictModel) -> bool:
    import networkx as nx

    result = cast(GraphInducedBipartiteMaximumOutput, result)
    return set(result.witness_vertices) <= set(graph) and nx.is_bipartite(
        graph.subgraph(result.witness_vertices)
    )


_WITNESS_VALIDATORS: dict[type[StrictModel], _WitnessValidator] = {
    GraphDominationMinimumOutput: _validate_domination,
    GraphMinimumMaximalMatchingOutput: _validate_matching,
    GraphInducedForestMaximumOutput: _validate_induced_forest,
    GraphInducedTreeMaximumOutput: _validate_induced_tree,
    GraphInducedBipartiteMaximumOutput: _validate_induced_bipartite,
}


def _valid_witness(graph: Any, result: StrictModel) -> bool:
    validate = _WITNESS_VALIDATORS.get(type(result))
    return validate(graph, result) if validate else False


def _fallback_unknown[ResultT: StrictModel](
    graph: Any,
    request: GraphOptimizationRequest,
    result_type: type[ResultT],
    detail: str,
) -> ResultT:
    """Return a source-derived feasible incumbent without an optimum claim."""

    vertices = tuple(sorted(request.graph.vertices))
    common: dict[str, object] = {
        "status": "UNKNOWN",
        "order": len(vertices),
        "optimum_value": None,
        "tested": (),
        "termination_reason": "SOLVER_UNKNOWN",
        "detail": detail,
    }
    if result_type is GraphDominationMinimumOutput:
        return result_type.model_validate(
            {
                **common,
                "incumbent_value": len(vertices),
                "lower_bound": 0 if not vertices else 1,
                "upper_bound": len(vertices),
                "witness_vertices": vertices,
            }
        )
    if result_type is GraphMinimumMaximalMatchingOutput:
        used: set[str] = set()
        selected: list[tuple[str, str]] = []
        for left, right in sorted(graph.edges):
            if left not in used and right not in used:
                selected.append((left, right) if left < right else (right, left))
                used.update((left, right))
        edges = tuple(selected)
        return result_type.model_validate(
            {
                **common,
                "incumbent_value": len(edges),
                "lower_bound": 0,
                "upper_bound": len(edges),
                "witness_edges": edges,
            }
        )
    witness = () if not vertices else (vertices[0],)
    return result_type.model_validate(
        {
            **common,
            "incumbent_value": len(witness),
            "lower_bound": len(witness),
            "upper_bound": len(vertices),
            "witness_vertices": witness,
        }
    )


def _execute[ResultT: StrictModel](
    operation_id: str,
    request: GraphOptimizationRequest,
    result_type: type[ResultT],
) -> ResultT:
    """Run one complete graph Z3 operation in its bounded owner worker."""

    if len(request.graph.vertices) > request.resource_budget.max_order:
        raise OperationDomainValidationError(
            location=("resource_budget", "max_order"),
            code="graph.optimization.max_order_budget",
            message="graph order exceeds the declared max_order budget",
        )
    graph = cast(Any, build_simple_graph(request.graph))
    deadline = time.monotonic() + request.resource_budget.wall_seconds
    try:
        with TemporaryDirectory(prefix="jacobian-graph-optimization-") as directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return _fallback_unknown(
                    graph,
                    request,
                    result_type,
                    "the graph optimization request expired before worker startup",
                )
            completed = run_bounded_process(
                [sys.executable, str(_OPTIMIZATION_WORKER)],
                input_bytes=json.dumps(
                    {
                        "operation_id": operation_id,
                        "request": request.model_dump(mode="json"),
                    },
                    separators=(",", ":"),
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_OUTPUT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(request.resource_budget.wall_seconds)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return _fallback_unknown(
            graph,
            request,
            result_type,
            "the bounded graph optimization worker could not be started",
        )
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return _fallback_unknown(
            graph,
            request,
            result_type,
            "the bounded graph optimization worker did not establish an outcome",
        )
    if time.monotonic() >= deadline:
        return _fallback_unknown(
            graph,
            request,
            result_type,
            "the graph optimization request expired before response validation",
        )
    try:
        result = result_type.model_validate(
            json.loads(completed.stdout.decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return _fallback_unknown(
            graph,
            request,
            result_type,
            "the bounded graph optimization worker returned malformed output",
        )
    if getattr(result, "order", None) != len(
        request.graph.vertices
    ) or not _valid_witness(graph, result):
        return _fallback_unknown(
            graph,
            request,
            result_type,
            "the bounded graph optimization worker returned an invalid witness",
        )
    if time.monotonic() >= deadline:
        return _fallback_unknown(
            graph,
            request,
            result_type,
            "the graph optimization request expired during response validation",
        )
    return result


def _operation[ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    result_type: type[ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[GraphOptimizationRequest, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=GraphOptimizationRequest,
        result_type=result_type,
        run=lambda request: _execute(operation_id, request, result_type),
        tags=("graph", *tags, "bounded", "z3"),
        examples=examples,
    )


DOMINATION_MINIMUM_OPERATION = _operation(
    "graph.domination.minimum.compute",
    "Minimum dominating set",
    "Compute the domination number and an attaining set within explicit budgets.",
    GraphDominationMinimumOutput,
    "domination",
    "minimum",
    examples=(
        example(
            "path_3",
            "Path graph on 3 vertices.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "resource_budget": {"max_solver_calls": 33},
            },
        ),
    ),
)

MINIMUM_MAXIMAL_MATCHING_OPERATION = _operation(
    "graph.matching.maximal.minimum.compute",
    "Minimum maximal matching",
    "Compute the saturation number and an attaining maximal matching within explicit budgets.",
    GraphMinimumMaximalMatchingOutput,
    "matching",
    "saturation_number",
    "minimum",
    examples=(
        example(
            "path_3_matching",
            "Minimum maximal matching of path graph P3.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "resource_budget": {"max_solver_calls": 33},
            },
        ),
    ),
)

INDUCED_FOREST_MAXIMUM_OPERATION = _operation(
    "graph.induced_forest.maximum.compute",
    "Maximum induced forest",
    "Compute a maximum-order induced forest and vertex witness within explicit budgets.",
    GraphInducedForestMaximumOutput,
    "induced_forest",
    "maximum",
    examples=(
        example(
            "path_3",
            "Path graph P3.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "resource_budget": {"max_solver_calls": 33},
            },
        ),
    ),
)

INDUCED_TREE_MAXIMUM_OPERATION = _operation(
    "graph.induced_tree.maximum.compute",
    "Maximum induced tree",
    "Compute a maximum-order induced tree and vertex witness within explicit budgets.",
    GraphInducedTreeMaximumOutput,
    "induced_tree",
    "maximum",
    examples=(
        example(
            "path_3",
            "Path graph P3.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "resource_budget": {"max_solver_calls": 33},
            },
        ),
    ),
)

INDUCED_BIPARTITE_MAXIMUM_OPERATION = _operation(
    "graph.induced_bipartite.maximum.compute",
    "Maximum induced bipartite subgraph",
    "Compute a maximum-order induced bipartite subgraph within explicit budgets.",
    GraphInducedBipartiteMaximumOutput,
    "induced_bipartite",
    "maximum",
    examples=(
        example(
            "path_3",
            "Path graph P3.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "resource_budget": {"max_solver_calls": 33},
            },
        ),
    ),
)

FINITE_GRAPH_OPTIMIZATION_OPERATIONS = (
    DOMINATION_MINIMUM_OPERATION,
    MINIMUM_MAXIMAL_MATCHING_OPERATION,
    INDUCED_FOREST_MAXIMUM_OPERATION,
    INDUCED_TREE_MAXIMUM_OPERATION,
    INDUCED_BIPARTITE_MAXIMUM_OPERATION,
)


def _run_worker_kernel(
    operation_id: str,
    request: GraphOptimizationRequest,
) -> StrictModel:
    """Build, encode, solve, and validate one graph operation inside its worker."""

    graph = cast(Any, build_simple_graph(request.graph))
    started = time.monotonic()
    result: StrictModel
    if operation_id == "graph.domination.minimum.compute":
        result = solve_domination(
            graph, request.graph, request.resource_budget, started
        )
    elif operation_id == "graph.matching.maximal.minimum.compute":
        result = solve_minimum_maximal_matching(
            graph, request.graph, request.resource_budget, started
        )
    elif operation_id == "graph.induced_forest.maximum.compute":
        result = solve_induced_forest(
            graph, request.graph, request.resource_budget, started
        )
    elif operation_id == "graph.induced_tree.maximum.compute":
        result = solve_induced_tree(
            graph, request.graph, request.resource_budget, started
        )
    elif operation_id == "graph.induced_bipartite.maximum.compute":
        result = solve_induced_bipartite(
            graph, request.graph, request.resource_budget, started
        )
    else:
        raise ValueError("unknown graph optimization operation")
    if not _valid_witness(graph, result):
        raise ValueError("graph optimization kernel returned an invalid witness")
    return result
