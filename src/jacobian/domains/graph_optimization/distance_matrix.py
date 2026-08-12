"""Exact all-pairs distances with source and target labels bound in the value."""

from __future__ import annotations

from typing import Any, cast

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.contracts.graph_invariant_operations import (
    GraphDistanceMatrixResult,
    GraphDistanceMatrixRow,
    GraphInvariantRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.graph_optimization.operations import build_simple_graph
from jacobian.operation_bindings import inline_operation
from jacobian.operations import OperationRefusalError, OperationSpec

_INVALID_REQUEST = CapabilityDiagnostic(
    code="INVALID_GRAPH_INVARIANT_REQUEST",
    stage="graph_invariant_input_validation",
    message="Input does not satisfy the bounded finite simple-graph contract.",
    hint="Supply a canonical simple graph with at most 32 vertices.",
)


def compute_distance_matrix(
    request: GraphInvariantRequest,
) -> GraphDistanceMatrixResult:
    """Compute every exact unweighted distance with explicit endpoint labels."""

    import networkx as nx

    try:
        graph = cast(Any, build_simple_graph(request.graph))
        vertices = tuple(sorted(graph.nodes))
        shortest_paths = {
            source: nx.single_source_shortest_path_length(graph, source)
            for source in vertices
        }
        rows = tuple(
            GraphDistanceMatrixRow(
                source_vertex=source,
                distances_by_target={
                    target: shortest_paths[source].get(target) for target in vertices
                },
            )
            for source in vertices
        )
        connected = bool(vertices) and all(
            distance is not None
            for row in rows
            for distance in row.distances_by_target.values()
        )
        return GraphDistanceMatrixResult(
            semantics_version="unweighted-shortest-path-distance-matrix.v3",
            row_ordering="SOURCE_VERTEX_LEXICOGRAPHIC_ASCENDING",
            target_ordering="TARGET_VERTEX_LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            target_vertices=vertices,
            rows=rows,
            connected=connected,
        )
    except (ArithmeticError, nx.NetworkXError, TypeError, ValueError) as exc:
        raise OperationRefusalError(
            CapabilityDiagnostic(
                code="GRAPH_INVARIANT_NOT_APPLICABLE",
                stage="graph_invariant_computation",
                message=str(exc),
                hint="Check the invariant's graph preconditions.",
            )
        ) from exc


DISTANCE_MATRIX_CAPABILITY = inline_operation(
    OperationSpec(
        operation_id="graph.distance_matrix.compute",
        version="3",
        title="All-pairs distance matrix",
        description=(
            "Compute every exact unweighted shortest-path distance in a bounded "
            "finite simple graph, binding every entry directly to source and target "
            "vertex labels and using JSON null for unreachable pairs."
        ),
        request_type=GraphInvariantRequest,
        result_type=GraphDistanceMatrixResult,
        execute=compute_distance_matrix,
        tags=("graph", "invariant", "distance", "matrix", "exact"),
        invalid_request=_INVALID_REQUEST,
        invocation_examples=(
            example(
                "path_three_distance_matrix",
                "Compute all ordered-pair distances in a three-vertex path.",
                {
                    "graph": {
                        "vertices": ["c", "a", "b"],
                        "edges": [["a", "b"], ["b", "c"]],
                    }
                },
            ),
        ),
    )
)


__all__ = ["DISTANCE_MATRIX_CAPABILITY", "compute_distance_matrix"]
