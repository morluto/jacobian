"""Exact all-pairs distances under a distance-matrix-owned graph bound."""

from __future__ import annotations

from typing import Any, cast

from jacobian.contracts.graph_distance_matrix import (
    GraphDistanceMatrixRequest,
    GraphDistanceMatrixResult,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains._examples import example
from jacobian.domains.graph_optimization.operations import build_simple_graph
from jacobian.operation_declarations import OperationDeclaration, OperationRefusalError


def compute_distance_matrix(
    request: GraphDistanceMatrixRequest,
) -> GraphDistanceMatrixResult:
    """Compute every exact unweighted distance in canonical vertex order."""

    import networkx as nx

    try:
        graph = cast(Any, build_simple_graph(request.graph))
        vertices = tuple(sorted(graph.nodes))
        shortest_paths = {
            source: nx.single_source_shortest_path_length(graph, source)
            for source in vertices
        }
        distances = tuple(
            tuple(shortest_paths[source].get(target) for target in vertices)
            for source in vertices
        )
        connected = bool(vertices) and all(
            distance is not None for row in distances for distance in row
        )
        return GraphDistanceMatrixResult(
            semantics_version="unweighted-shortest-path-distance-matrix.v1",
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=vertices,
            distances=distances,
            connected=connected,
        )
    except (ArithmeticError, nx.NetworkXError, TypeError, ValueError) as exc:
        raise OperationRefusalError(
            OperationDiagnostic(
                code="GRAPH_DISTANCE_MATRIX_NOT_APPLICABLE",
                stage="graph_distance_matrix_computation",
                message=str(exc),
                hint="Check the distance matrix graph preconditions.",
            )
        ) from exc


DISTANCE_MATRIX_OPERATION = OperationDeclaration(
    operation_id="graph.distance_matrix.compute",
    version="2",
    title="All-pairs distance matrix",
    description=(
        "Compute every exact unweighted shortest-path distance in a finite "
        "simple graph of at most 64 vertices, using JSON null for unreachable "
        "vertex pairs."
    ),
    request_type=GraphDistanceMatrixRequest,
    result_type=GraphDistanceMatrixResult,
    execute=compute_distance_matrix,
    invalid_request=OperationDiagnostic(
        code="INVALID_GRAPH_DISTANCE_MATRIX_REQUEST",
        stage="graph_distance_matrix_input_validation",
        message="Input does not satisfy the bounded graph distance-matrix contract.",
        hint="Supply a canonical simple graph with at most 64 vertices.",
    ),
    tags=("graph", "invariant", "distance", "matrix", "exact"),
    examples=(
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

__all__ = ["DISTANCE_MATRIX_OPERATION", "compute_distance_matrix"]
