"""Exact all-pairs distances under a distance-matrix-owned graph bound."""

from __future__ import annotations

from typing import Any, cast

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graph_optimization._distance_models import (
    GraphDistanceMatrixRequest,
    GraphDistanceMatrixResult,
)
from jacobian.math.graph_optimization._operations import build_simple_graph


def compute_distance_matrix(
    request: GraphDistanceMatrixRequest,
) -> GraphDistanceMatrixResult:
    """Compute every exact unweighted distance in canonical vertex order."""

    import networkx as nx

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


DISTANCE_MATRIX_OPERATION = MathTool(
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
    run=compute_distance_matrix,
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
