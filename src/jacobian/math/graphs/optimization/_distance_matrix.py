"""Exact all-pairs distances under a distance-matrix-owned graph bound."""

from __future__ import annotations

from typing import Any, cast

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.optimization._chromatic_kernel import build_simple_graph
from jacobian.math.graphs.optimization._distance_models import (
    GraphDistanceMatrixRequest,
    GraphDistanceMatrixResult,
    GraphDistanceRow,
)


def compute_distance_matrix(
    request: GraphDistanceMatrixRequest,
) -> GraphDistanceMatrixResult:
    """Compute every exact unweighted distance in canonical vertex order.

    Rows are labelled with their source vertex so the dense positional
    matrix stays bound to the authoritative lexicographic vertex axis.
    """

    import networkx as nx

    graph = cast(Any, build_simple_graph(request.graph))
    vertices = tuple(sorted(graph.nodes))
    shortest_paths = {
        source: nx.single_source_shortest_path_length(graph, source)
        for source in vertices
    }
    rows = tuple(
        GraphDistanceRow(
            source=source,
            distances=tuple(shortest_paths[source].get(target) for target in vertices),
        )
        for source in vertices
    )
    connected = bool(vertices) and all(
        distance is not None for row in rows for distance in row.distances
    )
    return GraphDistanceMatrixResult._from_kernel(
        vertices=vertices,
        rows=rows,
        connected=connected,
    )


DISTANCE_MATRIX_OPERATION = MathTool(
    operation_id="graph.distance_matrix.compute",
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
