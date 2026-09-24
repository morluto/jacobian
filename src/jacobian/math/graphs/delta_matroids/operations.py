"""Exact constructions connecting looped graphs and delta-matroids."""

from __future__ import annotations

from jacobian.math.combinatorics.matroids.delta.extra import BinarySymmetricMatrix
from jacobian.math.combinatorics.matroids.delta.extra_ops import binary
from jacobian.math.graphs.delta_matroids._models import (
    LoopedGraphDeltaMatroidRequest,
    LoopedGraphDeltaMatroidResult,
)
from jacobian.math.graphs.values import LoopedSimpleGraph


def looped_adjacency_delta_matroid(
    graph: LoopedSimpleGraph,
) -> LoopedGraphDeltaMatroidResult:
    """Return the principal-minor delta-matroid of a looped simple graph."""

    request = LoopedGraphDeltaMatroidRequest(graph=graph)
    graph = LoopedSimpleGraph.model_validate(request.graph.model_dump(mode="python"))
    vertices = graph.vertices
    edge_set = set(graph.edges)
    loop_set = set(graph.loops)
    matrix = BinarySymmetricMatrix(
        ground=vertices,
        entries=tuple(
            tuple(
                int(
                    (vertices[i] in loop_set)
                    if i == j
                    else tuple(sorted((vertices[i], vertices[j]))) in edge_set
                )
                for j in range(len(vertices))
            )
            for i in range(len(vertices))
        ),
    )
    binary_result = binary(matrix)
    return LoopedGraphDeltaMatroidResult(
        graph=graph,
        matrix=binary_result.matrix,
        delta_matroid=binary_result.delta_matroid,
    )


__all__ = ["looped_adjacency_delta_matroid"]
