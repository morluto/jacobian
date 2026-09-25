"""Exact constructions connecting looped graphs and delta-matroids."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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

    try:
        canonical = LoopedSimpleGraph.model_validate(graph)
    except ValidationError as error:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.looped_graph_invalid",
            message="graph must be a canonical LoopedSimpleGraph value",
        ) from error
    try:
        LoopedGraphDeltaMatroidRequest(graph=canonical)
    except ValidationError as error:
        raise OperationResourceAdmissionError(
            location=("graph", "vertices"),
            code="delta_matroid.binary_work",
            message="looped graph conversion supports at most 8 vertices",
        ) from error
    graph = canonical
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
