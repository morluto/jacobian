"""Bounded conversion of simple graphs to binary linear matroids."""

from __future__ import annotations

import json

from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    MAX_GROUND_SIZE,
    GraphicMatroidRequest,
    LinearMatroid,
)
from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

MAX_GRAPHIC_MATROID_RESULT_BYTES = CanonicalLimits().max_output_bytes
"""Canonical JSON egress limit used by the incidence expansion preflight."""

MAX_GRAPHIC_MATROID_LABEL_BYTES = 2_048
"""Conservative wire allowance for one JSON-encoded endpoint pair label."""


def _reject(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _refuse(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(location=location, code=code, message=message)


def graphic_matroid(request: GraphicMatroidRequest) -> LinearMatroid:
    """Return the binary incidence-column representation of a simple graph."""

    if not isinstance(request, GraphicMatroidRequest):
        _reject(("request",), "matroid.graphic.request_type", "invalid graph request")
    graph = request.graph
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject(("graph",), "matroid.graphic.graph_type", "graph must be canonical")
    if len(graph.vertices) > MAX_SIMPLE_GRAPH_VERTICES:
        _refuse(
            ("graph", "vertices"),
            "matroid.graphic.vertex_bound",
            f"graphic representation admits at most {MAX_SIMPLE_GRAPH_VERTICES} vertices",
        )
    if len(graph.edges) > MAX_GROUND_SIZE:
        _refuse(
            ("graph", "edges"),
            "matroid.graphic.edge_bound",
            f"linear-matroid ground set admits at most {MAX_GROUND_SIZE} graph edges",
        )

    try:
        source = SimpleUndirectedGraph.model_validate(graph.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("graph",),
            code="matroid.graphic.invalid_graph",
            message="graph must satisfy the canonical simple-graph contract",
        ) from exc

    vertices = tuple(sorted(source.vertices))
    edges = tuple(sorted(source.edges))
    rows = len(vertices)
    columns = len(edges)
    cells = rows * columns
    estimated_output_bytes = (
        cells * 2 + rows * 4 + columns * MAX_GRAPHIC_MATROID_LABEL_BYTES + 4_096
    )
    if estimated_output_bytes > MAX_GRAPHIC_MATROID_RESULT_BYTES:
        _refuse(
            ("graph",),
            "matroid.graphic.result_bytes",
            f"conservative result bound {estimated_output_bytes} exceeds "
            f"{MAX_GRAPHIC_MATROID_RESULT_BYTES} bytes",
        )

    # JSON array encoding is injective on endpoint pairs and independent of
    # delimiters that may occur in graph labels.
    ground_labels = tuple(
        json.dumps(edge, ensure_ascii=False, separators=(",", ":")) for edge in edges
    )
    vertex_indices = {vertex: index for index, vertex in enumerate(vertices)}
    entries = [[0] * columns for _ in vertices]
    for column, (left, right) in enumerate(edges):
        entries[vertex_indices[left]][column] = 1
        entries[vertex_indices[right]][column] = 1
    matrix = PrimeFieldMatrix._from_admitted(
        prime=2,
        entries=tuple(tuple(row) for row in entries),
        columns=columns,
    )
    return LinearMatroid(matrix=matrix, ground_labels=ground_labels)


__all__ = [
    "MAX_GRAPHIC_MATROID_RESULT_BYTES",
    "graphic_matroid",
]
