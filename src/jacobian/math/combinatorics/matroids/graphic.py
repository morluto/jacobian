"""Bounded conversion of simple graphs to binary linear matroids."""

from __future__ import annotations

import json

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    MAX_GROUND_AXIS_CODEPOINTS,
    MAX_GROUND_SIZE,
    LinearMatroid,
)
from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _reject(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _refuse(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(location=location, code=code, message=message)


def graphic_matroid(request: SimpleUndirectedGraph) -> LinearMatroid:
    """Return the binary incidence-column representation of a simple graph."""

    graph = request
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
    columns = len(edges)
    # The incidence entries are GF(2) residues and the ground axis is a JSON
    # pair of endpoint labels, so the retained result grows only with the two
    # cardinality bounds admitted above plus the endpoint label text each
    # pair repeats. Charge that text in Unicode codepoints, not encoded bytes.
    retained_axis_codepoints = sum(
        len(json.dumps(edge, ensure_ascii=False, separators=(",", ":")))
        for edge in edges
    )
    if retained_axis_codepoints > MAX_GROUND_AXIS_CODEPOINTS:
        _refuse(
            ("graph",),
            "matroid.graphic.retained_axis_bound",
            f"retained ground axis exceeds the {MAX_GROUND_AXIS_CODEPOINTS}"
            "-codepoint allocation bound",
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
    "graphic_matroid",
]
