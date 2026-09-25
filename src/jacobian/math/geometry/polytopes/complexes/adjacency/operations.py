"""Exact maximal-cell facet adjacency graphs."""

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_COMPLEX_CELLS,
    PolytopalComplexClosureRequest,
)
from jacobian.math.geometry.polytopes.complexes.adjacency._models import (
    PolytopalAdjacencyCell,
    PolytopalComplexAdjacencyGraph,
    PolytopalFacetAdjacency,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_POLYTOPAL_ADJACENCY_RESULT_BYTES = 4 * 1024 * 1024


def polytopal_complex_adjacency_graph(
    request: PolytopalComplexClosureRequest,
) -> PolytopalComplexAdjacencyGraph:
    """Return the simple graph joining cells that share a codimension-one face.

    The supplied maximal-cell family is first admitted and closed by the
    canonical exact complex constructor. Graph vertices use its generated
    maximal-cell IDs. Two vertices are adjacent precisely when their exact
    pairwise intersection is a face of dimension one less than the ambient
    complex dimension. A vertex-only or lower-dimensional intersection does
    not create an edge.
    """

    if not isinstance(request, PolytopalComplexClosureRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="polytopal_complex.adjacency.request_type",
            message="expected a typed polytopal complex request",
        )
    if len(request.cells) > MAX_COMPLEX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.adjacency.cell_count_over_envelope",
            message=(
                "maximal-cell presentations exceed the "
                f"{MAX_COMPLEX_CELLS}-cell adjacency envelope"
            ),
        )

    complex_value = polytopal_complex_closure(request.cells)
    vertices = tuple(cell.cell_id for cell in complex_value.maximal_cells)
    vertex_count = len(vertices)
    maximum_edges = vertex_count * (vertex_count - 1) // 2
    if vertex_count > MAX_COMPLEX_CELLS or maximum_edges > 120:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.adjacency.graph_envelope",
            message="maximal-cell adjacency graph exceeds its vertex or edge bound",
        )

    face_by_id = {face.face_id: face for face in complex_value.faces}
    adjacent_rows = tuple(
        (record, face_by_id[record.intersection_face_id])
        for record in complex_value.pairwise_intersections
        if record.status == "face"
        and record.intersection_face_id is not None
        and face_by_id[record.intersection_face_id].dimension
        == complex_value.dimension - 1
    )
    edge_count = len(adjacent_rows)
    if edge_count > maximum_edges:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.adjacency.edge_count_over_envelope",
            message="maximal-cell adjacency edges exceed their combinatorial bound",
        )

    facet_edges = tuple(
        PolytopalFacetAdjacency(
            left_cell_id=record.first_cell_id,
            right_cell_id=record.second_cell_id,
            facet=face,
        )
        for record, face in adjacent_rows
    )
    cells = tuple(
        PolytopalAdjacencyCell(cell_id=cell.cell_id, vertices=cell.vertices)
        for cell in complex_value.maximal_cells
    )
    edges = tuple((row.left_cell_id, row.right_cell_id) for row in facet_edges)
    result_payload = {
        "space": complex_value.space.model_dump(mode="json"),
        "dimension": complex_value.dimension,
        "cells": [cell.model_dump(mode="json") for cell in cells],
        "graph": {"vertices": list(vertices), "edges": [list(edge) for edge in edges]},
        "facet_edges": [row.model_dump(mode="json") for row in facet_edges],
    }
    try:
        encode_strict_json(
            result_payload,
            limits=CanonicalLimits(
                max_output_bytes=MAX_POLYTOPAL_ADJACENCY_RESULT_BYTES
            ),
        )
    except CanonicalizationError as exc:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.adjacency.result_over_envelope",
            message=(
                "maximal-cell adjacency graph and labelled facets exceed the "
                f"{MAX_POLYTOPAL_ADJACENCY_RESULT_BYTES}-byte result bound"
            ),
        ) from exc
    return PolytopalComplexAdjacencyGraph(
        space=complex_value.space,
        dimension=complex_value.dimension,
        cells=cells,
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        facet_edges=facet_edges,
    )


__all__ = [
    "MAX_POLYTOPAL_ADJACENCY_RESULT_BYTES",
    "polytopal_complex_adjacency_graph",
]
