"""Exact maximal-cell facet adjacency graphs."""

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import RationalVPolytope
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_COMPLEX_CELLS,
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

MAX_POLYTOPAL_ADJACENCY_RESULT_COORDINATES = (16 * 64 + 120 * 64) * 4
MAX_POLYTOPAL_ADJACENCY_RESULT_DIGITS = 2_500_000


def _digits(value: int) -> int:
    return len(str(abs(value)))


def polytopal_complex_adjacency_graph(
    source_cells: tuple[RationalVPolytope, ...],
) -> PolytopalComplexAdjacencyGraph:
    """Return the simple graph joining cells that share a codimension-one face.

    The supplied maximal-cell family is first admitted and closed by the
    canonical exact complex constructor. Graph vertices use its generated
    maximal-cell IDs. Two vertices are adjacent precisely when their exact
    pairwise intersection is a face of dimension one less than the ambient
    complex dimension. A vertex-only or lower-dimensional intersection does
    not create an edge.
    """

    if not isinstance(source_cells, tuple):
        raise OperationDomainValidationError(
            location=("cells",),
            code="polytopal_complex.adjacency.cell_collection_type",
            message="maximal cells must be a canonical tuple",
        )
    if len(source_cells) > MAX_COMPLEX_CELLS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.adjacency.cell_count_over_envelope",
            message=(
                "maximal-cell presentations exceed the "
                f"{MAX_COMPLEX_CELLS}-cell adjacency envelope"
            ),
        )

    complex_value = polytopal_complex_closure(source_cells)
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

    cell_coordinate_count = sum(
        len(cell.vertices) * complex_value.dimension
        for cell in complex_value.maximal_cells
    )
    facet_coordinate_count = sum(
        len(face.vertices) * complex_value.dimension for _, face in adjacent_rows
    )
    result_coordinate_count = cell_coordinate_count + facet_coordinate_count
    if result_coordinate_count > MAX_POLYTOPAL_ADJACENCY_RESULT_COORDINATES:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.adjacency.coordinate_count_over_envelope",
            message="cell and shared-facet coordinate output exceeds its allocation bound",
        )
    result_digits = sum(
        _digits(coordinate.num) + _digits(coordinate.den)
        for cell in complex_value.maximal_cells
        for point in cell.vertices
        for coordinate in point.coordinates
    ) + sum(
        _digits(coordinate.num) + _digits(coordinate.den)
        for _, face in adjacent_rows
        for point in face.vertices
        for coordinate in point.coordinates
    )
    if result_digits > MAX_POLYTOPAL_ADJACENCY_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("cells",),
            code="polytopal_complex.adjacency.digit_count_over_envelope",
            message="exact cell and facet coordinate digits exceed their result bound",
        )

    facet_edges = tuple(
        sorted(
            (
                PolytopalFacetAdjacency(
                    left_cell_id=min(record.first_cell_id, record.second_cell_id),
                    right_cell_id=max(record.first_cell_id, record.second_cell_id),
                    facet=face,
                )
                for record, face in adjacent_rows
            ),
            key=lambda edge: (edge.left_cell_id, edge.right_cell_id),
        )
    )
    result_cells = tuple(
        PolytopalAdjacencyCell(cell_id=cell.cell_id, vertices=cell.vertices)
        for cell in complex_value.maximal_cells
    )
    edges = tuple((row.left_cell_id, row.right_cell_id) for row in facet_edges)
    return PolytopalComplexAdjacencyGraph(
        space=complex_value.space,
        dimension=complex_value.dimension,
        cells=result_cells,
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        facet_edges=facet_edges,
    )


__all__ = [
    "MAX_POLYTOPAL_ADJACENCY_RESULT_COORDINATES",
    "MAX_POLYTOPAL_ADJACENCY_RESULT_DIGITS",
    "polytopal_complex_adjacency_graph",
]
