"""Canonical labelled adjacency graphs of rational polytopal cells."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError
from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalVPolytope,
    _filter_redundant_vertices,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_COMPLEX_DIMENSION,
    ComplexFace,
    ComplexPoint,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_ADJACENCY_RESULT_COMPONENT_DIGITS = 1_024
"""Maximum exact coordinate component size admitted before geometric checks."""

MAX_ADJACENCY_RESULT_COORDINATE_BITS = 10_000_000
"""Aggregate exact-coordinate bit budget for result-value validation."""


class PolytopalAdjacencyRequest(StrictModel):
    """Bounded maximal-cell input for the adjacency operation."""

    cells: tuple[RationalVPolytope, ...] = Field(
        min_length=1,
        max_length=16,
        description="Between 1 and 16 maximal rational cells in one ambient space.",
    )


def _facet_support_sides(
    facet_points: set[tuple[CanonicalRational, ...]],
    cell_points: set[tuple[CanonicalRational, ...]],
    dimension: int,
) -> tuple[set[int], set[tuple[CanonicalRational, ...]]] | None:
    """Return exact support sides and all vertices on the facet plane."""

    ordered_points = tuple(
        sorted(
            facet_points,
            key=lambda point: tuple((value.num, value.den) for value in point),
        )
    )
    if not ordered_points or any(len(point) != dimension for point in ordered_points):
        return None
    rational_points = [
        [Rational(value.num, value.den) for value in point] for point in ordered_points
    ]
    rows = [[*point, 1] for point in rational_points]
    plane_space = Matrix(rows).nullspace()
    if len(plane_space) != 1:
        return None
    coefficients = plane_space[0]
    sides: set[int] = set()
    zero_points: set[tuple[CanonicalRational, ...]] = set()
    for point in cell_points:
        rational = [Rational(value.num, value.den) for value in point]
        evaluation = (
            sum(
                coefficient * coordinate
                for coefficient, coordinate in zip(
                    coefficients[:-1], rational, strict=True
                )
            )
            + coefficients[-1]
        )
        if evaluation > 0:
            sides.add(1)
        elif evaluation < 0:
            sides.add(-1)
        else:
            zero_points.add(point)
    return sides, zero_points


def _validate_cell_geometry(cell: PolytopalAdjacencyCell, dimension: int) -> None:
    points = tuple(point.coordinates for point in cell.vertices)
    ordered_points = tuple(
        sorted(
            points,
            key=lambda point: tuple(Rational(value.num, value.den) for value in point),
        )
    )
    if points != ordered_points:
        raise PydanticCustomError(
            "polytopal_complex.adjacency_cell_vertex_order",
            "maximal-cell vertices must use canonical coordinate order",
        )
    exact_points = [
        [Rational(value.num, value.den) for value in point] for point in points
    ]
    differences = [
        [coordinate - exact_points[0][axis] for axis, coordinate in enumerate(point)]
        for point in exact_points[1:]
    ]
    if Matrix(differences).rank() != dimension:
        raise PydanticCustomError(
            "polytopal_complex.adjacency_cell_dimension",
            "every maximal cell must be full-dimensional in the shared space",
        )
    if len(_filter_redundant_vertices(exact_points, dimension)) != len(exact_points):
        raise PydanticCustomError(
            "polytopal_complex.adjacency_cell_extreme_vertices",
            "maximal-cell coordinates must all be extreme hull vertices",
        )


def _validate_value_coordinates(
    cells: tuple[PolytopalAdjacencyCell, ...],
    facet_edges: tuple[PolytopalFacetAdjacency, ...],
    dimension: int,
) -> dict[str, set[tuple[CanonicalRational, ...]]]:
    point_maps: dict[str, set[tuple[CanonicalRational, ...]]] = {}
    coordinate_bits = 0
    for cell in cells:
        points = tuple(point.coordinates for point in cell.vertices)
        if len(set(points)) != len(points) or any(
            len(point) != dimension for point in points
        ):
            raise PydanticCustomError(
                "polytopal_complex.adjacency_cell_coordinates",
                "maximal-cell coordinates must be unique and use the ambient dimension",
            )
        for cell_point in points:
            for coordinate in cell_point:
                _validate_coordinate_bound(coordinate)
                coordinate_bits += (
                    coordinate.num.bit_length() + coordinate.den.bit_length()
                )
        point_maps[cell.cell_id] = set(points)
    for edge in facet_edges:
        for facet_point in edge.facet.vertices:
            for coordinate in facet_point.coordinates:
                _validate_coordinate_bound(coordinate)
                coordinate_bits += (
                    coordinate.num.bit_length() + coordinate.den.bit_length()
                )
    if coordinate_bits > MAX_ADJACENCY_RESULT_COORDINATE_BITS:
        raise PydanticCustomError(
            "polytopal_complex.adjacency_coordinate_work",
            "exact adjacency coordinate validation exceeds its bit budget",
        )
    for cell in cells:
        _validate_cell_geometry(cell, dimension)
    return point_maps


def _validate_coordinate_bound(coordinate: CanonicalRational) -> None:
    try:
        require_bounded_rational(
            coordinate,
            max_digits=MAX_ADJACENCY_RESULT_COMPONENT_DIGITS,
            label="adjacency result coordinate",
        )
    except ValueError as exc:
        raise PydanticCustomError(
            "polytopal_complex.adjacency_coordinate_digits",
            str(exc),
        ) from exc


def _validate_facet_edge(
    edge: PolytopalFacetAdjacency,
    point_maps: dict[str, set[tuple[CanonicalRational, ...]]],
    dimension: int,
) -> None:
    if edge.facet.dimension != dimension - 1:
        raise PydanticCustomError(
            "polytopal_complex.adjacency_facet_dimension",
            "each graph edge must carry a codimension-one face",
        )
    facet_points = {point.coordinates for point in edge.facet.vertices}
    left_points = point_maps[edge.left_cell_id]
    right_points = point_maps[edge.right_cell_id]
    if facet_points != left_points & right_points:
        raise PydanticCustomError(
            "polytopal_complex.adjacency_facet_vertices",
            "shared facet vertices must be exactly the common cell vertices",
        )
    left_support = _facet_support_sides(facet_points, left_points, dimension)
    right_support = _facet_support_sides(facet_points, right_points, dimension)
    if (
        left_support is None
        or right_support is None
        or len(left_support[0]) != 1
        or len(right_support[0]) != 1
        or left_support[0] == right_support[0]
        or left_support[1] != facet_points
        or right_support[1] != facet_points
    ):
        raise PydanticCustomError(
            "polytopal_complex.adjacency_facet_support",
            "shared vertices must span a supporting facet on opposite sides of both cell hulls",
        )


class PolytopalAdjacencyCell(StrictModel):
    """One canonical maximal cell attached to its graph vertex label."""

    cell_id: str = Field(min_length=1, max_length=64)
    vertices: tuple[ComplexPoint, ...] = Field(min_length=1, max_length=64)


class PolytopalFacetAdjacency(StrictModel):
    """One graph edge and its exact shared codimension-one face."""

    left_cell_id: str = Field(min_length=1, max_length=64)
    right_cell_id: str = Field(min_length=1, max_length=64)
    facet: ComplexFace

    @model_validator(mode="after")
    def canonical_edge(self) -> Self:
        if self.left_cell_id >= self.right_cell_id:
            raise PydanticCustomError(
                "polytopal_complex.adjacency_edge_order",
                "adjacency endpoints must be distinct and lexicographically ordered",
            )
        if self.facet.maximal_cell_ids != (
            self.left_cell_id,
            self.right_cell_id,
        ):
            raise PydanticCustomError(
                "polytopal_complex.adjacency_facet_support",
                "shared facet support must be exactly the adjacent cell pair",
            )
        return self


class PolytopalComplexAdjacencyGraph(StrictModel):
    """Facet adjacency graph with its exact cell and shared-facet labels.

    ``graph`` is the reusable ordinary simple graph. ``cells`` binds every
    vertex label to the canonical maximal-cell geometry, and ``facet_edges``
    identifies the exact shared face for each graph edge.
    """

    space: RationalCoordinateSpace
    dimension: int = Field(ge=1, le=MAX_COMPLEX_DIMENSION)
    cells: tuple[PolytopalAdjacencyCell, ...] = Field(min_length=1, max_length=16)
    graph: SimpleUndirectedGraph
    facet_edges: tuple[PolytopalFacetAdjacency, ...] = Field(max_length=120)

    @model_validator(mode="after")
    def canonical_graph_binding(self) -> Self:
        if len(self.space.axes) != self.dimension:
            raise PydanticCustomError(
                "polytopal_complex.adjacency_space_dimension",
                "ambient coordinate count must equal the complex dimension",
            )
        cell_ids = tuple(cell.cell_id for cell in self.cells)
        if cell_ids != tuple(f"M{index}" for index in range(len(cell_ids))):
            raise PydanticCustomError(
                "polytopal_complex.adjacency_cell_order",
                "adjacency cells must follow the complex's canonical numeric ID order",
            )
        if self.graph.vertices != cell_ids:
            raise PydanticCustomError(
                "polytopal_complex.adjacency_vertex_binding",
                "graph vertices must be exactly the canonical maximal-cell IDs",
            )
        point_maps = _validate_value_coordinates(
            self.cells, self.facet_edges, self.dimension
        )
        edge_keys = tuple(
            (edge.left_cell_id, edge.right_cell_id) for edge in self.facet_edges
        )
        if edge_keys != tuple(sorted(set(edge_keys))):
            raise PydanticCustomError(
                "polytopal_complex.adjacency_edge_order",
                "facet adjacency rows must be unique and ordered by cell pair",
            )
        if edge_keys != self.graph.edges:
            raise PydanticCustomError(
                "polytopal_complex.adjacency_graph_binding",
                "labelled facet rows must match the graph edges exactly",
            )
        if any(
            edge.left_cell_id not in point_maps or edge.right_cell_id not in point_maps
            for edge in self.facet_edges
        ):
            raise PydanticCustomError(
                "polytopal_complex.adjacency_edge_endpoint",
                "every adjacency edge endpoint must name a graph cell",
            )
        facet_ids = tuple(edge.facet.face_id for edge in self.facet_edges)
        if len(set(facet_ids)) != len(facet_ids):
            raise PydanticCustomError(
                "polytopal_complex.adjacency_facet_unique",
                "each shared facet may label at most one adjacency edge",
            )
        for edge in self.facet_edges:
            _validate_facet_edge(edge, point_maps, self.dimension)
        return self


__all__ = [
    "PolytopalAdjacencyRequest",
    "PolytopalAdjacencyCell",
    "PolytopalComplexAdjacencyGraph",
    "PolytopalFacetAdjacency",
]
