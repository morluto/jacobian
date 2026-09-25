"""Exact maximal-cell facet adjacency projections."""

from fractions import Fraction
from itertools import combinations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes.adjacency.operations import (
    polytopal_complex_adjacency_graph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _cell(points: tuple[tuple[int, int], ...]) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=label,
                coordinates=tuple(
                    CanonicalRational.from_fraction(Fraction(value)) for value in point
                ),
            )
            for label, point in zip(("a", "b", "c"), points, strict=True)
        ),
    )


def test_adjacency_uses_shared_facets_and_not_vertex_contacts() -> None:
    cells = (
        _cell(((0, 0), (1, 0), (0, 1))),
        _cell(((1, 0), (1, 1), (0, 1))),
        _cell(((1, 1), (2, 1), (2, 2))),
    )
    result = polytopal_complex_adjacency_graph(cells)

    cell_points = {
        cell.cell_id: {
            tuple(coord.as_integer_ratio() for coord in p.coordinates)
            for p in cell.vertices
        }
        for cell in result.cells
    }
    oracle_edges = {
        pair
        for pair in combinations(sorted(cell_points), 2)
        if len(cell_points[pair[0]] & cell_points[pair[1]]) == 2
    }
    assert result.graph.edges == tuple(sorted(oracle_edges))
    assert len(result.graph.edges) == 1
    assert len(result.facet_edges) == 1
    shared = result.facet_edges[0]
    intersection = cell_points[shared.left_cell_id] & cell_points[shared.right_cell_id]
    facet_points = {
        tuple(coord.as_integer_ratio() for coord in point.coordinates)
        for point in shared.facet.vertices
    }
    assert shared.facet.dimension == 1
    assert facet_points == intersection
    assert (
        SimpleUndirectedGraph.model_validate_json(result.graph.model_dump_json())
        == result.graph
    )


def test_adjacency_admits_only_bounded_cell_families() -> None:
    triangle = _cell(((0, 0), (1, 0), (0, 1)))
    cells = (triangle,) * 17
    with pytest.raises(
        OperationResourceAdmissionError, match="16-cell adjacency envelope"
    ):
        polytopal_complex_adjacency_graph(cells)
