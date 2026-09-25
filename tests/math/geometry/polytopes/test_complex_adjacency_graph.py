"""Exact maximal-cell facet adjacency projections."""

import json
from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes.adjacency._models import (
    PolytopalComplexAdjacencyGraph,
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


def test_multi_digit_cell_ids_keep_numeric_vertices_and_lexical_graph_edges() -> None:
    cells = (
        *(
            _cell(((3 * index, 0), (3 * index + 1, 0), (3 * index, 1)))
            for index in range(10)
        ),
        _cell(((28, 0), (28, 1), (27, 1))),
    )
    result = polytopal_complex_adjacency_graph(cells)

    assert result.graph.vertices == tuple(f"M{index}" for index in range(11))
    assert result.graph.edges == (("M10", "M9"),)
    assert len(result.facet_edges) == 1
    assert result.facet_edges[0].facet.maximal_cell_ids == ("M10", "M9")
    assert (
        PolytopalComplexAdjacencyGraph.model_validate_json(result.model_dump_json())
        == result
    )


def test_result_json_rejects_forged_nonfacet_with_correct_declared_dimension() -> None:
    result = polytopal_complex_adjacency_graph(
        (
            _cell(((0, 0), (1, 0), (0, 1))),
            _cell(((1, 0), (1, 1), (0, 1))),
        )
    )
    payload = json.loads(result.model_dump_json())
    facet = payload["facet_edges"][0]["facet"]
    facet["vertices"] = [facet["vertices"][0]]

    with pytest.raises(ValidationError, match="exactly the common cell vertices"):
        PolytopalComplexAdjacencyGraph.model_validate_json(json.dumps(payload))
