"""Exact Freudenthal triangulation into the canonical simplicial value."""

import json

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.cubical_complexes._models import CubicalCell
from jacobian.math.topology.cubical_complexes.extensions import (
    CubicalTriangulationRequest,
    CubicalTriangulationResult,
    triangulate,
)


def _cell(*intervals: tuple[int, int]) -> CubicalCell:
    return CubicalCell(intervals=intervals)


def test_square_and_cube_have_the_expected_freudenthal_simplices() -> None:
    square = triangulate(CubicalTriangulationRequest(cells=(_cell((0, 1), (0, 1)),)))
    assert square.simplicial_complex.f_vector == (4, 5, 2)
    assert len(square.cell_maps) == 1
    assert len(square.cell_maps[0].simplices) == 2
    assert square.cell_maps[0].source_cell == _cell((0, 1), (0, 1))
    assert tuple(entry.coordinate for entry in square.vertex_map) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    square_vertices = {entry.vertex: entry.coordinate for entry in square.vertex_map}
    square_edges = square.simplicial_complex.faces_by_dimension[1].faces
    diagonal = tuple(
        sorted(
            vertex
            for vertex, point in square_vertices.items()
            if point in ((0, 0), (1, 1))
        )
    )
    assert diagonal in square_edges

    cube = triangulate(
        CubicalTriangulationRequest(cells=(_cell((0, 1), (0, 1), (0, 1)),))
    )
    assert cube.simplicial_complex.f_vector == (8, 19, 18, 6)
    assert len(cube.cell_maps[0].simplices) == 6


def test_adjacent_cubes_have_one_identical_shared_face_triangulation() -> None:
    left = _cell((0, 1), (0, 1), (0, 1))
    right = _cell((1, 2), (0, 1), (0, 1))
    result = triangulate(CubicalTriangulationRequest(cells=(right, left)))
    coordinates = {entry.vertex: entry.coordinate for entry in result.vertex_map}
    edge_faces = result.simplicial_complex.faces_by_dimension[1].faces
    shared_diagonal = tuple(
        sorted(
            vertex
            for vertex, point in coordinates.items()
            if point in ((1, 0, 0), (1, 1, 1))
        )
    )
    assert len(shared_diagonal) == 2
    assert shared_diagonal in edge_faces
    assert tuple(mapping.source_cell for mapping in result.cell_maps) == (left, right)
    assert all(
        any(set(shared_diagonal).issubset(simplex) for simplex in mapping.simplices)
        for mapping in result.cell_maps
    )


def test_serialized_transport_rejects_missing_or_forged_source_maps() -> None:
    result = triangulate(
        CubicalTriangulationRequest(
            cells=(
                _cell((0, 1), (0, 1), (0, 1)),
                _cell((1, 2), (0, 1), (0, 1)),
            )
        )
    )
    result.require_valid_transport()
    payload = result.model_dump(mode="json")
    payload["cell_maps"].pop()
    decoded = CubicalTriangulationResult.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="exact maximal source presentation"):
        decoded.require_valid_transport()

    payload = result.model_dump(mode="json")
    payload["cell_maps"][0]["simplices"][0] = (
        result.simplicial_complex.maximal_simplices[-1]
    )
    decoded = CubicalTriangulationResult.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="Freudenthal subdivision"):
        decoded.require_valid_transport()

    payload = result.model_dump(mode="json")
    payload["simplicial_complex"]["faces_by_dimension"][1]["faces"].pop()
    payload["simplicial_complex"]["f_vector"][1] -= 1
    payload["simplicial_complex"]["closure_size"] -= 1
    decoded = CubicalTriangulationResult.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="closure"):
        decoded.require_valid_transport()

    payload = result.model_dump(mode="json")
    payload["vertex_map"][-1]["coordinate"] = [3, 3, 3]
    decoded = CubicalTriangulationResult.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="exact maximal source presentation"):
        decoded.require_valid_transport()


def test_vertex_bound_is_accepted_at_and_rejected_above_64() -> None:
    exact = tuple(_cell((point, point)) for point in range(64))
    result = triangulate(CubicalTriangulationRequest(cells=exact))
    assert len(result.simplicial_complex.vertices) == 64
    above = (*exact, _cell((64, 64)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        triangulate(CubicalTriangulationRequest(cells=above))
    assert (
        error.value.errors()[0]["type"] == "cubical_complex.triangulation_point_budget"
    )


def test_facet_factorial_is_admitted_before_generating_a_six_cube() -> None:
    request = CubicalTriangulationRequest(
        cells=(_cell((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        triangulate(request)
    assert (
        error.value.errors()[0]["type"] == "cubical_complex.triangulation_output_budget"
    )
