from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import (
    polytopal_complex_affine_transform as exported_affine_transform,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    PolytopalComplexAffineTransformRequest,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_affine_transform,
    polytopal_complex_closure,
)

SPACE = RationalCoordinateSpace(axes=("x", "y"))


def _poly(points: tuple[tuple[int, int], ...], prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=SPACE,
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=tuple({"num": x, "den": 1} for x in point),
            )
            for index, point in enumerate(points)
        ),
    )


def _fraction_point(point) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(*coordinate.as_integer_ratio()) for coordinate in point.coordinates
    )


def _request(complex_value, matrix, translation):
    return PolytopalComplexAffineTransformRequest(
        complex=complex_value,
        matrix=tuple(
            tuple(CanonicalRational(num=value, den=1) for value in row)
            for row in matrix
        ),
        translation=tuple(CanonicalRational(num=value, den=1) for value in translation),
    )


def test_unimodular_shear_transports_square_triangulation_and_incidence():
    source = polytopal_complex_closure(
        (
            _poly(((0, 0), (1, 0), (1, 1)), "a"),
            _poly(((0, 0), (1, 1), (0, 1)), "b"),
        )
    )
    result = polytopal_complex_affine_transform(
        _request(source, ((1, 1), (0, 1)), (2, -1))
    )

    assert result.target.f_vector == source.f_vector == (1, 4, 5, 2)
    assert result.target.dimension == source.dimension == 2
    expected_vertices = {
        (Fraction(2), Fraction(-1)),
        (Fraction(3), Fraction(-1)),
        (Fraction(4), Fraction(0)),
        (Fraction(3), Fraction(0)),
    }
    assert {
        _fraction_point(point)
        for face in result.target.faces
        if face.dimension == 0
        for point in face.vertices
    } == expected_vertices

    face_map = {row.source_face_id: row.target_face_id for row in result.face_transport}
    cell_map = {row.source_cell_id: row.target_cell_id for row in result.cell_transport}
    source_faces = {face.face_id: face for face in source.faces}
    target_faces = {face.face_id: face for face in result.target.faces}
    for source_face_id, target_face_id in face_map.items():
        source_face = source_faces[source_face_id]
        target_face = target_faces[target_face_id]
        transformed = {
            (point[0] + point[1] + 2, point[1] - 1)
            for point in map(_fraction_point, source_face.vertices)
        }
        assert {_fraction_point(point) for point in target_face.vertices} == transformed
        assert target_face.dimension == source_face.dimension
        assert set(target_face.maximal_cell_ids) == {
            cell_map[cell_id] for cell_id in source_face.maximal_cell_ids
        }

    assert {
        (face_map[row.lower_face_id], face_map[row.upper_face_id])
        for row in source.cover_relations
    } == {
        (row.lower_face_id, row.upper_face_id) for row in result.target.cover_relations
    }
    for source_cell in source.maximal_cells:
        target_cell_id = cell_map[source_cell.cell_id]
        target_cell = next(
            cell
            for cell in result.target.maximal_cells
            if cell.cell_id == target_cell_id
        )
        assert {face_map[face_id] for face_id in source_cell.facet_face_ids} == set(
            target_cell.facet_face_ids
        )


def test_affine_transform_requires_nonsingular_matrix():
    source = polytopal_complex_closure((_poly(((0, 0), (1, 0), (0, 1)), "t"),))
    with pytest.raises(OperationDomainValidationError, match="invertible matrix"):
        polytopal_complex_affine_transform(_request(source, ((1, 2), (2, 4)), (0, 0)))


def test_affine_transform_admits_matrix_height_before_geometry_work():
    source = polytopal_complex_closure((_poly(((0, 0), (1, 0), (0, 1)), "t"),))
    request = _request(source, ((10**32, 0), (0, 1)), (0, 0))
    with pytest.raises(
        OperationResourceAdmissionError, match="32-digit input envelope"
    ):
        polytopal_complex_affine_transform(request)


def test_affine_transform_admits_result_coordinate_growth_before_target_closure():
    base = 10**31
    source = polytopal_complex_closure(
        (_poly(((base, 0), (base + 1, 0), (base, 1)), "large"),)
    )
    request = _request(source, ((base, 0), (0, 1)), (0, 0))
    with pytest.raises(
        OperationResourceAdmissionError, match="transformed coordinate exceeds"
    ):
        polytopal_complex_affine_transform(request)


def test_affine_transform_catalog_example_roundtrips():
    catalog = Catalog.open()
    operation = catalog.operation("polytopal_complex.affine_transform.compute")
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    target = result.output["target"]
    assert target["maximal_cells"][0]["vertices"] == [
        {"coordinates": [{"num": "3", "den": "1"}]},
        {"coordinates": [{"num": "5", "den": "1"}]},
    ]
    assert len(result.output["cell_transport"]) == 1
    assert exported_affine_transform is polytopal_complex_affine_transform
