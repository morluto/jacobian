"""Tests for the exact rational polytopal-complex closure operation."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import _models
from jacobian.math.geometry.polytopes.complexes._models import (
    PolytopalComplexClosureRequest,
    PolytopalComplexClosureResult,
)
from jacobian.math.geometry.polytopes.complexes._tools import (
    TOOLS,
    compute_polytopal_complex_closure,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.geometry.polytopes.values import Vertex

OPERATION_ID = "polytopal_complex.closure.compute"


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _vertex(vertex_id: str, *coordinates: int) -> RationalPolytopeVertex:
    return RationalPolytopeVertex(
        vertex_id=vertex_id,
        coordinates=tuple(_rational(value) for value in coordinates),
    )


def _cells(
    axes: tuple[str, ...], rows: tuple[tuple[str, tuple[int, ...]], ...]
) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=axes),
        vertices=tuple(_vertex(vertex_id, *coords) for vertex_id, coords in rows),
    )


def _simplex(axes: tuple[str, ...], origin: int) -> RationalVPolytope:
    rows = [("o", tuple(origin if axis == 0 else 0 for axis in range(len(axes))))]
    for index in range(len(axes)):
        coordinates = [origin if axis == 0 else 0 for axis in range(len(axes))]
        coordinates[index] += 1
        rows.append((f"v{index}", tuple(coordinates)))
    ordered = tuple(sorted(rows))
    return _cells(axes, ordered)


def _unit_4cube(shift: int) -> RationalVPolytope:
    axes = ("a", "b", "c", "d")
    rows = []
    for index in range(16):
        bits = [1 if digit == "1" else 0 for digit in format(index, "04b")]
        coordinates = [
            (shift if axis == 0 else 0) + bits[axis] for axis in range(len(axes))
        ]
        rows.append((f"v{index:02d}", tuple(coordinates)))
    return _cells(axes, tuple(rows))


def _face_coordinates(
    result: PolytopalComplexClosureResult,
) -> set[frozenset[tuple[Fraction, ...]]]:
    return {
        frozenset(
            tuple(coordinate.as_fraction() for coordinate in point.coordinates)
            for point in face.vertices
        )
        for face in result.faces
    }


class TestKnownAnswer:
    def test_triangle_simplex_face_lattice(self) -> None:
        triangle = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        result = polytopal_complex_closure((triangle,))
        assert result.f_vector == (1, 3, 3, 1)
        assert result.euler_characteristic == 1
        assert result.reduced_euler_characteristic == 0
        assert result.component_count == 1

    def test_tetrahedron_f_vector(self) -> None:
        tetra = _simplex(("x", "y", "z"), 0)
        result = polytopal_complex_closure((tetra,))
        assert result.f_vector == (1, 4, 6, 4, 1)
        assert result.euler_characteristic == 1

    def test_facet_incidence_face_count_matches(self) -> None:
        square = _cells(
            ("x", "y"),
            (("a", (0, 0)), ("b", (1, 0)), ("c", (1, 1)), ("d", (0, 1))),
        )
        result = polytopal_complex_closure((square,))
        profile = facet_incidence(
            tuple(Vertex(coordinates=v.coordinates) for v in square.vertices), 2
        )
        assert len(result.maximal_cells) == 1
        assert len(result.maximal_cells[0].facet_face_ids) == len(profile.facets)
        assert result.f_vector == (1, 4, 4, 1)

    def test_two_triangles_share_edge_identified_once(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (1, 0)), ("e", (1, 1)), ("f", (0, 1))))
        result = polytopal_complex_closure((first, second))
        assert result.f_vector == (1, 4, 5, 2)
        assert result.euler_characteristic == 1
        diagonal = frozenset({(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))})
        shared = [
            face
            for face in result.faces
            if face.dimension == 1
            and frozenset(
                tuple(c.as_fraction() for c in point.coordinates)
                for point in face.vertices
            )
            == diagonal
        ]
        assert len(shared) == 1
        assert shared[0].maximal_cell_ids == ("M0", "M1")

    def test_disjoint_cells_are_disconnected(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (10, 10)), ("e", (11, 10)), ("f", (10, 11))))
        result = polytopal_complex_closure((first, second))
        assert result.component_count == 2
        assert [row.status for row in result.pairwise_intersections] == ["empty"]
        assert result.f_vector == (1, 6, 6, 2)


class TestBoundary:
    def test_single_segment(self) -> None:
        segment = _cells(("x",), (("a", (0,)), ("b", (1,))))
        result = polytopal_complex_closure((segment,))
        assert result.dimension == 1
        assert result.f_vector == (1, 2, 1)
        assert result.component_count == 1

    def test_empty_face_is_admitted_once(self) -> None:
        segment = _cells(("x",), (("a", (0,)), ("b", (1,))))
        result = polytopal_complex_closure((segment,))
        empty = [face for face in result.faces if face.dimension == -1]
        assert len(empty) == 1
        assert empty[0].vertices == ()
        assert result.empty_face_admitted is True
        assert empty[0].maximal_cell_ids == ("M0",)

    def test_redundant_interior_vertex_is_dropped(self) -> None:
        triangle = _cells(
            ("x", "y"),
            (("a", (0, 0)), ("b", (2, 0)), ("c", (0, 1)), ("m", (1, 0))),
        )
        result = polytopal_complex_closure((triangle,))
        assert result.f_vector == (1, 3, 3, 1)
        assert len(result.maximal_cells[0].vertices) == 3

    def test_duplicate_maximal_cells_merge_with_provenance(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (0, 0)), ("e", (1, 0)), ("f", (0, 1))))
        result = polytopal_complex_closure((first, second))
        assert len(result.maximal_cells) == 1
        assert result.maximal_cells[0].source_indices == (0, 1)
        assert result.pairwise_intersections == ()
        assert result.f_vector == (1, 3, 3, 1)


class TestAdversarial:
    def test_non_face_intersection_is_typed_obstruction(self) -> None:
        first = _cells(
            ("x", "y"),
            (("a", (0, 0)), ("b", (2, 0)), ("c", (2, 2)), ("d", (0, 2))),
        )
        second = _cells(
            ("x", "y"),
            (("e", (1, 0)), ("f", (3, 0)), ("g", (3, 2)), ("h", (1, 2))),
        )
        with pytest.raises(OperationDomainValidationError) as caught:
            polytopal_complex_closure((first, second))
        error = caught.value.errors()[0]
        assert error["type"] == "polytopal_complex.non_face_intersection"
        assert error["loc"] == ("cells", 0, 1)

    def test_mixed_ambient_spaces_rejected(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("y", "x"), (("d", (0, 0)), ("e", (0, 1)), ("f", (1, 0))))
        with pytest.raises(OperationDomainValidationError) as caught:
            polytopal_complex_closure((first, second))
        assert caught.value.errors()[0]["type"] == (
            "polytopal_complex.mixed_ambient_spaces"
        )

    def test_lower_dimensional_maximal_cell_rejected(self) -> None:
        collinear = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (2, 0))))
        with pytest.raises(OperationDomainValidationError):
            polytopal_complex_closure((collinear,))

    def test_dimension_envelope_rejected(self) -> None:
        simplex = _simplex(tuple(f"x{index}" for index in range(5)), 0)
        with pytest.raises(OperationResourceAdmissionError) as caught:
            polytopal_complex_closure((simplex,))
        assert caught.value.errors()[0]["type"] == (
            "polytopal_complex.dimension_over_envelope"
        )

    def test_cell_count_envelope_rejected(self) -> None:
        cells = tuple(
            _cells(("x",), (("a", (2 * index,)), ("b", (2 * index + 1,))))
            for index in range(_models.MAX_COMPLEX_CELLS + 1)
        )
        with pytest.raises(OperationResourceAdmissionError) as caught:
            polytopal_complex_closure(cells)
        assert caught.value.errors()[0]["type"] == (
            "polytopal_complex.cell_count_over_envelope"
        )

    def test_coordinate_digit_envelope_rejected(self) -> None:
        huge = 10 ** (_models.MAX_COMPLEX_COORDINATE_DIGITS + 1)
        segment = _cells(("x",), (("a", (0,)), ("b", (huge,))))
        with pytest.raises(OperationResourceAdmissionError) as caught:
            polytopal_complex_closure((segment,))
        assert caught.value.errors()[0]["type"] == (
            "polytopal_complex.coordinate_digits_over_envelope"
        )

    def test_face_closure_envelope_rejected(self) -> None:
        cubes = tuple(_unit_4cube(10 * index) for index in range(13))
        with pytest.raises(OperationResourceAdmissionError) as caught:
            polytopal_complex_closure(cubes)
        assert caught.value.errors()[0]["type"] == (
            "polytopal_complex.face_closure_over_envelope"
        )


class TestDefiningInvariant:
    def test_face_closure_and_cover_relations_are_codimension_one(self) -> None:
        square = _cells(
            ("x", "y"),
            (("a", (0, 0)), ("b", (1, 0)), ("c", (1, 1)), ("d", (0, 1))),
        )
        result = polytopal_complex_closure((square,))
        dimension_of = {face.face_id: face.dimension for face in result.faces}
        for relation in result.cover_relations:
            assert (
                dimension_of[relation.upper_face_id]
                == dimension_of[relation.lower_face_id] + 1
            )

    def test_euler_alternating_sums(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (1, 0)), ("e", (1, 1)), ("f", (0, 1))))
        result = polytopal_complex_closure((first, second))
        ordinary = sum(
            (-1) ** level * result.f_vector[level + 1]
            for level in range(result.dimension + 1)
        )
        reduced = -1 + ordinary
        assert result.euler_characteristic == ordinary
        assert result.reduced_euler_characteristic == reduced

    def test_pairwise_symmetry_and_transport(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (1, 0)), ("e", (1, 1)), ("f", (0, 1))))
        result = polytopal_complex_closure((first, second))
        face_by_id = {face.face_id: face for face in result.faces}
        for row in result.pairwise_intersections:
            if row.status != "face":
                continue
            intersection_face_id = row.intersection_face_id
            assert intersection_face_id is not None
            assert (
                row.first_cell_id
                in face_by_id[intersection_face_id].maximal_cell_ids
            )
            assert (
                row.second_cell_id
                in face_by_id[intersection_face_id].maximal_cell_ids
            )
        assert [row.source_index for row in result.source_cell_map] == [0, 1]
        assert [row.cell_id for row in result.source_cell_map] == ["M0", "M1"]

    def test_presentation_order_invariance(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (1, 0)), ("e", (1, 1)), ("f", (0, 1))))
        forward = polytopal_complex_closure((first, second))
        backward = polytopal_complex_closure((second, first))
        assert forward.f_vector == backward.f_vector
        assert _face_coordinates(forward) == _face_coordinates(backward)


class TestNativeVsCatalogParity:
    def test_catalog_entry_matches_native(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (1, 0)), ("e", (1, 1)), ("f", (0, 1))))
        request = PolytopalComplexClosureRequest(cells=(first, second))
        assert compute_polytopal_complex_closure(request) == polytopal_complex_closure(
            (first, second)
        )

    def test_operation_is_published(self) -> None:
        assert OPERATION_ID in {tool.operation_id for tool in TOOLS}

    def test_catalog_example_runs(self) -> None:
        example = TOOLS[0].examples[0]
        request = PolytopalComplexClosureRequest.model_validate_json(
            json.dumps(example.input)
        )
        result = compute_polytopal_complex_closure(request)
        assert result.f_vector == (1, 4, 5, 2)


class TestSerializationRoundTrip:
    def test_result_json_round_trip(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        second = _cells(("x", "y"), (("d", (1, 0)), ("e", (1, 1)), ("f", (0, 1))))
        result = polytopal_complex_closure((first, second))
        restored = PolytopalComplexClosureResult.model_validate_json(
            result.model_dump_json()
        )
        assert restored == result

    def test_request_json_round_trip(self) -> None:
        first = _cells(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        request = PolytopalComplexClosureRequest(cells=(first,))
        restored = PolytopalComplexClosureRequest.model_validate_json(
            request.model_dump_json()
        )
        assert restored == request
