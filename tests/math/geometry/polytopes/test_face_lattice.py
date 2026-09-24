from __future__ import annotations

from collections import Counter
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes import (
    PolytopeFaceLatticeResult,
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
    polytope_face_lattice,
)
from jacobian.math.geometry.polytopes import operations as polytope_operations

_OPERATION_ID = "polytope.face_lattice.compute"


def _rational(value: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational(num=value, den=denominator)


def _polytope(
    rows: tuple[tuple[str, tuple[tuple[int, int], ...]], ...],
) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y", "z")),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=vertex_id,
                coordinates=tuple(_rational(*coordinate) for coordinate in coordinates),
            )
            for vertex_id, coordinates in sorted(rows)
        ),
    )


def _tetrahedron() -> RationalVPolytope:
    return _polytope(
        (
            ("ex", ((1, 1), (0, 1), (0, 1))),
            ("ey", ((0, 1), (1, 1), (0, 1))),
            ("ez", ((0, 1), (0, 1), (1, 1))),
            ("origin", ((0, 1), (0, 1), (0, 1))),
        )
    )


def _cube() -> RationalVPolytope:
    return _polytope(
        tuple(
            (f"v{x}{y}{z}", ((x, 1), (y, 1), (z, 1)))
            for x, y, z in product((0, 1), repeat=3)
        )
    )


def _square_pyramid() -> RationalVPolytope:
    return _polytope(
        (
            ("apex", ((0, 1), (0, 1), (1, 1))),
            ("v00", ((0, 1), (0, 1), (0, 1))),
            ("v01", ((0, 1), (1, 1), (0, 1))),
            ("v10", ((1, 1), (0, 1), (0, 1))),
            ("v11", ((1, 1), (1, 1), (0, 1))),
        )
    )


def _f_vector(result: PolytopeFaceLatticeResult) -> tuple[int, int, int, int, int]:
    counts = Counter(face.dimension for face in result.faces)
    return tuple(counts[index] for index in (-1, 0, 1, 2, 3))


def test_tetrahedron_has_complete_canonical_face_lattice() -> None:
    result = polytope_face_lattice(_tetrahedron())

    assert _f_vector(result) == (1, 4, 6, 4, 1)
    assert len(result.covers) == 32
    assert result.extreme_vertex_indices == (0, 1, 2, 3)
    assert result.faces == tuple(
        sorted(
            result.faces, key=lambda face: (face.dimension, face.source_vertex_indices)
        )
    )
    assert tuple(
        (cover.lower_face_index, cover.upper_face_index) for cover in result.covers
    ) == tuple(
        sorted(
            (cover.lower_face_index, cover.upper_face_index) for cover in result.covers
        )
    )
    restored = PolytopeFaceLatticeResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_cube_and_non_simple_square_pyramid_face_lattices() -> None:
    cube = polytope_face_lattice(_cube())
    assert _f_vector(cube) == (1, 8, 12, 6, 1)
    assert len(cube.covers) == 62

    pyramid = polytope_face_lattice(_square_pyramid())
    assert _f_vector(pyramid) == (1, 5, 8, 5, 1)
    assert len(pyramid.covers) == 42
    apex_index = next(
        index
        for index, vertex in enumerate(pyramid.polytope.vertices)
        if vertex.vertex_id == "apex"
    )
    apex_face_index = next(
        index
        for index, face in enumerate(pyramid.faces)
        if face.dimension == 0 and face.source_vertex_indices == (apex_index,)
    )
    assert (
        sum(cover.lower_face_index == apex_face_index for cover in pyramid.covers) == 4
    )


def test_redundant_source_rows_are_retained_but_not_called_faces() -> None:
    cube = _cube()
    with_redundant_center = RationalVPolytope(
        space=cube.space,
        vertices=tuple(
            sorted(
                (
                    *cube.vertices,
                    RationalPolytopeVertex(
                        vertex_id="center",
                        coordinates=tuple(_rational(1, 2) for _ in range(3)),
                    ),
                ),
                key=lambda vertex: vertex.vertex_id,
            )
        ),
    )
    result = polytope_face_lattice(with_redundant_center)
    center_index = next(
        index
        for index, vertex in enumerate(result.polytope.vertices)
        if vertex.vertex_id == "center"
    )
    assert center_index not in result.extreme_vertex_indices
    assert all(center_index not in face.source_vertex_indices for face in result.faces)
    assert len(result.polytope.vertices) == 9


def test_face_lattice_revalidates_source_and_rejects_bad_dimensions() -> None:
    valid = _tetrahedron()
    forged = RationalVPolytope.model_construct(
        space=valid.space,
        vertices=(valid.vertices[0], valid.vertices[0], *valid.vertices[2:]),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        polytope_face_lattice(forged)
    assert (
        error.value.errors()[0]["type"]
        == "polytope.face_lattice.source_structure_invalid"
    )

    plane = _polytope(
        (
            ("a", ((0, 1), (0, 1), (0, 1))),
            ("b", ((1, 1), (0, 1), (0, 1))),
            ("c", ((0, 1), (1, 1), (0, 1))),
            ("d", ((1, 1), (1, 1), (0, 1))),
        )
    )
    with pytest.raises(OperationDomainValidationError) as error:
        polytope_face_lattice(plane)
    assert (
        error.value.errors()[0]["type"] == "polytope.face_lattice.facet_profile_invalid"
    )

    four_dimensional = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y", "z", "w")),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"v{index}",
                coordinates=tuple(_rational(int(index == axis)) for axis in range(4)),
            )
            for index in range(5)
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        polytope_face_lattice(four_dimensional)
    assert (
        error.value.errors()[0]["type"] == "polytope.face_lattice.dimension_not_three"
    )


def test_face_lattice_admits_postprocessing_before_facet_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(polytope_operations, "MAX_POLYTOPE_FACE_LATTICE_WORK", 1)
    with pytest.raises(OperationResourceAdmissionError) as error:
        polytope_face_lattice(_cube())
    assert (
        error.value.errors()[0]["type"] == "polytope.face_lattice.work_budget_exceeded"
    )


def test_face_lattice_operation_is_catalogued_and_runs_its_example() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(_OPERATION_ID)
    assert operation is not None
    result = invoke_operation(_OPERATION_ID, operation.examples[0].input, catalog)
    assert result.output["faces"]
    assert len(result.output["covers"]) == 32
