"""Tests for the exact rational-polytope pyramid operation."""

from __future__ import annotations

from fractions import Fraction

import pytest
from sympy import Matrix, Rational

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
from jacobian.math.geometry.polytopes._models import PyramidRequest
from jacobian.math.geometry.polytopes._tools import TOOLS, compute_polytope_pyramid
from jacobian.math.geometry.polytopes.operations import (
    facet_incidence,
    polytope_pyramid,
)
from jacobian.math.geometry.polytopes.values import (
    MAX_RATIONAL_POLYTOPE_DIMENSION,
    Vertex,
)


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _vertex(vertex_id: str, *coordinates: int) -> RationalPolytopeVertex:
    return RationalPolytopeVertex(
        vertex_id=vertex_id,
        coordinates=tuple(_rational(value) for value in coordinates),
    )


def _polytope(
    axes: tuple[str, ...], rows: tuple[tuple[str, tuple[int, ...]], ...]
) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=axes),
        vertices=tuple(_vertex(vid, *coords) for vid, coords in rows),
    )


def _segment() -> RationalVPolytope:
    return _polytope(("x",), (("left", (0,)), ("right", (1,))))


def _square() -> RationalVPolytope:
    return _polytope(
        ("x", "y"),
        (
            ("bottom_left", (0, 0)),
            ("bottom_right", (1, 0)),
            ("top_left", (0, 1)),
            ("top_right", (1, 1)),
        ),
    )


def _coords(polytope: RationalVPolytope) -> dict[str, tuple[Fraction, ...]]:
    return {
        vertex.vertex_id: tuple(c.as_fraction() for c in vertex.coordinates)
        for vertex in polytope.vertices
    }


def _extreme_ids(vertices: tuple[Vertex, ...], dim: int) -> set[int]:
    """Exact extreme-vertex positions via the active-facet-normal rank test."""
    profile = facet_incidence(vertices, dim)
    normals = [
        [Rational(*c.as_integer_ratio()) for c in facet.halfspace.coefficients]
        for facet in profile.facets
    ]
    extreme: set[int] = set()
    for index in range(len(vertices)):
        active = [
            normals[k]
            for k, facet in enumerate(profile.facets)
            if index in facet.source_vertex_indices
        ]
        if active and Matrix(active).rank() == dim:
            extreme.add(index)
    return extreme


class TestKnownAnswer:
    def test_segment_pyramid_is_triangle(self) -> None:
        result = polytope_pyramid(_segment(), "h")
        assert result.source_affine_dimension == 1
        assert result.pyramid_affine_dimension == 2
        assert result.apex_vertex_id == "apex"
        coords = _coords(result.pyramid)
        assert coords == {
            "apex": (Fraction(0), Fraction(1)),
            "left": (Fraction(0), Fraction(0)),
            "right": (Fraction(1), Fraction(0)),
        }
        assert tuple(result.pyramid.space.axes) == ("x", "h")
        assert [
            (row.source_vertex_id, row.pyramid_vertex_id)
            for row in result.base_vertex_map
        ] == [
            ("left", "left"),
            ("right", "right"),
        ]

    def test_square_pyramid_has_five_vertices(self) -> None:
        result = polytope_pyramid(_square(), "h")
        assert result.source_affine_dimension == 2
        assert result.pyramid_affine_dimension == 3
        coords = _coords(result.pyramid)
        assert coords["apex"] == (Fraction(0), Fraction(0), Fraction(1))
        assert len(coords) == 5
        for vid in ("bottom_left", "bottom_right", "top_left", "top_right"):
            assert coords[vid][2] == 0


class TestBoundary:
    def test_redundant_source_row_keeps_dimension_identity(self) -> None:
        # NOTE: the value allows redundant boundary rows with distinct
        # coordinates; the pyramid still gains exactly one dimension.
        tri = _polytope(
            ("x", "y"),
            (("a", (0, 0)), ("b", (2, 0)), ("c", (1, 0)), ("d", (0, 1))),
        )
        result = polytope_pyramid(tri, "h")
        assert result.pyramid_affine_dimension == result.source_affine_dimension + 1
        assert len(result.pyramid.vertices) == 5

    def test_height_axis_at_dimension_envelope_boundary(self) -> None:
        axes = tuple(f"x{i}" for i in range(MAX_RATIONAL_POLYTOPE_DIMENSION - 1))
        rows = tuple(
            (f"v{i}", tuple(1 if j == i else 0 for j in range(len(axes))))
            for i in range(len(axes))
        )
        origin = ("origin", tuple(0 for _ in axes))
        polytope = _polytope(axes, (origin, *rows))
        result = polytope_pyramid(polytope, "h")
        assert len(result.pyramid.space.axes) == MAX_RATIONAL_POLYTOPE_DIMENSION


class TestAdversarial:
    def test_fresh_height_axis_required(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_pyramid(_segment(), "x")

    def test_reserved_apex_label_rejected(self) -> None:
        taken = _polytope(("x",), (("apex", (0,)), ("right", (1,))))
        with pytest.raises(OperationDomainValidationError):
            polytope_pyramid(taken, "h")

    def test_native_rejects_non_polytope(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_pyramid("not-a-polytope", "h")  # type: ignore[arg-type]

    def test_over_dimension_envelope_refused(self) -> None:
        axes = tuple(f"x{i}" for i in range(MAX_RATIONAL_POLYTOPE_DIMENSION))
        rows = tuple(
            (f"v{i}", tuple(1 if j == i else 0 for j in range(len(axes))))
            for i in range(len(axes))
        )
        origin = ("origin", tuple(0 for _ in axes))
        polytope = _polytope(axes, (origin, *rows))
        with pytest.raises(OperationResourceAdmissionError):
            polytope_pyramid(polytope, "h")


class TestDefiningInvariant:
    def test_base_face_exposure_and_apex_uniqueness(self) -> None:
        result = polytope_pyramid(_square(), "h")
        coords = _coords(result.pyramid)
        heights = {vid: c[-1] for vid, c in coords.items()}
        assert heights["apex"] == 1
        assert all(h == 0 for vid, h in heights.items() if vid != "apex")

    @pytest.mark.parametrize("source", [_segment(), _square()])
    def test_source_extreme_vertices_stay_extreme(
        self, source: RationalVPolytope
    ) -> None:
        result = polytope_pyramid(source, "h")
        bare = tuple(
            Vertex(coordinates=vertex.coordinates) for vertex in result.pyramid.vertices
        )
        extreme = _extreme_ids(bare, result.pyramid_affine_dimension)
        assert len(extreme) == len(result.pyramid.vertices)
        base_positions = {
            index
            for index, vertex in enumerate(result.pyramid.vertices)
            if vertex.vertex_id != "apex"
        }
        assert base_positions <= extreme


class TestNativeVsCatalogParity:
    def test_catalog_entry_matches_native(self) -> None:
        request = PyramidRequest(polytope=_square(), height_axis="h")
        assert compute_polytope_pyramid(request) == polytope_pyramid(_square(), "h")

    def test_operation_is_published(self) -> None:
        assert "polytope.rational.pyramid.compute" in {
            tool.operation_id for tool in TOOLS
        }
