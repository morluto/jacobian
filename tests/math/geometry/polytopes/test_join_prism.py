"""Tests for the exact rational-polytope prism and join operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

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
from jacobian.math.geometry.polytopes._models import (
    JoinRequest,
    JoinResult,
    PrismRequest,
    PrismResult,
)
from jacobian.math.geometry.polytopes._tools import (
    TOOLS,
    compute_polytope_join,
    compute_polytope_prism,
)
from jacobian.math.geometry.polytopes.operations import (
    convex_hull_volume,
    polytope_join,
    polytope_prism,
)
from jacobian.math.geometry.polytopes.values import MAX_RATIONAL_POLYTOPE_DIMENSION


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
    ordered = tuple(sorted(rows, key=lambda row: row[0]))
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=axes),
        vertices=tuple(_vertex(vid, *coords) for vid, coords in ordered),
    )


def _segment(ids: tuple[str, str] = ("left", "right")) -> RationalVPolytope:
    return _polytope(("x",), ((ids[0], (0,)), (ids[1], (1,))))


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


class TestPrismKnownAnswer:
    def test_segment_prism_is_unit_square(self) -> None:
        result = polytope_prism(_segment(), "h")
        assert result.source_affine_dimension == 1
        assert result.prism_affine_dimension == 2
        assert _coords(result.prism) == {
            "left_bottom": (Fraction(0), Fraction(0)),
            "left_top": (Fraction(0), Fraction(1)),
            "right_bottom": (Fraction(1), Fraction(0)),
            "right_top": (Fraction(1), Fraction(1)),
        }
        assert tuple(result.prism.space.axes) == ("x", "h")
        assert result.height_axis == "h"
        assert [row.source_vertex_id for row in result.bottom_vertex_map] == [
            "left",
            "right",
        ]
        assert [row.source_vertex_id for row in result.top_vertex_map] == [
            "left",
            "right",
        ]
        assert all(row.side == "bottom" for row in result.bottom_vertex_map)
        assert all(row.side == "top" for row in result.top_vertex_map)
        assert PrismResult.model_validate_json(result.model_dump_json()) == result

    def test_square_prism_has_eight_vertices(self) -> None:
        result = polytope_prism(_square(), "h")
        assert result.source_affine_dimension == 2
        assert result.prism_affine_dimension == 3
        coords = _coords(result.prism)
        assert len(coords) == 8
        for vid, coord in coords.items():
            if vid.endswith("_bottom"):
                assert coord[2] == 0
            else:
                assert coord[2] == 1

    def test_named_prism_height_axis_is_bound_to_output_axis(self) -> None:
        result = polytope_prism(_segment(), "h")
        payload = result.model_dump(mode="json")
        payload["height_axis"] = "wrong"
        with pytest.raises(ValidationError):
            PrismResult.model_validate(payload)


class TestJoinKnownAnswer:
    def test_segment_segment_join_is_tetrahedron(self) -> None:
        left = _segment(("left_a", "left_b"))
        right = _polytope(("y",), (("right_a", (0,)), ("right_b", (1,))))
        result = polytope_join(left, right, "h")
        assert result.left_affine_dimension == 1
        assert result.right_affine_dimension == 1
        assert result.join_affine_dimension == 3
        assert _coords(result.join) == {
            "left_a": (Fraction(0), Fraction(0), Fraction(0)),
            "left_b": (Fraction(1), Fraction(0), Fraction(0)),
            "right_a": (Fraction(0), Fraction(0), Fraction(1)),
            "right_b": (Fraction(0), Fraction(1), Fraction(1)),
        }
        assert tuple(result.join.space.axes) == ("x", "y", "h")
        assert result.height_axis == "h"
        assert JoinResult.model_validate_json(result.model_dump_json()) == result

    def test_named_join_height_axis_is_bound_to_output_axis(self) -> None:
        result = polytope_join(
            _segment(("left_a", "left_b")),
            _polytope(("y",), (("right_a", (0,)), ("right_b", (1,)))),
            "h",
        )
        payload = result.model_dump(mode="json")
        payload["height_axis"] = "wrong"
        with pytest.raises(ValidationError):
            JoinResult.model_validate(payload)

    def test_segment_square_join_dimension_identity(self) -> None:
        left = _segment(("left_a", "left_b"))
        right = _polytope(
            ("y", "z"),
            (
                ("sq_bl", (0, 0)),
                ("sq_br", (1, 0)),
                ("sq_tl", (0, 1)),
                ("sq_tr", (1, 1)),
            ),
        )
        result = polytope_join(left, right, "h")
        assert result.left_affine_dimension == 1
        assert result.right_affine_dimension == 2
        assert result.join_affine_dimension == 4
        assert tuple(result.join.space.axes) == ("x", "y", "z", "h")


class TestVolumeOracle:
    """Independent-oracle checks: prism preserves volume, join scales it."""

    def test_prism_volume_equals_base_volume(self) -> None:
        base = _square()
        result = polytope_prism(base, "h")
        assert convex_hull_volume(base) == convex_hull_volume(result.prism)

    def test_segment_prism_volume_is_one(self) -> None:
        result = polytope_prism(_segment(), "h")
        assert convex_hull_volume(result.prism) == CanonicalRational(num=1, den=1)

    def test_join_volume_identity(self) -> None:
        # vol(P * Q) = vol(P) vol(Q) m! n! / (m+n+1)! for dim m, n.
        # Two unit segments: 1 * 1 * 1! 1! / 3! = 1/6.
        left = _segment(("left_a", "left_b"))
        right = _polytope(("y",), (("right_a", (0,)), ("right_b", (1,))))
        result = polytope_join(left, right, "h")
        assert convex_hull_volume(result.join) == CanonicalRational(num=1, den=6)

    def test_prism_composes_with_volume_consumer(self) -> None:
        # CONTRIBUTING consumer check: the accepted prism value feeds the
        # real volume consumer unchanged through serialization.
        result = polytope_prism(_segment(), "h")
        payload = result.prism.model_dump_json()
        revived = RationalVPolytope.model_validate_json(payload)
        assert revived == result.prism
        assert convex_hull_volume(revived) == CanonicalRational(num=1, den=1)


class TestBoundary:
    def test_prism_at_vertex_envelope_boundary(self) -> None:
        rows = tuple((f"v{i}", (i,)) for i in range(32))
        polytope = _polytope(("x",), rows)
        result = polytope_prism(polytope, "h")
        assert len(result.prism.vertices) == 64

    def test_prism_over_vertex_envelope_refused(self) -> None:
        rows = tuple((f"v{i}", (i,)) for i in range(33))
        polytope = _polytope(("x",), rows)
        with pytest.raises(OperationResourceAdmissionError):
            polytope_prism(polytope, "h")

    def test_join_at_dimension_envelope_boundary(self) -> None:
        left_axes = tuple(f"a{i}" for i in range(3))
        right_axes = tuple(f"b{i}" for i in range(3))
        left_rows = (
            ("origin_l", (0, 0, 0)),
            *((f"l{i}", tuple(1 if j == i else 0 for j in range(3))) for i in range(3)),
        )
        right_rows = (
            ("origin_r", (0, 0, 0)),
            *((f"r{i}", tuple(1 if j == i else 0 for j in range(3))) for i in range(3)),
        )
        left = _polytope(left_axes, left_rows)
        right = _polytope(right_axes, right_rows)
        result = polytope_join(left, right, "h")
        assert len(result.join.space.axes) == MAX_RATIONAL_POLYTOPE_DIMENSION

    def test_join_over_dimension_envelope_refused(self) -> None:
        left = _polytope(("x",), (("la", (0,)), ("lb", (1,))))
        right_axes = tuple(f"b{i}" for i in range(MAX_RATIONAL_POLYTOPE_DIMENSION - 1))
        right_rows = (
            ("origin", tuple(0 for _ in right_axes)),
            *(
                (f"r{i}", tuple(1 if j == i else 0 for j in range(len(right_axes))))
                for i in range(len(right_axes))
            ),
        )
        right = _polytope(right_axes, right_rows)
        with pytest.raises(OperationResourceAdmissionError):
            polytope_join(left, right, "h")


class TestAdversarial:
    def test_prism_fresh_height_axis_required(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_prism(_segment(), "x")

    def test_prism_transport_label_overflow_rejected(self) -> None:
        # Suffixing must respect the 64-char label bound: a 60-char source ID
        # yields a 67-char prism ID.
        long_id = "v" * 60
        taken = _polytope(("x",), ((long_id, (0,)), ("w", (1,))))
        with pytest.raises(OperationDomainValidationError):
            polytope_prism(taken, "h")

    def test_prism_native_rejects_non_polytope(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_prism("not-a-polytope", "h")  # type: ignore[arg-type]

    def test_join_overlapping_axes_rejected(self) -> None:
        left = _segment(("la", "lb"))
        right = _segment(("ra", "rb"))
        with pytest.raises(OperationDomainValidationError):
            polytope_join(left, right, "h")

    def test_join_overlapping_vertex_ids_rejected(self) -> None:
        left = _polytope(("x",), (("v0", (0,)), ("v1", (1,))))
        right = _polytope(("y",), (("v0", (0,)), ("v1", (1,))))
        with pytest.raises(OperationDomainValidationError):
            polytope_join(left, right, "h")

    def test_join_stale_height_axis_rejected(self) -> None:
        left = _polytope(("x",), (("la", (0,)), ("lb", (1,))))
        right = _polytope(("y",), (("ra", (0,)), ("rb", (1,))))
        with pytest.raises(OperationDomainValidationError):
            polytope_join(left, right, "x")

    def test_join_native_rejects_non_polytope(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_join("nope", _segment(), "h")  # type: ignore[arg-type]


class TestDefiningInvariant:
    def test_prism_height_sections_recover_source(self) -> None:
        source = _square()
        result = polytope_prism(source, "h")
        coords = _coords(result.prism)
        source_coords = _coords(source)
        for row in result.bottom_vertex_map:
            prism_coord = coords[row.prism_vertex_id]
            assert prism_coord[-1] == 0
            assert prism_coord[:-1] == source_coords[row.source_vertex_id]
        for row in result.top_vertex_map:
            prism_coord = coords[row.prism_vertex_id]
            assert prism_coord[-1] == 1
            assert prism_coord[:-1] == source_coords[row.source_vertex_id]

    def test_join_zero_blocks_and_projections(self) -> None:
        left = _polytope(("x",), (("la", (0,)), ("lb", (1,))))
        right = _polytope(("y",), (("ra", (0,)), ("rb", (1,))))
        result = polytope_join(left, right, "h")
        coords = _coords(result.join)
        left_coords = _coords(left)
        right_coords = _coords(right)
        for row in result.left_vertex_map:
            coord = coords[row.join_vertex_id]
            assert coord[1] == 0 and coord[2] == 0
            assert (coord[0],) == left_coords[row.source_vertex_id]
        for row in result.right_vertex_map:
            coord = coords[row.join_vertex_id]
            assert coord[0] == 0 and coord[2] == 1
            assert (coord[1],) == right_coords[row.source_vertex_id]

    def test_redundant_source_row_keeps_dimension_identity(self) -> None:
        tri = _polytope(
            ("x", "y"),
            (("a", (0, 0)), ("b", (2, 0)), ("c", (1, 0)), ("d", (0, 1))),
        )
        result = polytope_prism(tri, "h")
        assert result.prism_affine_dimension == result.source_affine_dimension + 1


class TestNativeVsCatalogParity:
    def test_prism_catalog_entry_matches_native(self) -> None:
        request = PrismRequest(polytope=_square(), height_axis="h")
        assert compute_polytope_prism(request) == polytope_prism(_square(), "h")

    def test_join_catalog_entry_matches_native(self) -> None:
        left = _polytope(("x",), (("la", (0,)), ("lb", (1,))))
        right = _polytope(("y",), (("ra", (0,)), ("rb", (1,))))
        request = JoinRequest(left=left, right=right, height_axis="h")
        assert compute_polytope_join(request) == polytope_join(left, right, "h")

    def test_operations_are_published(self) -> None:
        ids = {tool.operation_id for tool in TOOLS}
        assert "polytope.rational.prism.compute" in ids
        assert "polytope.rational.join.compute" in ids
