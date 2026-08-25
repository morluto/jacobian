"""Tests for bounded exact lattice-point enumeration and counting."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer
from jacobian.math.lattice_polytopes._models import (
    MAX_BOUND_SPAN,
    MAX_DIMENSION,
    EnumerateLatticePointsRequest,
    Halfspace,
    LatticePolytopeRequest,
    Vertex,
)
from jacobian.math.lattice_polytopes._operations import (
    LatticePointBudgetError,
    count_lattice_points,
    enumerate_lattice_points,
)


def _cr(num: str, den: str = "1") -> dict[str, str]:
    return {"num": num, "den": den}


def _v(*coords: tuple[str, str]) -> Vertex:
    return Vertex(coordinates=tuple(_cr(n, d) for n, d in coords))


def _hs(coeffs: tuple[tuple[str, str], ...], offset: tuple[str, str]) -> Halfspace:
    return Halfspace(
        coefficients=tuple(_cr(n, d) for n, d in coeffs),
        offset=_cr(offset[0], offset[1]),
    )


@contextmanager
def raises_code(code: str):
    with pytest.raises(ValidationError) as error:
        yield
    assert error.value.errors()[0]["type"] == f"lattice_polytope.{code}"


UNIT_SQUARE_V = (
    _v(("0", "1"), ("0", "1")),
    _v(("1", "1"), ("0", "1")),
    _v(("1", "1"), ("1", "1")),
    _v(("0", "1"), ("1", "1")),
)

UNIT_SQUARE_H = (
    _hs((("1", "1"), ("0", "1")), ("1", "1")),  #  x <= 1
    _hs((("-1", "1"), ("0", "1")), ("0", "1")),  # -x <= 0
    _hs((("0", "1"), ("1", "1")), ("1", "1")),  #  y <= 1
    _hs((("0", "1"), ("-1", "1")), ("0", "1")),  # -y <= 0
)


class TestEnumerate:
    def test_unit_square_vertices(self) -> None:
        result = enumerate_lattice_points(
            LatticePolytopeRequest(vertices=UNIT_SQUARE_V)
        )
        assert result.point_count == 4
        assert result.representation == "vertices"
        coords = {p.coordinates for p in result.points}
        assert coords == {
            ("0", "0"),
            ("1", "0"),
            ("0", "1"),
            ("1", "1"),
        }

    def test_unit_square_halfspaces(self) -> None:
        result = enumerate_lattice_points(
            LatticePolytopeRequest(halfspaces=UNIT_SQUARE_H)
        )
        assert result.point_count == 4
        assert result.representation == "halfspaces"
        coords = {p.coordinates for p in result.points}
        assert coords == {
            ("0", "0"),
            ("1", "0"),
            ("0", "1"),
            ("1", "1"),
        }

    def test_simplex_two_by_two(self) -> None:
        # conv((0,0),(2,0),(0,2)) contains 6 lattice points.
        result = enumerate_lattice_points(
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1")),
                    _v(("2", "1"), ("0", "1")),
                    _v(("0", "1"), ("2", "1")),
                ),
            )
        )
        assert result.point_count == 6
        coords = {p.coordinates for p in result.points}
        assert coords == {
            ("0", "0"),
            ("1", "0"),
            ("2", "0"),
            ("0", "1"),
            ("1", "1"),
            ("0", "2"),
        }

    def test_three_dimensional_tetrahedron(self) -> None:
        # conv(0, e1, e2, e3) has exactly its 4 vertices as lattice points.
        result = enumerate_lattice_points(
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1"), ("0", "1")),
                    _v(("1", "1"), ("0", "1"), ("0", "1")),
                    _v(("0", "1"), ("1", "1"), ("0", "1")),
                    _v(("0", "1"), ("0", "1"), ("1", "1")),
                ),
            )
        )
        assert result.point_count == 4
        assert result.dimension == 3

    def test_one_dimensional_interval(self) -> None:
        result = enumerate_lattice_points(
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1")),
                    _v(("3", "1")),
                ),
            )
        )
        assert result.point_count == 4
        assert result.dimension == 1
        assert [p.coordinates for p in result.points] == [
            ("0",),
            ("1",),
            ("2",),
            ("3",),
        ]

    def test_rational_vertex_box_rounds_inclusively(self) -> None:
        # A square with a non-integer vertex span still scans inclusively.
        # vertices (0,0), (3/2, 0), (3/2, 3/2), (0, 3/2): box [0,2]x[0,2].
        result = enumerate_lattice_points(
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1")),
                    _v(("3", "2"), ("0", "1")),
                    _v(("3", "2"), ("3", "2")),
                    _v(("0", "1"), ("3", "2")),
                ),
            )
        )
        # The half-open convex hull of these vertices is [0, 3/2] x [0, 3/2],
        # which contains integer points (0,0),(0,1),(1,0),(1,1) only.
        coords = {p.coordinates for p in result.points}
        assert coords == {("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")}


class TestCount:
    def test_unit_square_count_matches_enumerate(self) -> None:
        request = LatticePolytopeRequest(vertices=UNIT_SQUARE_V)
        assert count_lattice_points(request).point_count == 4
        assert count_lattice_points(request).representation == "vertices"

    def test_simplex_count(self) -> None:
        result = count_lattice_points(
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1")),
                    _v(("2", "1"), ("0", "1")),
                    _v(("0", "1"), ("2", "1")),
                ),
            )
        )
        assert result.point_count == 6

    def test_count_and_enumerate_agree_on_cube(self) -> None:
        cube = tuple(
            _v(
                (str(a), "1"),
                (str(b), "1"),
                (str(c), "1"),
            )
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
        )
        request = LatticePolytopeRequest(vertices=cube)
        enumerated = enumerate_lattice_points(request).point_count
        request2 = LatticePolytopeRequest(vertices=cube)
        counted = count_lattice_points(request2).point_count
        assert enumerated == 8
        assert counted == 8


class TestRejection:
    def test_unbounded_halfspace_representation_is_rejected(self) -> None:
        # Only x <= 1: the polytope is unbounded in every other direction.
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                halfspaces=(_hs((("1", "1"), ("0", "1")), ("1", "1")),)
            )

    def test_unbounded_quadrant_is_rejected(self) -> None:
        # x >= 0 and y >= 0 only: unbounded.
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                halfspaces=(
                    _hs((("-1", "1"), ("0", "1")), ("0", "1")),
                    _hs((("0", "1"), ("-1", "1")), ("0", "1")),
                ),
            )

    def test_dimension_exceeds_bound(self) -> None:
        with pytest.raises(ValidationError):
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1"), ("0", "1"), ("0", "1"), ("0", "1")),
                    _v(("1", "1"), ("0", "1"), ("0", "1"), ("0", "1"), ("0", "1")),
                ),
            )

    def test_explicit_dimension_bound_rejects_larger(self) -> None:
        # A 5-dimensional vertex is rejected by the field max_length.
        with pytest.raises(ValidationError):
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1"), ("0", "1"), ("0", "1"), ("0", "1")),
                ),
                dimension_bound=MAX_DIMENSION,
            )

    def test_both_representations_rejected(self) -> None:
        with raises_code("representation_not_exclusive"):
            LatticePolytopeRequest(
                vertices=(_v(("0", "1"), ("0", "1")),),
                halfspaces=(_hs((("1", "1"), ("0", "1")), ("1", "1")),),
            )

    def test_neither_representation_rejected(self) -> None:
        with raises_code("representation_not_exclusive"):
            LatticePolytopeRequest()

    def test_all_zero_halfspace_normal_rejected(self) -> None:
        with raises_code("halfspace_normal_zero"):
            Halfspace(
                coefficients=(_cr("0"), _cr("0")),
                offset=_cr("1"),
            )


class TestBudgets:
    def test_bounding_box_span_bound_enforced(self) -> None:
        # A 1D interval spanning more than MAX_BOUND_SPAN integer points.
        far = str(MAX_BOUND_SPAN + 5)
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1")),
                    _v((far, "1")),
                )
            )

    def test_lattice_point_cap_enforced(self) -> None:
        # A 2D box just within the per-axis span bound but with more than
        # MAX_LATTICE_POINTS interior points. Use a near-square box.
        from jacobian.math.lattice_polytopes._models import MAX_LATTICE_POINTS

        side = 1010  # 1010 x 1010 = 1_020_100 > 1_000_000; per-axis 1011 <= 10000
        far = str(side)
        request = LatticePolytopeRequest(
            vertices=(
                _v(("0", "1"), ("0", "1")),
                _v((far, "1"), ("0", "1")),
                _v((far, "1"), (far, "1")),
                _v(("0", "1"), (far, "1")),
            )
        )
        assert (side + 1) * (side + 1) > MAX_LATTICE_POINTS
        # Enumeration materializes the points, so it fails closed with a
        # typed budget outcome (the point cap or the output-size estimate).
        with pytest.raises(LatticePointBudgetError):
            enumerate_lattice_points(request)
        # ...while counting returns the small exact integer answer.
        result = count_lattice_points(request)
        assert result.point_count == (side + 1) * (side + 1)

    def test_thousand_square_count_exceeding_the_cap(self) -> None:
        # [0,1000]^2 holds 1_002_001 lattice points: above the
        # materialization cap but inside the admitted scan budget.
        request = LatticePolytopeRequest(
            vertices=(
                _v(("0", "1"), ("0", "1")),
                _v(("1000", "1"), ("0", "1")),
                _v(("1000", "1"), ("1000", "1")),
                _v(("0", "1"), ("1000", "1")),
            )
        )
        assert count_lattice_points(request).point_count == 1001 * 1001


class TestMembershipWorkBudget:
    """Membership work is bounded by scan times *distinct* facet inequalities."""

    def test_normalization_merges_equivalent_halfspaces(self) -> None:
        from fractions import Fraction

        from jacobian.math.lattice_polytopes._operations import (
            _dedupe_normalized_halfspaces,
        )

        base = [
            ([Fraction(1), Fraction(0)], Fraction(2499)),
            ([Fraction(-1), Fraction(0)], Fraction(0)),
            ([Fraction(0), Fraction(1)], Fraction(3999)),
            ([Fraction(0), Fraction(-1)], Fraction(0)),
        ]
        deduped = _dedupe_normalized_halfspaces(base * 16)
        assert len(deduped) == 4
        assert _dedupe_normalized_halfspaces(base) == deduped
        # Positive rescalings of the same inequality collapse onto the
        # primitive form: 2x <= 4998 and -3x <= 0 are x <= 2499, x >= 0.
        rescaled = [
            *base,
            ([Fraction(2), Fraction(0)], Fraction(4998)),
            ([Fraction(-3), Fraction(0)], Fraction(0)),
            ([Fraction(1, 2), Fraction(0)], Fraction(Fraction(2499, 2))),
        ]
        assert len(_dedupe_normalized_halfspaces(rescaled)) == 4
        for coeffs, _ in _dedupe_normalized_halfspaces(rescaled):
            assert all(c.denominator == 1 for c in coeffs)

    def test_duplicated_inequalities_return_exact_counts(self) -> None:
        # Each side of the [0,9]^2 box repeated 16 times: the scan must use
        # the four distinct normalized facets and stay exact.
        sides = (
            _hs((("1", "1"), ("0", "1")), ("9", "1")),
            _hs((("-1", "1"), ("0", "1")), ("0", "1")),
            _hs((("0", "1"), ("1", "1")), ("9", "1")),
            _hs((("0", "1"), ("-1", "1")), ("0", "1")),
        )
        request = LatticePolytopeRequest(halfspaces=sides * 16)
        assert len(request.halfspaces) == 64
        assert count_lattice_points(request).point_count == 100

    def test_reviewer_wide_box_with_repeats_is_admitted(self) -> None:
        # [0,2499] x [0,3999] with every inequality repeated 16 times passes
        # the 10M-candidate scan bound; normalization keeps the membership
        # work at 4 distinct facets so the request is admitted instead of
        # failing internally after acceptance.
        sides = (
            _hs((("1", "1"), ("0", "1")), ("2499", "1")),
            _hs((("-1", "1"), ("0", "1")), ("0", "1")),
            _hs((("0", "1"), ("1", "1")), ("3999", "1")),
            _hs((("0", "1"), ("-1", "1")), ("0", "1")),
        )
        request = LatticePolytopeRequest(halfspaces=sides * 16)
        assert len(request.halfspaces) == 64

    def test_distinct_facet_excess_is_rejected_at_validation(self) -> None:
        # 4 box sides + 7 distinct redundant diagonal cuts = 11 distinct
        # facets over a 10M-candidate scan: beyond the membership budget,
        # so the domain is narrowed in the request model itself.
        sides = [
            _hs((("1", "1"), ("0", "1")), ("9999", "1")),
            _hs((("-1", "1"), ("0", "1")), ("0", "1")),
            _hs((("0", "1"), ("1", "1")), ("999", "1")),
            _hs((("0", "1"), ("-1", "1")), ("0", "1")),
        ]
        diagonals = [
            _hs((("1", "1"), ("1", "1")), (str(c), "1")) for c in range(10998, 11005)
        ]
        with raises_code("geometry_work_exceeds_bound"):
            LatticePolytopeRequest(halfspaces=tuple(sides + diagonals))

    def test_membership_work_boundary_accepts_the_limit(self) -> None:
        # Exactly 10 distinct facets over a 10M-candidate scan sits at the
        # 100M-test budget and is admitted; duplicates do not push past it.
        sides = [
            _hs((("1", "1"), ("0", "1")), ("9999", "1")),
            _hs((("-1", "1"), ("0", "1")), ("0", "1")),
            _hs((("0", "1"), ("1", "1")), ("999", "1")),
            _hs((("0", "1"), ("-1", "1")), ("0", "1")),
        ]
        six_cuts = [
            _hs((("1", "1"), ("1", "1")), (str(c), "1")) for c in range(10998, 11004)
        ]
        boundary = LatticePolytopeRequest(halfspaces=tuple(sides + six_cuts))
        padded = LatticePolytopeRequest(
            halfspaces=tuple(list(boundary.halfspaces) + [sides[0]] * 6)
        )
        assert len(padded.halfspaces) == 16


class TestCountResultConstraints:
    def test_count_result_dimension_is_capped(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            MAX_DIMENSION,
            CountLatticePointsResult,
        )

        assert (
            CountLatticePointsResult(
                dimension=MAX_DIMENSION,
                point_count=0,
                representation="vertices",
            ).dimension
            == MAX_DIMENSION
        )
        with pytest.raises(ValidationError):
            CountLatticePointsResult(
                dimension=MAX_DIMENSION + 1,
                point_count=0,
                representation="vertices",
            )

    def test_result_representation_is_a_closed_vocabulary(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            CountLatticePointsResult,
            EnumerateLatticePointsResult,
            LatticePoint,
        )

        with pytest.raises(ValidationError):
            CountLatticePointsResult(
                dimension=2, point_count=0, representation="anything"
            )
        with pytest.raises(ValidationError):
            EnumerateLatticePointsResult(
                dimension=2,
                point_count=1,
                points=(LatticePoint(coordinates=("0", "0")),),
                representation="anything",
            )


class TestLowerDimensionalRejection:
    def test_segment_in_three_d_rejected_at_validation(self) -> None:
        # Two affinely dependent vertices in 3-D define a segment.  The old
        # behaviour skipped the rank guard when len(vertices) < dimension and
        # counted the whole eight-point bounding box instead of the segment.
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1"), ("0", "1")),
                    _v(("1", "1"), ("1", "1"), ("1", "1")),
                )
            )

    def test_single_vertex_in_two_d_rejected(self) -> None:
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(vertices=(_v(("0", "1"), ("0", "1")),))

    def test_collinear_triangle_rejected(self) -> None:
        # Three collinear vertices in 2-D: affine rank 1 < 2.
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1")),
                    _v(("1", "1"), ("1", "1")),
                    _v(("2", "1"), ("2", "1")),
                )
            )

    def test_declarations_advertise_full_dimensional_restriction(self) -> None:
        """Both declarations and the request schema state the restriction."""
        from jacobian.math.lattice_polytopes._tools import TOOLS

        tools = {tool.operation_id: tool for tool in TOOLS}
        for operation_id in (
            "polytope.lattice_points.enumerate",
            "polytope.lattice_points.count",
        ):
            description = tools[operation_id].description.lower()
            assert "full-dimensional" in description
            assert "affinely span" in description
        schema = LatticePolytopeRequest.model_json_schema()
        vertices_description = schema["properties"]["vertices"]["description"].lower()
        assert "affinely span" in vertices_description


class TestLargeCoordinateBounds:
    def test_singleton_beyond_python_digit_limit_enumerates(self) -> None:
        # A singleton 1-D V-representation at 10**5000: the public contract
        # admits 32,768-digit coordinates, so measuring the bounding box must
        # not trip CPython's default 4,300-digit int-to-str limit.
        huge = format_canonical_integer(10**5000)
        result = enumerate_lattice_points(
            LatticePolytopeRequest(vertices=(_v((huge, "1")),))
        )
        assert result.point_count == 1
        assert result.points[0].coordinates == (huge,)


class TestEnumerateRequestBoundary:
    def test_oversize_artifact_rejected_at_request_validation(self) -> None:
        # [0,599]^2 holds 360k lattice points; the serialized artifact
        # exceeds the 10 MiB output limit, which the dispatch layer only
        # translates for request parsing, so admission rejects it at
        # request construction instead of failing inside operation.run.
        far = "599"
        with raises_code("enumeration_artifact_invalid"):
            EnumerateLatticePointsRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1")),
                    _v((far, "1"), ("0", "1")),
                    _v((far, "1"), (far, "1")),
                    _v(("0", "1"), (far, "1")),
                )
            )

    def test_oversize_count_rejected_at_enumerate_boundary(self) -> None:
        # [0,1000]^2 holds more than MAX_LATTICE_POINTS points: counting
        # succeeds but enumeration must be refused before execution.
        far = "1000"
        with raises_code("enumeration_artifact_invalid"):
            EnumerateLatticePointsRequest(
                vertices=(
                    _v(("0", "1"), ("0", "1")),
                    _v((far, "1"), ("0", "1")),
                    _v((far, "1"), (far, "1")),
                    _v(("0", "1"), (far, "1")),
                )
            )

    def test_result_count_must_match_points(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            EnumerateLatticePointsResult,
            LatticePoint,
        )

        with raises_code("point_count_mismatch"):
            EnumerateLatticePointsResult(
                dimension=1,
                point_count=0,
                points=(LatticePoint(coordinates=("1",)),),
                representation="vertices",
            )

    def test_coordinates_require_canonical_integers(self) -> None:
        from jacobian.math.lattice_polytopes._models import LatticePoint

        with pytest.raises(ValidationError):
            LatticePoint(coordinates=("01",))
        with pytest.raises(ValidationError):
            LatticePoint(coordinates=("x",))
        assert LatticePoint(coordinates=("-42",)).coordinates == ("-42",)


class TestEnumerationResultInvariants:
    def test_duplicate_points_rejected(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            EnumerateLatticePointsResult,
            LatticePoint,
        )

        with raises_code("duplicate_lattice_point"):
            EnumerateLatticePointsResult(
                dimension=1,
                point_count=2,
                points=(
                    LatticePoint(coordinates=("0",)),
                    LatticePoint(coordinates=("0",)),
                ),
                representation="vertices",
            )

    def test_dimension_must_match_point_coordinates(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            EnumerateLatticePointsResult,
            LatticePoint,
        )

        with raises_code("point_dimension_mismatch"):
            EnumerateLatticePointsResult(
                dimension=2,
                point_count=1,
                points=(LatticePoint(coordinates=("0",)),),
                representation="vertices",
            )

    def test_coordinate_digit_limit_is_exact(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            COORDINATE_DIGITS,
            LatticePoint,
        )

        exactly_at = "9" * COORDINATE_DIGITS
        assert LatticePoint(coordinates=(exactly_at,)).coordinates == (exactly_at,)
        with raises_code("lattice_point_coordinate_digit_bound"):
            LatticePoint(coordinates=("9" * (COORDINATE_DIGITS + 1),))


class TestVertexFacetMembershipBudget:
    def test_parabola_vertex_facet_excess_rejected_at_validation(self) -> None:
        """64 points on a strictly convex parabola fill a 7.88M-candidate
        scan box and generate ~64 hull facets; scan-times-facet-count far
        exceeds the 100M membership budget, so validation must reject the
        request instead of permitting hundreds of millions of exact
        evaluations (review counterexample shape)."""
        vertices = tuple(
            _v((str(i * 63), "1"), (str(i * i // 2), "1")) for i in range(64)
        )
        with raises_code("geometry_work_exceeds_bound"):
            LatticePolytopeRequest(vertices=vertices)

    def test_small_vertex_polygon_still_admitted(self) -> None:
        """A small full-dimensional V-representation stays within the
        combined scan-times-facet budget."""
        triangle = (
            _v(("0", "1"), ("0", "1")),
            _v(("4", "1"), ("0", "1")),
            _v(("0", "1"), ("4", "1")),
        )
        request = LatticePolytopeRequest(vertices=triangle)
        assert count_lattice_points(request).point_count == 15


class TestOneDimensionalSingletonException:
    def test_declarations_document_the_one_dimensional_exception(self) -> None:
        """The admitted 1-D singleton is a documented exception, not an
        undocumented gap in the full-dimensional restriction."""
        from jacobian.math.lattice_polytopes._tools import TOOLS

        tools = {tool.operation_id: tool for tool in TOOLS}
        for operation_id in (
            "polytope.lattice_points.enumerate",
            "polytope.lattice_points.count",
        ):
            description = tools[operation_id].description.lower()
            assert "exception" in description
            assert "one-dimensional" in description
        schema = LatticePolytopeRequest.model_json_schema()
        vertices_description = schema["properties"]["vertices"]["description"].lower()
        assert "exception" in vertices_description

    def test_singleton_roundtrip_unchanged(self) -> None:
        result = enumerate_lattice_points(
            LatticePolytopeRequest(vertices=(_v(("3", "1")),))
        )
        assert result.point_count == 1


class TestFacetGeometryComputedOnce:
    """The C(n,d) subset enumeration runs once per request, not per phase."""

    SQUARE_WITH_EDGE_MIDPOINTS = (
        _v(("0", "1"), ("0", "1")),
        _v(("1", "1"), ("0", "1")),
        _v(("2", "1"), ("0", "1")),
        _v(("2", "1"), ("1", "1")),
        _v(("2", "1"), ("2", "1")),
        _v(("1", "1"), ("2", "1")),
        _v(("0", "1"), ("2", "1")),
        _v(("0", "1"), ("1", "1")),
    )

    def _count_facet_passes(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        from jacobian.math.lattice_polytopes import _operations

        passes = []

        original = _operations._facets_from_points

        def counting(verts, d):
            passes.append(d)
            return original(verts, d)

        monkeypatch.setattr(_operations, "_facets_from_points", counting)
        return passes

    def test_enumerate_request_and_execution_share_one_facet_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validation, artifact admission, and execution must reuse the
        computed facet geometry instead of repeating the bounded
        C(n,d)-subset enumeration for each phase."""
        passes = self._count_facet_passes(monkeypatch)
        request = EnumerateLatticePointsRequest(
            vertices=self.SQUARE_WITH_EDGE_MIDPOINTS
        )
        assert len(passes) == 1
        result = enumerate_lattice_points(request)
        assert result.point_count == 9
        assert len(passes) == 1

    def test_count_execution_reuses_validation_geometry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passes = self._count_facet_passes(monkeypatch)
        request = LatticePolytopeRequest(vertices=self.SQUARE_WITH_EDGE_MIDPOINTS)
        assert count_lattice_points(request).point_count == 9
        assert len(passes) == 1


class TestCountResultScanCap:
    def test_count_point_count_is_capped_at_the_scan_maximum(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            MAX_TOTAL_SCAN,
            CountLatticePointsResult,
        )

        at_limit = CountLatticePointsResult(
            dimension=1,
            point_count=MAX_TOTAL_SCAN,
            representation="vertices",
        )
        assert at_limit.point_count == MAX_TOTAL_SCAN
        with pytest.raises(ValidationError):
            CountLatticePointsResult(
                dimension=1,
                point_count=MAX_TOTAL_SCAN + 1,
                representation="vertices",
            )

    def test_admitted_count_results_satisfy_the_cap(self) -> None:
        from jacobian.math.lattice_polytopes._models import MAX_TOTAL_SCAN

        result = count_lattice_points(LatticePolytopeRequest(vertices=UNIT_SQUARE_V))
        assert 0 <= result.point_count <= MAX_TOTAL_SCAN


class TestTightBoundingBox:
    def test_rational_extrema_use_integer_tight_bounds(self) -> None:
        # The interval [1/2, 19999/2] contains exactly the 9,999 lattice
        # points 1..9999. Rounding the box outwards would scan [0,10000]
        # (10,001 points) and falsely fail the per-axis span admission.
        result = count_lattice_points(
            LatticePolytopeRequest(
                vertices=(
                    _v(
                        ("1", "2"),
                    ),
                    _v(
                        ("19999", "2"),
                    ),
                ),
            )
        )
        assert result.point_count == 9_999

    def test_polytope_without_lattice_points_counts_zero(self) -> None:
        # [1/2, 3/4] contains no integer, so ceil(min) > floor(max): the
        # tight box is empty and the exact count is zero.
        result = count_lattice_points(
            LatticePolytopeRequest(
                vertices=(
                    _v(
                        ("1", "2"),
                    ),
                    _v(
                        ("3", "4"),
                    ),
                ),
            )
        )
        assert result.point_count == 0


class TestBoundedEmptyHalfspacePolytopes:
    """A bounded but empty H-representation has an exact empty value.

    Boundedness (trivial recession cone) is established first; complete
    vertex enumeration then finding no vertex proves the polyhedron
    empty, so count zero and an empty enumeration are the exact answers.
    """

    EMPTY_INTERVAL = (
        _hs((("1", "1"),), ("0", "1")),  #  x <= 0
        _hs((("-1", "1"),), ("-1", "1")),  # -x <= -1, i.e. x >= 1
    )

    def test_reviewer_bounded_empty_interval_counts_zero(self) -> None:
        request = LatticePolytopeRequest(halfspaces=self.EMPTY_INTERVAL)
        result = count_lattice_points(request)
        assert result.point_count == 0
        assert result.dimension == 1
        assert result.representation == "halfspaces"

    def test_reviewer_bounded_empty_interval_enumerates_no_points(self) -> None:
        request = EnumerateLatticePointsRequest(halfspaces=self.EMPTY_INTERVAL)
        result = enumerate_lattice_points(request)
        assert result.point_count == 0
        assert result.points == ()
        assert result.dimension == 1
        assert result.representation == "halfspaces"

    def test_bounded_empty_square_is_admitted_and_empty(self) -> None:
        # x <= -1 with x >= 2 and 0 <= y <= 0: positively spanning
        # normals, trivial recession cone, no feasible point in 2-D.
        square = (
            _hs((("1", "1"), ("0", "1")), ("-1", "1")),
            _hs((("-1", "1"), ("0", "1")), ("-2", "1")),
            _hs((("0", "1"), ("1", "1")), ("0", "1")),
            _hs((("0", "1"), ("-1", "1")), ("0", "1")),
        )
        counted = count_lattice_points(LatticePolytopeRequest(halfspaces=square))
        assert counted.point_count == 0
        enumerated = enumerate_lattice_points(
            EnumerateLatticePointsRequest(halfspaces=square)
        )
        assert enumerated.point_count == 0
        assert enumerated.points == ()

    def test_unbounded_representation_still_rejected(self) -> None:
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(halfspaces=(_hs((("1", "1"),), ("0", "1")),))

    def test_declarations_disclose_the_empty_h_result(self) -> None:
        from jacobian.math.lattice_polytopes._tools import TOOLS

        tools = {tool.operation_id: tool for tool in TOOLS}
        for operation_id in (
            "polytope.lattice_points.enumerate",
            "polytope.lattice_points.count",
        ):
            description = tools[operation_id].description.lower()
            assert "empty" in description
        schema = LatticePolytopeRequest.model_json_schema()
        halfspaces_description = schema["properties"]["halfspaces"]["description"]
        assert "empty" in halfspaces_description.lower()


class TestEnumerationResultPointCap:
    """The serialized enumeration cannot represent an impossible artifact."""

    def test_point_count_above_the_materialization_cap_is_rejected(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            MAX_LATTICE_POINTS,
            EnumerateLatticePointsResult,
        )

        # Field bounds run before cross-field validators, so the raised
        # error is the cap itself rather than the equality mismatch.
        with pytest.raises(ValidationError) as excinfo:
            EnumerateLatticePointsResult(
                dimension=1,
                point_count=MAX_LATTICE_POINTS + 1,
                points=(),
                representation="vertices",
            )
        assert any(
            error["type"] == "less_than_equal" and error["loc"] == ("point_count",)
            for error in excinfo.value.errors()
        )

    def test_schema_advertises_the_admitted_caps(self) -> None:
        from jacobian.math.lattice_polytopes._models import (
            MAX_LATTICE_POINTS,
            EnumerateLatticePointsResult,
        )

        properties = EnumerateLatticePointsResult.model_json_schema()["properties"]
        assert properties["point_count"]["maximum"] == MAX_LATTICE_POINTS
        assert properties["points"]["maxItems"] == MAX_LATTICE_POINTS

    def test_admitted_enumeration_results_satisfy_the_cap(self) -> None:
        from jacobian.math.lattice_polytopes._models import MAX_LATTICE_POINTS

        result = enumerate_lattice_points(
            LatticePolytopeRequest(vertices=UNIT_SQUARE_V)
        )
        assert 0 <= result.point_count <= MAX_LATTICE_POINTS
        assert len(result.points) <= MAX_LATTICE_POINTS


class TestReviewRegressions:
    def test_infeasible_but_bounded_h_system_admitted_as_empty(self):
        """x<=0 and -x<=-1 is empty (bounded); normals span only one axis."""
        from jacobian.math.lattice_polytopes._operations import _facets_and_box

        request = LatticePolytopeRequest(
            halfspaces=(
                _hs((("1", "1"), ("0", "1")), ("0", "1")),
                _hs((("-1", "1"), ("0", "1")), ("-1", "1")),
            )
        )
        geometry = _facets_and_box(request)
        # The canonical empty box: per-axis [0, -1] scanning no candidate.
        assert geometry[2] == [-1, -1]

    def test_feasible_unbounded_lineality_system_still_rejected(self):
        """A feasible vertex-free system with lineality stays rejected."""

        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                halfspaces=(_hs((("1", "1"), ("0", "1")), ("0", "1")),)
            )

    def test_examples_state_their_validator_owned_preconditions(self):
        """Discovery examples teach the admission preconditions."""
        from jacobian.math.lattice_polytopes._tools import TOOLS

        for tool in TOOLS:
            for ex in tool.examples:
                lowered = ex.description.lower()
                if ex.name == "unit_square_vertices":
                    assert "full" in lowered
                if ex.name == "unit_square_halfspaces":
                    assert "bounded" in lowered


class TestSecondWaveRegressions:
    def test_feasibility_probe_uses_unrestricted_coordinates(self):
        """x <= -1 is unbounded, not empty: the probe must not assume x >= 0."""
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(halfspaces=(_hs((("-1", "1"),), ("-1", "1")),))

    def test_empty_integer_slice_detected_before_span_budgets(self):
        """[0,9999]^2 x [1/3,2/3] has an exactly empty integer scan."""
        request = LatticePolytopeRequest.model_validate(
            {
                "vertices": [
                    {
                        "coordinates": [
                            {"num": a, "den": "1"},
                            {"num": b, "den": "1"},
                            {"num": c, "den": "3"},
                        ]
                    }
                    for a, b, c in (
                        ("0", "0", "1"),
                        ("9999", "0", "1"),
                        ("0", "9999", "1"),
                        ("9999", "9999", "2"),
                    )
                ]
            }
        )
        result = count_lattice_points(request)
        assert result.point_count == 0

    def test_derived_h_vertex_coordinates_bounded_at_admission(self):
        """x/(10^m) <= 10^k pinned from below derives x = 10^(k+m).

        Every input component stays beneath the canonical component limit,
        yet the solved vertex coordinate exceeds it; admission must reject
        before an accepted request fails while constructing LatticePoint.
        """
        small_den = "1" + "0" * 10000  # 10^10000, 10,001 digits
        big_offset = "1" + "0" * 25000  # 10^25000, 25,001 digits
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                halfspaces=(
                    _hs((("1", small_den),), (big_offset, "1")),
                    _hs((("-1", small_den),), ("-" + big_offset, "1")),
                )
            )

    def test_representation_bounds_are_schema_visible(self):
        from jacobian.math.lattice_polytopes._models import (
            MAX_HALFSPACES,
            MAX_VERTICES,
            LatticePolytopeRequest,
        )

        vertices_field = LatticePolytopeRequest.model_fields["vertices"]
        assert vertices_field.metadata or vertices_field.annotation is not None
        schema = LatticePolytopeRequest.model_json_schema()
        v_schema = schema["properties"]["vertices"]
        h_schema = schema["properties"]["halfspaces"]
        any_of_v = v_schema.get("anyOf", [v_schema])
        any_of_h = h_schema.get("anyOf", [h_schema])
        assert any(
            item.get("maxItems") == MAX_VERTICES
            for item in any_of_v
            if isinstance(item, dict)
        ), v_schema
        assert any(
            item.get("maxItems") == MAX_HALFSPACES
            for item in any_of_h
            if isinstance(item, dict)
        ), h_schema


BOUNDARY_COORDINATE = "1" + "0" * 32_767


def _singleton_halfspaces(*, negative: bool) -> tuple[Halfspace, ...]:
    """Pin x to ±10^32767 with matching upper and lower half-spaces."""
    upper = ("-" + BOUNDARY_COORDINATE) if negative else BOUNDARY_COORDINATE
    lower = BOUNDARY_COORDINATE if negative else "-" + BOUNDARY_COORDINATE
    return (
        _hs((("1", "1"),), (upper, "1")),
        _hs((("-1", "1"),), (lower, "1")),
    )


class TestDerivedCoordinateSignBoundary:
    """The canonical digit bound measures magnitude, not sign length."""

    def test_reviewer_negative_singleton_at_boundary_is_admitted(self):
        """x pinned to -10^32767 derives a vertex whose magnitude has exactly
        32,768 digits; the request and its single lattice point are valid."""
        request = LatticePolytopeRequest(
            halfspaces=_singleton_halfspaces(negative=True)
        )
        assert count_lattice_points(request).point_count == 1

    def test_negative_boundary_singleton_enumerates_exactly(self):
        from jacobian.math.lattice_polytopes._models import COORDINATE_DIGITS

        result = enumerate_lattice_points(
            EnumerateLatticePointsRequest(
                halfspaces=_singleton_halfspaces(negative=True)
            )
        )
        assert result.point_count == 1
        assert result.points[0].coordinates == ("-" + BOUNDARY_COORDINATE,)
        # The returned coordinate satisfies the same magnitude convention.
        assert len(result.points[0].coordinates[0].lstrip("-")) == COORDINATE_DIGITS

    def test_positive_boundary_singleton_is_admitted_symmetrically(self):
        request = LatticePolytopeRequest(
            halfspaces=_singleton_halfspaces(negative=False)
        )
        assert count_lattice_points(request).point_count == 1
        enumerated = enumerate_lattice_points(request)
        assert enumerated.points[0].coordinates == (BOUNDARY_COORDINATE,)

    @pytest.mark.parametrize("negative", [False, True])
    def test_derived_magnitudes_beyond_the_bound_stay_rejected(self, negative):
        """x/10^10000 <= ±10^25000 pins x = ±10^35000: beyond the canonical
        representable magnitude regardless of sign, so admission rejects."""
        small_den = "1" + "0" * 10000
        big_offset = "1" + "0" * 25000
        upper = ("-" + big_offset) if negative else big_offset
        lower = big_offset if negative else "-" + big_offset
        with raises_code("geometry_invalid"):
            LatticePolytopeRequest(
                halfspaces=(
                    _hs((("1", small_den),), (upper, "1")),
                    _hs((("-1", small_den),), (lower, "1")),
                )
            )


UNIT_SQUARE_4D_SIDES = (
    _hs(
        (("1", "1"), ("0", "1"), ("0", "1"), ("0", "1")),
        ("1", "1"),
    ),
    _hs(
        (("-1", "1"), ("0", "1"), ("0", "1"), ("0", "1")),
        ("0", "1"),
    ),
    _hs(
        (("0", "1"), ("1", "1"), ("0", "1"), ("0", "1")),
        ("1", "1"),
    ),
    _hs(
        (("0", "1"), ("-1", "1"), ("0", "1"), ("0", "1")),
        ("0", "1"),
    ),
    _hs(
        (("0", "1"), ("0", "1"), ("1", "1"), ("0", "1")),
        ("1", "1"),
    ),
    _hs(
        (("0", "1"), ("0", "1"), ("-1", "1"), ("0", "1")),
        ("0", "1"),
    ),
    _hs(
        (("0", "1"), ("0", "1"), ("0", "1"), ("1", "1")),
        ("1", "1"),
    ),
    _hs(
        (("0", "1"), ("0", "1"), ("0", "1"), ("-1", "1")),
        ("0", "1"),
    ),
)


class TestThirdWaveRegressions:
    def test_nonzero_normal_precondition_is_schema_visible(self):
        """The validator-owned nonzero-normal restriction is published in
        the Halfspace schema, the halfspaces field, and both declarations."""
        from jacobian.math.lattice_polytopes._tools import TOOLS

        schema = LatticePolytopeRequest.model_json_schema()
        halfspace_schema = schema["$defs"]["Halfspace"]
        coefficients_description = halfspace_schema["properties"]["coefficients"][
            "description"
        ].lower()
        assert "nonzero" in coefficients_description
        halfspaces_field = LatticePolytopeRequest.model_fields["halfspaces"]
        assert "nonzero" in halfspaces_field.description.lower()
        tools = {tool.operation_id: tool for tool in TOOLS}
        for operation_id in (
            "polytope.lattice_points.enumerate",
            "polytope.lattice_points.count",
        ):
            description = tools[operation_id].description.lower()
            assert "nonzero" in description

    def test_halfspace_example_no_longer_claims_lower_dimensional_rejection(self):
        """Only V-representations are rejected for lower dimension; the
        H-example must not advertise that restriction."""
        from jacobian.math.lattice_polytopes._tools import TOOLS

        for tool in TOOLS:
            for ex in tool.examples:
                if ex.name == "unit_square_halfspaces":
                    assert "lower-dimensional" not in ex.description.lower()

    def test_lower_dimensional_h_segment_is_accepted(self):
        """x<=0, -x<=0, y<=1, -y<=0 is the segment from (0,0) to (0,1):
        bounded with positively spanning normals, so it is admitted and
        counts its two endpoints."""
        segment = (
            _hs((("1", "1"), ("0", "1")), ("0", "1")),
            _hs((("-1", "1"), ("0", "1")), ("0", "1")),
            _hs((("0", "1"), ("1", "1")), ("1", "1")),
            _hs((("0", "1"), ("-1", "1")), ("0", "1")),
        )
        result = count_lattice_points(LatticePolytopeRequest(halfspaces=segment))
        assert result.point_count == 2

    def test_repeated_rows_deduplicated_before_vertex_enumeration(self, monkeypatch):
        """The reviewer's [0,1]^4 with every side repeated eight times:
        vertex enumeration and the recession-cone test see the 8 distinct
        primitive rows, not the 32 raw ones."""
        from jacobian.math.lattice_polytopes import _operations

        seen_sizes: list[int] = []
        original = _operations._vertices_from_h_representation

        def counting(halfspaces):
            seen_sizes.append(len(halfspaces))
            return original(halfspaces)

        monkeypatch.setattr(_operations, "_vertices_from_h_representation", counting)
        request = LatticePolytopeRequest(halfspaces=UNIT_SQUARE_4D_SIDES * 4)
        assert len(request.halfspaces) == 32
        assert seen_sizes == [8]
        geometry = request.admitted_geometry()
        assert len(geometry[0]) == 8
        assert count_lattice_points(request).point_count == 16
        assert seen_sizes == [8]

    def test_rescaled_duplicate_rows_collapse_before_enumeration(self, monkeypatch):
        """Positive rescalings of the same inequality collapse onto the
        primitive row before any geometry routine runs."""
        from jacobian.math.lattice_polytopes import _operations

        seen_sizes: list[int] = []
        original = _operations._vertices_from_h_representation

        def counting(halfspaces):
            seen_sizes.append(len(halfspaces))
            return original(halfspaces)

        monkeypatch.setattr(_operations, "_vertices_from_h_representation", counting)
        # Full square plus positive rescalings of two of its sides:
        # 6 raw rows collapse onto the 4 primitive constraints.
        sides = (
            *UNIT_SQUARE_H,
            _hs((("2", "1"), ("0", "1")), ("2", "1")),  # 2x <= 2 == x <= 1
            _hs((("0", "1"), ("-3", "1")), ("0", "1")),  # -3y <= 0 == -y <= 0
        )
        request = LatticePolytopeRequest(halfspaces=sides)
        assert seen_sizes == [4]
        assert count_lattice_points(request).point_count == 4
