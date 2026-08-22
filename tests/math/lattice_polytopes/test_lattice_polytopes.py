"""Tests for bounded exact lattice-point enumeration and counting."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.lattice_polytopes._models import (
    MAX_BOUND_SPAN,
    MAX_DIMENSION,
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
        with pytest.raises(ValidationError, match="unbounded"):
            LatticePolytopeRequest(
                halfspaces=(_hs((("1", "1"), ("0", "1")), ("1", "1")),)
            )

    def test_unbounded_quadrant_is_rejected(self) -> None:
        # x >= 0 and y >= 0 only: unbounded.
        with pytest.raises(ValidationError, match="unbounded"):
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
        with pytest.raises(ValidationError, match="exactly one"):
            LatticePolytopeRequest(
                vertices=(_v(("0", "1"), ("0", "1")),),
                halfspaces=(_hs((("1", "1"), ("0", "1")), ("1", "1")),),
            )

    def test_neither_representation_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            LatticePolytopeRequest()

    def test_empty_halfspace_polytope_rejected(self) -> None:
        # x <= 0 and -x <= -1 (i.e. x >= 1): no feasible point.
        with pytest.raises(ValidationError, match="empty"):
            LatticePolytopeRequest(
                halfspaces=(
                    _hs((("1", "1"),), ("0", "1")),
                    _hs((("-1", "1"),), ("-1", "1")),
                ),
            )

    def test_all_zero_halfspace_normal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not all be zero"):
            Halfspace(
                coefficients=(_cr("0"), _cr("0")),
                offset=_cr("1"),
            )


class TestBudgets:
    def test_bounding_box_span_bound_enforced(self) -> None:
        # A 1D interval spanning more than MAX_BOUND_SPAN integer points.
        far = str(MAX_BOUND_SPAN + 5)
        with pytest.raises(ValidationError, match="span bound"):
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
        with pytest.raises(LatticePointBudgetError, match="budget"):
            count_lattice_points(request)
