"""Tests for configuration-level geometry operations (#2107, #2106)."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    GeneralPositionRequest,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import (
    circumradius_profile,
    general_position_search,
)


def _point(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational.from_fraction(Fraction(x)),
        y=CanonicalRational.from_fraction(Fraction(y)),
    )


class TestGeneralPosition:
    def test_square_concyclic(self):
        """Four vertices of a square are concyclic."""
        points = [
            _point("0", "0"), _point("1", "0"),
            _point("1", "1"), _point("0", "1"),
        ]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert result.num_points == 4
        assert not result.has_collinear_triple
        assert result.has_concyclic_quadruple
        assert len(result.concyclic_quadruples) == 1
        assert result.concyclic_quadruples[0].indices == (0, 1, 2, 3)

    def test_collinear_triple(self):
        """Three points on the x-axis are collinear."""
        points = [
            _point("0", "0"), _point("1", "0"),
            _point("2", "0"), _point("0", "1"),
        ]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert result.has_collinear_triple
        assert not result.has_concyclic_quadruple
        assert len(result.collinear_triples) == 1
        assert result.collinear_triples[0].indices == (0, 1, 2)

    def test_general_position(self):
        """Four points in general position."""
        points = [
            _point("-1", "0"), _point("1", "0"),
            _point("0", "2"), _point("0", "-2"),
        ]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert not result.has_collinear_triple
        assert not result.has_concyclic_quadruple

    def test_triangle_only(self):
        """A triangle has no collinear triple and no quadruple."""
        points = [_point("0", "0"), _point("1", "0"), _point("0", "1")]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert not result.has_collinear_triple
        assert not result.has_concyclic_quadruple

    def test_duplicate_points_rejected(self):
        """Duplicate points should be rejected."""
        with pytest.raises(Exception):
            GeneralPositionRequest(
                points=(_point("0", "0"), _point("0", "0"), _point("1", "0"))
            )


class TestCircumradiusProfile:
    def test_triangle(self):
        """A single triangle has one circumradius entry."""
        points = [_point("0", "0"), _point("1", "0"), _point("0", "1")]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        assert result.num_points == 3
        assert len(result.entries) == 1
        assert not result.entries[0].is_degenerate
        assert result.entries[0].radius_squared is not None
        assert result.entries[0].indices == (0, 1, 2)

    def test_collinear_triple_degenerate(self):
        """Collinear triple is marked as degenerate."""
        points = [
            _point("0", "0"), _point("1", "0"), _point("2", "0"),
        ]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        assert len(result.entries) == 1
        assert result.entries[0].is_degenerate
        assert result.entries[0].radius_squared is None

    def test_four_points(self):
        """Four points have C(4,3) = 4 triples."""
        points = [
            _point("0", "0"), _point("1", "0"),
            _point("0", "1"), _point("1", "1"),
        ]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        assert result.num_points == 4
        assert len(result.entries) == 4
        for entry in result.entries:
            assert not entry.is_degenerate

    def test_equal_circumradius(self):
        """All four triangles of a unit square have equal circumradius."""
        points = [
            _point("0", "0"), _point("1", "0"),
            _point("1", "1"), _point("0", "1"),
        ]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        radii = set()
        for entry in result.entries:
            if not entry.is_degenerate and entry.radius_squared:
                radii.add((entry.radius_squared.num, entry.radius_squared.den))
        assert len(radii) == 1
