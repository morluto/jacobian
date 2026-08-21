"""Tests for exact circle inversion of rational planar points."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    CircleInversionRequest,
    GeometryPointResult,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circle_inversion


def _pt(x: int | str, y: int | str) -> RationalPoint2D:
    return RationalPoint2D(
        x={"num": str(x), "den": "1"},
        y={"num": str(y), "den": "1"},
    )


def _cr(num: str, den: str) -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


class TestCircleInversion:
    def test_unit_inversion_of_two_zero(self):
        result = circle_inversion(
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("1", "1"),
                point=_pt(2, 0),
            ),
        )
        assert isinstance(result, GeometryPointResult)
        assert result.point.x.as_fraction() == Fraction(1, 2)
        assert result.point.y.as_fraction() == Fraction(0)

    def test_unit_inversion_of_one_two(self):
        result = circle_inversion(
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("1", "1"),
                point=_pt(1, 2),
            ),
        )
        # ||(1,2)||^2 = 5, so I = (1/5, 2/5).
        assert result.point.x.as_fraction() == Fraction(1, 5)
        assert result.point.y.as_fraction() == Fraction(2, 5)

    def test_non_origin_center(self):
        result = circle_inversion(
            CircleInversionRequest(
                center=_pt(1, 1),
                power=_cr("2", "1"),
                point=_pt(3, 1),
            ),
        )
        # d = (2,0), ||d||^2 = 4, scale = 2/4 = 1/2, I = (1+1, 1+0) = (2,1).
        assert result.point.x.as_fraction() == Fraction(2, 1)
        assert result.point.y.as_fraction() == Fraction(1, 1)

    def test_involutive_on_the_inversion_circle(self):
        # A point at distance sqrt(s) from the center is fixed by inversion.
        result = circle_inversion(
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("4", "1"),
                point=_pt(2, 0),
            ),
        )
        assert result.point.x.as_fraction() == Fraction(2, 1)
        assert result.point.y.as_fraction() == Fraction(0, 1)

    def test_double_inversion_recovers_original(self):
        first = circle_inversion(
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("3", "1"),
                point=_pt(1, 1),
            ),
        )
        second = circle_inversion(
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("3", "1"),
                point=first.point,
            ),
        )
        assert second.point.x.as_fraction() == Fraction(1, 1)
        assert second.point.y.as_fraction() == Fraction(1, 1)

    def test_rejects_point_at_center(self):
        import pytest

        with pytest.raises(ValueError, match="differ from the center"):
            CircleInversionRequest(
                center=_pt(1, 1),
                power=_cr("1", "1"),
                point=_pt(1, 1),
            )

    def test_rejects_nonpositive_power(self):
        import pytest

        with pytest.raises(ValueError, match="positive"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("0", "1"),
                point=_pt(1, 0),
            )
        with pytest.raises(ValueError, match="positive"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("-1", "1"),
                point=_pt(1, 0),
            )
