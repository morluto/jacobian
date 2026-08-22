"""Tests for exact circle inversion of rational planar points."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
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
        with pytest.raises(ValidationError, match="differ from the center"):
            CircleInversionRequest(
                center=_pt(1, 1),
                power=_cr("1", "1"),
                point=_pt(1, 1),
            )

    def test_rejects_nonpositive_power(self):
        with pytest.raises(ValidationError, match="positive"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("0", "1"),
                point=_pt(1, 0),
            )
        with pytest.raises(ValidationError, match="positive"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("-1", "1"),
                point=_pt(1, 0),
            )

    def test_rejects_output_that_exceeds_canonical_digits(self):
        # center=(0,0), power=(10^20000+1)/1, point=(10^{-20000}, 0): the
        # exact inverted coordinate (10^20000+1)*10^20000 needs 40,001 digits,
        # so admission rejects before the unrepresentable value is built.
        with pytest.raises(ValidationError, match="canonical limit"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr(format_canonical_integer(10**20000 + 1), "1"),
                point=RationalPoint2D(
                    x=_cr("1", format_canonical_integer(10**20000)),
                    y=_cr("0", "1"),
                ),
            )

    def test_rejects_oversized_result_off_center(self):
        # c=(1,1), s=10^16000, p=(1+10^{-20000},1): scale*dx = 10^36000, so
        # the exact inverted x-coordinate needs 36,001 digits.
        with pytest.raises(ValidationError, match="canonical limit"):
            CircleInversionRequest(
                center=_pt(1, 1),
                power=_cr(format_canonical_integer(10**16000), "1"),
                point=RationalPoint2D(
                    x=CanonicalRational.from_fraction(
                        Fraction(1) + Fraction(1, 10**20000)
                    ),
                    y=_cr("1", "1"),
                ),
            )

    def test_reviewer_counterexample_is_closed_under_reinversion(self):
        # c=(0,0), s=1, p=(10^{-9000},1): the first request returns components
        # with 18,001 digits, which the former half-limit input bound rejected
        # on the second invocation. Admission must accept both rounds and the
        # double inversion must recover p exactly.
        tiny = CanonicalRational(num="1", den=format_canonical_integer(10**9000))
        center = _pt(0, 0)
        point = RationalPoint2D(x=tiny, y=_cr("1", "1"))
        first = CircleInversionRequest(center=center, power=_cr("1", "1"), point=point)
        result = circle_inversion(first)
        assert len(result.point.x.num) == 9001
        assert len(result.point.x.den) == 18001
        assert len(result.point.y.den) == 18001

        second = CircleInversionRequest(
            center=center, power=_cr("1", "1"), point=result.point
        )
        recovered = circle_inversion(second)
        assert recovered.point == point

    def test_admits_large_inputs_whose_exact_result_is_representable(self):
        # p=(10^{-20000},10^{-20000}) inverts to (5*10^19999, 5*10^19999);
        # every component fits the canonical limit even though the input
        # height exceeds any half-limit heuristic.
        tiny = _cr("1", format_canonical_integer(10**20000))
        request = CircleInversionRequest(
            center=_pt(0, 0),
            power=_cr("1", "1"),
            point=RationalPoint2D(x=tiny, y=tiny),
        )
        result = circle_inversion(request)
        assert result.point.x.as_fraction() == Fraction(10**20000, 2)
        assert result.point.y.as_fraction() == Fraction(10**20000, 2)

    def test_admitted_domain_is_symmetric_under_involution(self):
        # Defining invariant: for every admitted (c,s,p), the fed-back request
        # (c,s,I(p)) is admitted again and recovers p.
        cases = (
            (_pt(0, 0), _cr("1", "1"), _pt(3, -7)),
            (_pt(1, 1), _cr("2", "1"), _pt(-4, 9)),
            (_pt(-2, 5), _cr("7", "3"), _pt(11, -13)),
            (
                _pt(0, 0),
                _cr("6", "5"),
                RationalPoint2D(x=_cr("3", "7"), y=_cr("-2", "11")),
            ),
        )
        for center, power, point in cases:
            first = CircleInversionRequest(center=center, power=power, point=point)
            inverted = circle_inversion(first)
            second = CircleInversionRequest(
                center=center, power=power, point=inverted.point
            )
            assert circle_inversion(second).point == point

    def test_schema_documents_point_not_equal_center(self):
        schema = CircleInversionRequest.model_json_schema()
        assert "p != c" in schema.get("description", "")
        assert "p != c" in schema["properties"]["center"]["description"]
        assert "p != c" in schema["properties"]["point"]["description"]
