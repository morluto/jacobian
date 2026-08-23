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
        # exact inverted coordinate (10^20000+1)*10^20000 would need 40,001
        # digits, but the input components themselves already exceed the
        # 2,048-digit inversion admission bound, so the static check rejects
        # before any large intermediate is built.
        with pytest.raises(ValidationError, match="inversion admission bound"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr(format_canonical_integer(10**20000 + 1), "1"),
                point=RationalPoint2D(
                    x=_cr("1", format_canonical_integer(10**20000)),
                    y=_cr("0", "1"),
                ),
            )

    def test_rejects_oversized_result_off_center(self):
        # c=(1,1), s=10^1600, p=(1+10^{-3200},1): scale*dx = 10^4800, so the
        # exact inverted x-coordinate needs 4,801 digits; the inputs fit the
        # admission bound but the inverted result does not.
        with pytest.raises(ValidationError, match="inversion admission bound"):
            CircleInversionRequest(
                center=_pt(1, 1),
                power=_cr(format_canonical_integer(10**1600), "1"),
                point=RationalPoint2D(
                    x=CanonicalRational.from_fraction(
                        Fraction(1) + Fraction(1, 10**3200)
                    ),
                    y=_cr("1", "1"),
                ),
            )

    def test_rejects_near_limit_inputs_before_expanding_the_result(self):
        # Reviewer P1 shape: every request component fits the canonical limit
        # (exactly 32,768 digits here), so parsing is schema-valid, but the
        # exact inversion would form ~262,000-digit intermediates. Admission
        # must reject on the static input bound before computing anything.
        a = format_canonical_integer(10**32767)

        def frac(n: int, d: int) -> CanonicalRational:
            return _cr(
                format_canonical_integer(10**32767 + n),
                format_canonical_integer(10**32767 + d),
            )

        with pytest.raises(ValidationError, match="inversion admission bound"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr(a, format_canonical_integer(10**32767 + 29)),
                point=RationalPoint2D(
                    x=frac(1, 3),
                    y=frac(5, 7),
                ),
            )

    def test_reviewer_counterexample_is_closed_under_reinversion(self):
        # c=(0,0), s=1, p=(10^{-800},1): the first request returns components
        # within the admission bound (at most 1,601 digits), which the
        # fed-back request accepts identically. Admission must accept both
        # rounds and the double inversion must recover p exactly.
        tiny = CanonicalRational(num="1", den=format_canonical_integer(10**800))
        center = _pt(0, 0)
        point = RationalPoint2D(x=tiny, y=_cr("1", "1"))
        first = CircleInversionRequest(center=center, power=_cr("1", "1"), point=point)
        result = circle_inversion(first)
        assert len(result.point.x.num.lstrip("-")) == 801
        assert len(result.point.x.den) == 1601
        assert len(result.point.y.den) == 1601

        second = CircleInversionRequest(
            center=center, power=_cr("1", "1"), point=result.point
        )
        recovered = circle_inversion(second)
        assert recovered.point == point

    def test_admission_bound_boundary_is_exact(self):
        # A component with exactly 2,048 digits sits at the admission bound:
        # p=(10^2047,0) inverts to 1/10^2047 whose denominator carries
        # exactly 2,048 digits, so both sides fit and the request is
        # admitted. One digit more is rejected statically even though the
        # exact inverted point would remain canonically representable.
        edge = format_canonical_integer(10**2047)
        request = CircleInversionRequest(
            center=_pt(0, 0),
            power=_cr("1", "1"),
            point=RationalPoint2D(x=_cr(edge, "1"), y=_cr("0", "1")),
        )
        result = circle_inversion(request)
        assert result.point.x.as_fraction() == Fraction(1, 10**2047)

        over = format_canonical_integer(10**2048)
        with pytest.raises(ValidationError, match="inversion admission bound"):
            CircleInversionRequest(
                center=_pt(0, 0),
                power=_cr("1", "1"),
                point=RationalPoint2D(x=_cr(over, "1"), y=_cr("0", "1")),
            )

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

    def test_schema_publishes_numeric_admission_bound(self):
        from jacobian.math.geometry._models import INVERSION_ADMISSION_DIGITS

        assert INVERSION_ADMISSION_DIGITS == 2048
        schema = CircleInversionRequest.model_json_schema()
        description = schema.get("description", "")
        assert "INVERSION_ADMISSION_DIGITS" not in description
        assert f"{INVERSION_ADMISSION_DIGITS} decimal digits" in description
        assert (
            schema.get("inversion_admission_digit_bound") == INVERSION_ADMISSION_DIGITS
        )
        for field in ("center", "power", "point"):
            field_description = schema["properties"][field]["description"]
            assert f"{INVERSION_ADMISSION_DIGITS} decimal digits" in field_description
