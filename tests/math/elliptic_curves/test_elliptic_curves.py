"""Tests for short Weierstrass elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.elliptic_curves._models import (
    CurvePointRequest,
    EllipticCurvePointAdditionRequest,
    EllipticCurveRequest,
    RationalAffinePoint,
    ScalarMultiplicationRequest,
    ShortWeierstrassCurve,
)
from jacobian.math.elliptic_curves._operations import (
    add_points,
    check_point_on_curve,
    compute_discriminant,
    scalar_multiply,
)


def _pt(num: str, den: str = "1") -> dict:
    return {"num": num, "den": den}


class TestDiscriminant:
    def test_nonsingular_curve(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        result = compute_discriminant(EllipticCurveRequest(curve=curve))
        assert result.discriminant.as_fraction() == -64
        assert result.is_nonsingular

    def test_singular_curve(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("0"), coefficient_b=_pt("0"))
        result = compute_discriminant(EllipticCurveRequest(curve=curve))
        assert result.discriminant.as_fraction() == 0
        assert not result.is_nonsingular

    def test_curve_y2_x3_minus_2x(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        result = compute_discriminant(EllipticCurveRequest(curve=curve))
        assert result.discriminant.as_fraction() == 512
        assert result.is_nonsingular


class TestPointOnCurve:
    def test_point_on_curve(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        point = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = check_point_on_curve(CurvePointRequest(curve=curve, point=point))
        assert result.on_curve

    def test_point_not_on_curve(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        point = RationalAffinePoint(x=_pt("1"), y=_pt("1"))
        result = check_point_on_curve(CurvePointRequest(curve=curve, point=point))
        assert not result.on_curve


class TestPointAddition:
    def test_double_y_zero(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        point = RationalAffinePoint(x=_pt("0"), y=_pt("0"))
        result = add_points(
            EllipticCurvePointAdditionRequest(curve=curve, first=point, second=point)
        )
        assert result.at_infinity

    def test_add_distinct_points(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = add_points(
            EllipticCurvePointAdditionRequest(curve=curve, first=p, second=p)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == Fraction(9, 4)
        assert result.point.y.as_fraction() == Fraction(-21, 8)

    def test_add_negatives(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        neg_p = RationalAffinePoint(x=_pt("2"), y=_pt("-2"))
        result = add_points(
            EllipticCurvePointAdditionRequest(curve=curve, first=p, second=neg_p)
        )
        assert result.at_infinity


class TestScalarMultiplication:
    def test_zero_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=p, scalar=0)
        )
        assert result.at_infinity

    def test_one_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=p, scalar=1)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == 2
        assert result.point.y.as_fraction() == 2

    def test_two_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=p, scalar=2)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == Fraction(9, 4)
        assert result.point.y.as_fraction() == Fraction(-21, 8)

    def test_three_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=p, scalar=3)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == 338
        assert result.point.y.as_fraction() == 6214


class TestGroupLawAdmission:
    """The chord-and-tangent domain is enforced at the typed boundary."""

    def test_order_two_point_odd_multiple(self):
        """P=(0,0) on y^2=x^3+x has 2P=O, so 3P=P (not infinity)."""
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("0"), y=_pt("0"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=p, scalar=3)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == 0
        assert result.point.y.as_fraction() == 0

    def test_point_off_curve_rejected(self):
        """(1,1) does not lie on y^2=x^3+x; the old code returned a fake sum."""
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("1"), y=_pt("1"))
        with pytest.raises(ValidationError, match="lie on the curve"):
            add_points(
                EllipticCurvePointAdditionRequest(
                    curve=curve, first=p, second=p
                )
            )

    def test_singular_curve_rejected(self):
        """y^2=x^3 has a cusp at the origin: discriminant zero."""
        curve = ShortWeierstrassCurve(coefficient_a=_pt("0"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("1"), y=_pt("1"))
        with pytest.raises(ValidationError, match="nonsingular"):
            ScalarMultiplicationRequest(curve=curve, point=p, scalar=2)


class TestPrimeFieldModulusBound:
    def test_huge_modulus_rejected(self):
        from jacobian.math.prime_field_matrix_ops._models import RankRequest

        with pytest.raises(ValidationError):
            RankRequest(prime=2**200, entries=((1,),), columns=1)
