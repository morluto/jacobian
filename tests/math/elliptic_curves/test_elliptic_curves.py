"""Tests for short Weierstrass elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.elliptic_curves._models import (
    CurveDiscriminantResult,
    CurvePointRequest,
    EllipticCurvePointAdditionRequest,
    EllipticCurvePointResult,
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


def _assert_error_code(
    exc_info: pytest.ExceptionInfo[ValidationError], code: str
) -> None:
    assert exc_info.value.errors()[0]["type"] == code


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

    def test_coefficients_exceeding_result_bound_rejected(self):
        """Exact Δ = -64·10^59997 for A=10^19999 has ~60k digits, past the
        canonical limit."""
        curve = ShortWeierstrassCurve(
            coefficient_a=_pt("1" + "0" * 19999), coefficient_b=_pt("0")
        )
        with pytest.raises(ValidationError) as exc_info:
            EllipticCurveRequest(curve=curve)
        _assert_error_code(exc_info, "elliptic_curve.discriminant_result_bound")

    def test_boundary_coefficients_accepted_and_returned(self):
        """The exact reduced Δ = -64A^3 for A=10^N has 3N+2 digits: N=10923
        exceeds the canonical bound and N=10921 stays within it."""
        rejected = ShortWeierstrassCurve(
            coefficient_a=_pt("1" + "0" * 10923), coefficient_b=_pt("0")
        )
        with pytest.raises(ValidationError) as exc_info:
            EllipticCurveRequest(curve=rejected)
        _assert_error_code(exc_info, "elliptic_curve.discriminant_result_bound")
        accepted = EllipticCurveRequest(
            curve=ShortWeierstrassCurve(
                coefficient_a=_pt("1" + "0" * 10921), coefficient_b=_pt("0")
            )
        )
        result = compute_discriminant(accepted)
        assert result.is_nonsingular
        digits = max(len(result.discriminant.num), len(result.discriminant.den))
        assert digits <= 32_768

    def test_exact_discriminant_cancellation_admitted(self):
        """A=-3t², B=2t³ makes 4A³+27B² vanish exactly despite ~20k- and
        ~30k-digit terms; the reduced exact discriminant admits the request
        and reports singularity."""
        t = 10**10000
        curve = ShortWeierstrassCurve(
            coefficient_a=_pt(format_canonical_integer(-3 * t * t)),
            coefficient_b=_pt(format_canonical_integer(2 * t**3)),
        )
        result = compute_discriminant(EllipticCurveRequest(curve=curve))
        assert result.discriminant.as_fraction() == 0
        assert not result.is_nonsingular

    def test_near_cancellation_admitted_with_exact_value(self):
        """Large terms cancelling almost exactly stay admitted: with
        A=-3t², B=2t³+t the exact Δ = 108t⁴ + 27t² fits the bound."""
        t = 10**8000
        curve = ShortWeierstrassCurve(
            coefficient_a=_pt(format_canonical_integer(-3 * t * t)),
            coefficient_b=_pt(format_canonical_integer(2 * t**3 + t)),
        )
        result = compute_discriminant(EllipticCurveRequest(curve=curve))
        expected = -16 * (
            4 * Fraction(-3 * t * t) ** 3 + 27 * Fraction(2 * t**3 + t) ** 2
        )
        assert result.discriminant.as_fraction() == expected
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


def _operand(curve, point):
    """Wrap a rational affine point as the parent-bearing curve-point value."""
    return EllipticCurvePointResult(
        curve=curve,
        point=point,
        at_infinity=False,
    )


class TestPointAddition:
    def test_double_y_zero(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        point = RationalAffinePoint(x=_pt("0"), y=_pt("0"))
        result = add_points(
            EllipticCurvePointAdditionRequest(
                curve=curve, first=_operand(curve, point), second=_operand(curve, point)
            )
        )
        assert result.at_infinity

    def test_add_distinct_points(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = add_points(
            EllipticCurvePointAdditionRequest(
                curve=curve, first=_operand(curve, p), second=_operand(curve, p)
            )
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == Fraction(9, 4)
        assert result.point.y.as_fraction() == Fraction(-21, 8)

    def test_add_negatives(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        neg_p = RationalAffinePoint(x=_pt("2"), y=_pt("-2"))
        result = add_points(
            EllipticCurvePointAdditionRequest(
                curve=curve, first=_operand(curve, p), second=_operand(curve, neg_p)
            )
        )
        assert result.at_infinity


class TestScalarMultiplication:
    def test_zero_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=_operand(curve, p), scalar=0)
        )
        assert result.at_infinity

    def test_one_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=_operand(curve, p), scalar=1)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == 2
        assert result.point.y.as_fraction() == 2

    def test_two_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=_operand(curve, p), scalar=2)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == Fraction(9, 4)
        assert result.point.y.as_fraction() == Fraction(-21, 8)

    def test_three_times_point(self):
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=_operand(curve, p), scalar=3)
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
            ScalarMultiplicationRequest(curve=curve, point=_operand(curve, p), scalar=3)
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == 0
        assert result.point.y.as_fraction() == 0

    def test_point_off_curve_rejected(self):
        """(1,1) does not lie on y^2=x^3+x; the old code returned a fake sum."""
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("1"), y=_pt("1"))
        with pytest.raises(ValidationError) as exc_info:
            add_points(
                EllipticCurvePointAdditionRequest(
                    curve=curve, first=_operand(curve, p), second=_operand(curve, p)
                )
            )
        _assert_error_code(exc_info, "elliptic_curve.result_point_off_curve")

    def test_singular_curve_rejected(self):
        """y^2=x^3 has a cusp at the origin: discriminant zero."""
        curve = ShortWeierstrassCurve(coefficient_a=_pt("0"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("1"), y=_pt("1"))
        with pytest.raises(ValidationError) as exc_info:
            ScalarMultiplicationRequest(curve=curve, point=_operand(curve, p), scalar=2)
        _assert_error_code(exc_info, "elliptic_curve.singular_curve")

    def test_singular_curve_with_identity_operands_rejected(self):
        """The identity shortcut must not bypass nonsingularity: a singular
        curve is rejected even when every operand is the point at infinity."""
        curve = ShortWeierstrassCurve(coefficient_a=_pt("0"), coefficient_b=_pt("0"))
        identity = EllipticCurvePointResult(curve=curve, at_infinity=True)
        with pytest.raises(ValidationError) as exc_info:
            EllipticCurvePointAdditionRequest(
                curve=curve, first=identity, second=identity
            )
        _assert_error_code(exc_info, "elliptic_curve.singular_curve")
        with pytest.raises(ValidationError) as exc_info:
            ScalarMultiplicationRequest(curve=curve, point=identity, scalar=7)
        _assert_error_code(exc_info, "elliptic_curve.singular_curve")
        with pytest.raises(ValidationError) as exc_info:
            ScalarMultiplicationRequest(curve=curve, point=identity, scalar=0)
        _assert_error_code(exc_info, "elliptic_curve.singular_curve")

    def test_double_order_two_point_huge_denominator_admitted(self):
        """Doubling P=(1/q, 0) on y²=x³+x+B is O: admission must not
        propagate a fictional tangent slope through a y=0 doubling."""
        q = 10**4000 + 1
        curve = ShortWeierstrassCurve(
            coefficient_a=_pt("1"),
            # B = -(x³ + A·x) with x = 1/q, so P lies on the curve.
            coefficient_b={
                "num": format_canonical_integer(-(q * q + 1)),
                "den": format_canonical_integer(q**3),
            },
        )
        p = RationalAffinePoint(x=_pt("1", str(q)), y=_pt("0"))
        result = add_points(
            EllipticCurvePointAdditionRequest(
                curve=curve, first=_operand(curve, p), second=_operand(curve, p)
            )
        )
        assert result.at_infinity
        doubled = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=_operand(curve, p), scalar=2)
        )
        assert doubled.at_infinity

    def test_order_two_point_odd_scalar_admitted(self):
        """11P=P for order-two P=(0,0) on y²=x³+x; the height budget must
        track the infinity state instead of fabricating slope growth past
        the identity."""
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("0"), y=_pt("0"))
        result = scalar_multiply(
            ScalarMultiplicationRequest(
                curve=curve, point=_operand(curve, p), scalar=11
            )
        )
        assert result.point is not None
        assert result.point.x.as_fraction() == 0
        assert result.point.y.as_fraction() == 0


class TestResultSourceBinding:
    """Authoritative elliptic results are bound to their source curves."""

    def _curve(self, a: str = "1", b: str = "0"):
        return ShortWeierstrassCurve(
            coefficient_a=CanonicalRational(num=a, den="1"),
            coefficient_b=CanonicalRational(num=b, den="1"),
        )

    def test_discriminant_result_replays_the_retained_curve(self) -> None:
        request = EllipticCurveRequest(curve=self._curve())
        result = compute_discriminant(request)
        assert result.request == request
        reparsed = CurveDiscriminantResult.model_validate(
            result.model_dump(mode="json")
        )
        assert reparsed == result

        forged = dict(result.model_dump(mode="json"))
        # The thread's forgery: discriminant=1 for a curve whose exact
        # discriminant is -64.
        forged["discriminant"] = {"num": "1", "den": "1"}
        with pytest.raises(ValidationError) as exc_info:
            CurveDiscriminantResult.model_validate(forged)
        _assert_error_code(exc_info, "elliptic_curve.discriminant_source_mismatch")

    def test_point_on_curve_result_replays_the_predicate(self) -> None:
        from jacobian.math.elliptic_curves._models import (
            PointOnCurveResult as Result,
        )

        request = CurvePointRequest(
            curve=self._curve(),
            point=RationalAffinePoint(
                x=CanonicalRational(num="0", den="1"),
                y=CanonicalRational(num="0", den="1"),
            ),
        )
        result = check_point_on_curve(request)
        assert result.on_curve is True
        payload = result.model_dump(mode="json")
        assert Result.model_validate(payload).on_curve is True

        forged = dict(payload)
        forged["on_curve"] = False
        with pytest.raises(ValidationError) as exc_info:
            Result.model_validate(forged)
        _assert_error_code(exc_info, "elliptic_curve.point_membership_mismatch")

    def test_point_addition_result_retains_its_parent_curve(self) -> None:
        """Doubling (0,0) on y²=x³+x and on y²=x³-x must serialize
        differently: each result carries its parent curve."""
        origin = RationalAffinePoint(
            x=CanonicalRational(num="0", den="1"),
            y=CanonicalRational(num="0", den="1"),
        )
        first = add_points(
            EllipticCurvePointAdditionRequest(
                curve=self._curve("1", "0"),
                first=_operand(self._curve("1", "0"), origin),
                second=_operand(self._curve("1", "0"), origin),
            )
        )
        second = add_points(
            EllipticCurvePointAdditionRequest(
                curve=self._curve("-1", "0"),
                first=_operand(self._curve("-1", "0"), origin),
                second=_operand(self._curve("-1", "0"), origin),
            )
        )
        assert first.at_infinity and second.at_infinity
        assert first.curve != second.curve


class TestGroupLawComposition:
    """Group-law results compose into later group-law requests unchanged."""

    def test_infinity_result_feeds_addition_as_identity(self) -> None:
        curve = ShortWeierstrassCurve(coefficient_a=_pt("1"), coefficient_b=_pt("0"))
        origin = RationalAffinePoint(x=_pt("0"), y=_pt("0"))
        doubled = add_points(
            EllipticCurvePointAdditionRequest(
                curve=curve,
                first=_operand(curve, origin),
                second=_operand(curve, origin),
            )
        )
        assert doubled.at_infinity

        chained = add_points(
            EllipticCurvePointAdditionRequest(
                curve=curve,
                first=doubled,
                second=_operand(curve, origin),
            )
        )
        assert chained.at_infinity is False
        assert chained.point.x.as_fraction() == Fraction(0)

    def test_point_addition_result_feeds_scalar_multiply(self) -> None:
        curve = ShortWeierstrassCurve(coefficient_a=_pt("-2"), coefficient_b=_pt("0"))
        p = RationalAffinePoint(x=_pt("2"), y=_pt("2"))
        added = add_points(
            EllipticCurvePointAdditionRequest(
                curve=curve,
                first=_operand(curve, p),
                second=_operand(curve, p),
            )
        )
        result = scalar_multiply(
            ScalarMultiplicationRequest(curve=curve, point=added, scalar=2)
        )
        assert result.point is not None
