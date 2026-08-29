"""Typed wire contracts for elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights

MAX_SCALAR = 10_000


class ShortWeierstrassCurve(StrictModel):
    """A short Weierstrass curve y^2 = x^3 + A*x + B over QQ."""

    coefficient_a: CanonicalRational
    coefficient_b: CanonicalRational

    def discriminant(self) -> Fraction:
        """Exact Δ = -16(4A³ + 27B²); zero marks a singular cubic."""
        a = self.coefficient_a.as_fraction()
        b = self.coefficient_b.as_fraction()
        return -16 * (4 * a**3 + 27 * b**2)


class EllipticCurveRequest(StrictModel):
    """Compute the discriminant of a short Weierstrass curve."""

    curve: ShortWeierstrassCurve


class CurveDiscriminantResult(StrictModel):
    """The discriminant Δ = -16(4A^3 + 27B^2) of a source curve."""

    curve: ShortWeierstrassCurve
    discriminant: CanonicalRational
    is_nonsingular: bool

    @classmethod
    def _from_kernel(
        cls,
        *,
        curve: ShortWeierstrassCurve,
        discriminant: CanonicalRational,
        is_nonsingular: bool,
    ) -> Self:
        return cls.model_construct(
            curve=curve,
            discriminant=discriminant,
            is_nonsingular=is_nonsingular,
        )


class RationalAffinePoint(StrictModel):
    """An affine rational point on an elliptic curve."""

    x: CanonicalRational
    y: CanonicalRational


class CurvePointRequest(StrictModel):
    """A curve and a point to check or operate on."""

    curve: ShortWeierstrassCurve
    point: RationalAffinePoint


class PointOnCurveResult(StrictModel):
    """Whether a point lies on a source curve."""

    curve: ShortWeierstrassCurve
    point: RationalAffinePoint
    on_curve: bool

    @classmethod
    def _from_kernel(
        cls,
        *,
        curve: ShortWeierstrassCurve,
        point: RationalAffinePoint,
        on_curve: bool,
    ) -> Self:
        return cls.model_construct(curve=curve, point=point, on_curve=on_curve)


def _require_group_law(
    curve: ShortWeierstrassCurve,
    points: tuple[RationalAffinePoint, ...],
) -> None:
    """Enforce the advertised group-law domain at the typed boundary.

    The chord-and-tangent formulas compute on the curve only when the cubic
    is nonsingular and every operand satisfies y² = x³ + Ax + B.
    """
    if curve.discriminant() == 0:
        raise PydanticCustomError(
            "elliptic_curve.singular_curve",
            "curve must be nonsingular (nonzero discriminant)",
        )
    for point in points:
        x = point.x.as_fraction()
        y = point.y.as_fraction()
        if y * y != x**3 + curve.coefficient_a.as_fraction() * x + (
            curve.coefficient_b.as_fraction()
        ):
            raise PydanticCustomError(
                "elliptic_curve.point_off_curve",
                "point must lie on the curve",
            )


def _generic_lambda_height_from_heights(
    first: tuple[RationalHeight, RationalHeight],
    second: tuple[RationalHeight, RationalHeight],
) -> RationalHeight:
    """Height bound of lambda = (y2 - y1) / (x2 - x1) with symbolic operands."""
    dy = sum_heights((second[1], first[1]))
    dx = sum_heights((second[0], first[0]))
    return dy.quotient(dx)


def _doubling_lambda_height_from_heights(
    curve: ShortWeierstrassCurve, point: tuple[RationalHeight, RationalHeight]
) -> RationalHeight:
    """Height bound of lambda = (3x^2 + A) / (2y) for symbolic coordinates."""
    x, y = point
    three_x_squared = RationalHeight(
        2 * x.numerator_digits + 1, 2 * x.denominator_digits
    )
    numerator = sum_heights(
        (three_x_squared, RationalHeight.from_canonical(curve.coefficient_a))
    )
    return numerator.quotient(
        RationalHeight(y.numerator_digits + 1, y.denominator_digits)
    )


def _generic_lambda_height(
    first: RationalAffinePoint, second: RationalAffinePoint
) -> RationalHeight:
    """Height bound of lambda = (y2 - y1) / (x2 - x1)."""
    dy = sum_heights(
        (
            RationalHeight.from_canonical(second.y),
            RationalHeight.from_canonical(first.y),
        )
    )
    dx = sum_heights(
        (
            RationalHeight.from_canonical(second.x),
            RationalHeight.from_canonical(first.x),
        )
    )
    return dy.quotient(dx)


def _doubling_lambda_height(
    curve: ShortWeierstrassCurve, point: RationalAffinePoint
) -> RationalHeight:
    """Height bound of lambda = (3x^2 + A) / (2y)."""
    x = RationalHeight.from_canonical(point.x)
    three_x_squared = RationalHeight(
        2 * x.numerator_digits + 1, 2 * x.denominator_digits
    )
    numerator = sum_heights(
        (three_x_squared, RationalHeight.from_canonical(curve.coefficient_a))
    )
    y = RationalHeight.from_canonical(point.y)
    return numerator.quotient(
        RationalHeight(y.numerator_digits + 1, y.denominator_digits)
    )


def _chord_step_heights(
    lam: RationalHeight,
    first: tuple[RationalHeight, RationalHeight],
    second: tuple[RationalHeight, RationalHeight],
) -> tuple[RationalHeight, RationalHeight]:
    """Conservative coordinate heights of one chord-and-tangent output.

    With lambda bounded by ``lam``, x3 = lambda^2 - x1 - x2 and
    y3 = lambda * (x1 - x3) - y1 propagate through rational-height sums,
    products, and quotients.
    """
    lam_squared = lam.product(lam)
    x3 = sum_heights((lam_squared, first[0], second[0]))
    inner = sum_heights((first[0], x3))
    y3 = sum_heights((lam.product(inner), first[1]))
    return (
        RationalHeight(
            max(x3.numerator_digits, y3.numerator_digits),
            max(x3.denominator_digits, y3.denominator_digits),
        ),
        y3,
    )


def _point_heights(point: RationalAffinePoint) -> tuple[RationalHeight, RationalHeight]:
    return (
        RationalHeight.from_canonical(point.x),
        RationalHeight.from_canonical(point.y),
    )


class EllipticCurvePointResult(StrictModel):
    """The result of an elliptic curve point operation on its parent curve.

    The parent curve defines the group the result lives in: without it,
    identical coordinate pairs on different curves serialize to the same
    value and callers cannot feed the point back into another group-law
    operation.
    """

    curve: ShortWeierstrassCurve
    point: RationalAffinePoint | None = None
    at_infinity: bool = False

    @model_validator(mode="after")
    def require_consistent_point(self) -> Self:
        # One canonical infinity discriminator: at_infinity. Accepting a
        # second independent flag would let one mathematical value
        # serialize several ways and let downstream at_infinity readers
        # misread a validated infinity as finite-with-no-point.
        if self.point is not None and self.at_infinity:
            raise PydanticCustomError(
                "elliptic_curve.point_infinity_conflict",
                "a finite point and infinity are mutually exclusive",
            )
        if self.point is None and not self.at_infinity:
            raise PydanticCustomError(
                "elliptic_curve.point_missing",
                "must carry a finite point or indicate infinity",
            )
        if self.point is not None:
            x = self.point.x.as_fraction()
            y = self.point.y.as_fraction()
            a = self.curve.coefficient_a.as_fraction()
            b = self.curve.coefficient_b.as_fraction()
            if y * y != x**3 + a * x + b:
                raise PydanticCustomError(
                    "elliptic_curve.result_point_off_curve",
                    "result point must lie on the retained curve",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        curve: ShortWeierstrassCurve,
        point: RationalAffinePoint | None,
    ) -> Self:
        """Construct a point already established by the exact group-law kernel."""

        return cls.model_construct(
            curve=curve,
            point=point,
            at_infinity=point is None,
        )


class EllipticCurvePointAdditionRequest(StrictModel):
    """Add two points on a short Weierstrass elliptic curve.

    Both operands are parent-bearing curve-point values — exactly the shape
    the group-law producers return — so a doubling example's infinity result
    or any finite point result composes into this request unchanged.
    """

    curve: ShortWeierstrassCurve
    first: EllipticCurvePointResult
    second: EllipticCurvePointResult


class ScalarMultiplicationRequest(StrictModel):
    """Compute n*P on a short Weierstrass elliptic curve."""

    curve: ShortWeierstrassCurve
    point: EllipticCurvePointResult
    scalar: int = Field(ge=0, le=MAX_SCALAR)


class ScalarMultiplicationResult(EllipticCurvePointResult):
    """The result of scalar multiplication n*P on its retained parent curve.

    The shared parent-bearing curve-point value: a scalar-multiplication
    result passes unchanged as the operand of any later group-law request.
    """


# Membership replay is inherited from EllipticCurvePointResult's validator.
ScalarMultiplicationResult.model_rebuild()


__all__ = [
    "CurveDiscriminantResult",
    "CurvePointRequest",
    "EllipticCurvePointAdditionRequest",
    "EllipticCurvePointResult",
    "EllipticCurveRequest",
    "PointOnCurveResult",
    "RationalAffinePoint",
    "ScalarMultiplicationRequest",
    "ScalarMultiplicationResult",
    "ShortWeierstrassCurve",
]
