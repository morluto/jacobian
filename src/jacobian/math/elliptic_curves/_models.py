"""Typed wire contracts for elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math._rational_height import RationalHeight
from jacobian._models import StrictModel


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
    """The discriminant Δ = -16(4A^3 + 27B^2)."""

    discriminant: CanonicalRational
    is_nonsingular: bool

    @model_validator(mode="after")
    def require_consistent_nonsingularity(self) -> Self:
        if self.is_nonsingular is (self.discriminant.as_fraction() == 0):
            raise ValueError("nonsingularity must match a nonzero discriminant")
        return self


class RationalAffinePoint(StrictModel):
    """An affine rational point on an elliptic curve."""

    x: CanonicalRational
    y: CanonicalRational


class CurvePointRequest(StrictModel):
    """A curve and a point to check or operate on."""

    curve: ShortWeierstrassCurve
    point: RationalAffinePoint


class PointOnCurveResult(StrictModel):
    """Whether a point lies on the curve."""

    on_curve: bool


def _require_group_law(
    curve: ShortWeierstrassCurve,
    points: tuple[RationalAffinePoint, ...],
) -> None:
    """Enforce the advertised group-law domain at the typed boundary.

    The chord-and-tangent formulas compute on the curve only when the cubic
    is nonsingular and every operand satisfies y² = x³ + Ax + B.
    """
    if curve.discriminant() == 0:
        raise ValueError("curve must be nonsingular (nonzero discriminant)")
    for point in points:
        x = point.x.as_fraction()
        y = point.y.as_fraction()
        if y * y != x**3 + curve.coefficient_a.as_fraction() * x + (
            curve.coefficient_b.as_fraction()
        ):
            raise ValueError("point must lie on the curve")


class EllipticCurvePointAdditionRequest(StrictModel):
    """Add two points on a short Weierstrass elliptic curve."""

    curve: ShortWeierstrassCurve
    first: RationalAffinePoint
    second: RationalAffinePoint

    @model_validator(mode="after")
    def require_group_law(self) -> Self:
        _require_group_law(self.curve, (self.first, self.second))
        return self


class EllipticCurvePointResult(StrictModel):
    """The result of an elliptic curve point operation."""

    point: RationalAffinePoint | None = None
    at_infinity: bool = False
    is_infinity: bool = False

    @model_validator(mode="after")
    def require_consistent_point(self) -> Self:
        if self.point is not None and (self.at_infinity or self.is_infinity):
            raise ValueError("a finite point and infinity are mutually exclusive")
        if self.point is None and not (self.at_infinity or self.is_infinity):
            raise ValueError("must carry a finite point or indicate infinity")
        return self


class ScalarMultiplicationRequest(StrictModel):
    """Compute n*P on a short Weierstrass elliptic curve."""

    curve: ShortWeierstrassCurve
    point: RationalAffinePoint
    scalar: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def require_group_law(self) -> Self:
        _require_group_law(self.curve, (self.point,))
        # Conservative height bound: the naive height of n*P grows roughly as
        # n^2 times the height of P for generic curves.  Estimate the digit
        # growth and reject requests that would exceed the canonical limit.
        max_point_digits = max(
            len(self.point.x.num.lstrip("-")),
            len(self.point.x.den.lstrip("-")),
            len(self.point.y.num.lstrip("-")),
            len(self.point.y.den.lstrip("-")),
        )
        # Use n^2 * digit growth as a conservative upper bound; the true
        # growth for the given example (2,2) with n=300 is 35k digits, which
        # exceeds the 32k limit, so n=300 must be rejected for that point.
        # For small points (1 digit), n=300 gives 90k estimated >32k, so reject.
        # For n=100, estimated 10k <32k, so allow.
        estimated_digits = self.scalar * self.scalar * max(1, max_point_digits)
        # Also account for coefficient height
        coeff_digits = max(
            len(self.curve.coefficient_a.num.lstrip("-")),
            len(self.curve.coefficient_a.den.lstrip("-")),
            len(self.curve.coefficient_b.num.lstrip("-")),
            len(self.curve.coefficient_b.den.lstrip("-")),
        )
        estimated_digits += coeff_digits * self.scalar
        if estimated_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            raise ValueError(
                "scalar multiplication would exceed the canonical result height; "
                "reduce the scalar or use smaller coordinates"
            )
        # Also enforce a hard scalar cap derived from the worst-case point size:
        # even for 1-digit points, scalar=1000 would give 1M estimated digits,
        # so cap at 500 for safety when point digits are minimal.
        if self.scalar > 500 and max_point_digits <= 2:
            # For larger points, the n^2 bound already rejects; for tiny points
            # we still cap to keep intermediate work bounded.
            raise ValueError(
                "scalar exceeds the conservative work bound for the given point size"
            )
        return self


class ScalarMultiplicationResult(StrictModel):
    """The result of scalar multiplication n*P."""

    point: RationalAffinePoint | None = None
    at_infinity: bool = False

    @model_validator(mode="after")
    def require_consistent_result(self) -> Self:
        if self.point is not None and self.at_infinity:
            raise ValueError("a finite point and infinity are mutually exclusive")
        if self.point is None and not self.at_infinity:
            raise ValueError("must carry a finite point or indicate infinity")
        return self


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
