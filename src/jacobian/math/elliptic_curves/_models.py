"""Typed wire contracts for elliptic curve operations over QQ."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class ShortWeierstrassCurve(StrictModel):
    """A short Weierstrass curve y^2 = x^3 + A*x + B over QQ."""

    coefficient_a: CanonicalRational
    coefficient_b: CanonicalRational


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


class EllipticCurvePointAdditionRequest(StrictModel):
    """Add two points on a short Weierstrass elliptic curve."""

    curve: ShortWeierstrassCurve
    first: RationalAffinePoint
    second: RationalAffinePoint


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
