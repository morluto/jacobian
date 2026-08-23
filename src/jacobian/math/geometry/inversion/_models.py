"""Typed wire contracts for circle inversion of rational planar points."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry._models import RationalPoint2D as GeometryPoint


def _digit_parts(value: CanonicalRational) -> tuple[int, int]:
    fraction = value.as_fraction()
    return (
        len(format_canonical_integer(fraction.numerator).lstrip("-")),
        len(format_canonical_integer(fraction.denominator)),
    )


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p - c||²) * (p - c).
    """

    center: GeometryPoint = Field(
        description="Inversion center as a canonical geometry point"
    )
    power: CanonicalRational = Field(
        description="Positive rational inversion power (squared radius)"
    )
    point: GeometryPoint = Field(
        description="Point to invert as a canonical geometry point"
    )

    @model_validator(mode="after")
    def require_admissible_request(self) -> Self:
        if self.power.num == "0":
            raise ValueError("inversion power must be positive")
        if self.power.num.startswith("-"):
            raise ValueError("inversion power must be positive")
        # The contract requires p != c; inverting the center would divide by
        # the zero displacement, so reject it at this typed boundary.
        if (
            self.point.x.as_fraction() == self.center.x.as_fraction()
            and self.point.y.as_fraction() == self.center.y.as_fraction()
        ):
            raise ValueError("the inversion center cannot be inverted")
        for component in (
            self.center.x,
            self.center.y,
            self.power,
            self.point.x,
            self.point.y,
        ):
            require_bounded_rational(
                component,
                max_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="component",
            )
        # Admission must account for multiplicative growth: each output
        # coordinate multiplies the power by a displacement component over
        # the squared displacement norm. Derive a strict per-component digit
        # bound from the concrete request (no reduction credit is taken, so
        # these are upper bounds) and reject any request whose inverted
        # coordinates could exceed the canonical limit.
        cxn, cxd = _digit_parts(self.center.x)
        cyn, cyd = _digit_parts(self.center.y)
        pxn, pxd = _digit_parts(self.point.x)
        pyn, pyd = _digit_parts(self.point.y)
        sn, sd = _digit_parts(self.power)

        dxn = max(pxn + cxd, cxn + pxd) + 1
        dxd = pxd + cxd
        dyn = max(pyn + cyd, cyn + pyd) + 1
        dyd = pyd + cyd

        # ||dp||^2 over the common denominator dx^2 * dy^2.
        norm_num = max(2 * dxn + 2 * dyd, 2 * dyn + 2 * dxd) + 1
        norm_den = 2 * dxd + 2 * dyd

        # t = s / ||dp||^2.
        scale_num = sn + norm_den
        scale_den = sd + norm_num

        qxn = max(cxn + scale_den + dxd, scale_num + dxn + cxd) + 1
        qxd = cxd + scale_den + dxd
        qyn = max(cyn + scale_den + dyd, scale_num + dyn + cyd) + 1
        qyd = cyd + scale_den + dyd
        if max(qxn, qxd, qyn, qyd) > MAX_CANONICAL_RATIONAL_DIGITS:
            raise ValueError(
                "inversion result growth exceeds the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit limit; reduce the "
                "center, point, or power component magnitude"
            )
        return self


class CircleInversionResult(CircleInversionRequest):
    inverted_point: GeometryPoint
    complete: Literal[True] = True
    method: Literal["EXACT_RATIONAL_INVERSION"] = "EXACT_RATIONAL_INVERSION"

    @model_validator(mode="after")
    def bind_inversion(self) -> Self:
        from jacobian.math.geometry.inversion._operations import invert_point

        result = invert_point(
            self.center.x.as_fraction(),
            self.center.y.as_fraction(),
            self.power.as_fraction(),
            self.point.x.as_fraction(),
            self.point.y.as_fraction(),
        )
        expected = GeometryPoint(
            x=CanonicalRational.from_fraction(result[0]),
            y=CanonicalRational.from_fraction(result[1]),
        )
        if self.inverted_point != expected:
            raise ValueError("inverted_point must be the exact inversion result")
        return self


__all__ = [
    "CircleInversionRequest",
    "CircleInversionResult",
]
