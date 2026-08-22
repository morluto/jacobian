"""Typed wire contracts for circle inversion of rational planar points."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights


class RationalPoint2D(StrictModel):
    x: CanonicalRational
    y: CanonicalRational


def _inversion_height_bound_ok(
    center_x: CanonicalRational,
    center_y: CanonicalRational,
    power: CanonicalRational,
    point_x: CanonicalRational,
    point_y: CanonicalRational,
) -> bool:
    """Conservative admission for inversion growth.

    Uses RationalHeight to estimate result digits of
    q = c + (s/||p-c||^2)*(p-c).  Requires that both inverted coordinates
    stay within MAX_CANONICAL_RATIONAL_DIGITS. Also enforces half-digit
    input bound so domain is symmetric under involution.
    """
    half = MAX_CANONICAL_RATIONAL_DIGITS // 2
    for v in (center_x, center_y, point_x, point_y, power):
        if RationalHeight.from_canonical(v).exceeds(half):
            return False
    # Estimate heights
    def _disp(a: CanonicalRational, b: CanonicalRational) -> RationalHeight:
        return sum_heights((RationalHeight.from_canonical(a), RationalHeight.from_canonical(b)))
    dx = _disp(point_x, center_x)
    dy = _disp(point_y, center_y)
    norm2 = sum_heights((dx.product(dx), dy.product(dy)))
    scale = RationalHeight.from_canonical(power).quotient(norm2)
    hx = sum_heights((RationalHeight.from_canonical(center_x), scale.product(dx)))
    hy = sum_heights((RationalHeight.from_canonical(center_y), scale.product(dy)))
    return not hx.exceeds(MAX_CANONICAL_RATIONAL_DIGITS) and not hy.exceeds(MAX_CANONICAL_RATIONAL_DIGITS)


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p - c||²) * (p - c).
    """

    center_x: CanonicalRational = Field(description="x-coordinate of the inversion center")
    center_y: CanonicalRational = Field(description="y-coordinate of the inversion center")
    power: CanonicalRational = Field(description="Positive rational inversion power (squared radius)")
    point_x: CanonicalRational = Field(description="x-coordinate of the point to invert")
    point_y: CanonicalRational = Field(description="y-coordinate of the point to invert")

    @model_validator(mode="after")
    def require_admissible_request(self) -> Self:
        if self.power.num == "0":
            raise ValueError("inversion power must be positive")
        if self.power.num.startswith("-"):
            raise ValueError("inversion power must be positive")
        # The contract requires p != c; inverting the center would divide by
        # the zero displacement, so reject it at this typed boundary.
        if (
            self.point_x.as_fraction() == self.center_x.as_fraction()
            and self.point_y.as_fraction() == self.center_y.as_fraction()
        ):
            raise ValueError("the inversion center cannot be inverted")
        if not _inversion_height_bound_ok(
            self.center_x, self.center_y, self.power, self.point_x, self.point_y
        ):
            raise ValueError(
                "circle inversion inputs exceed the conservative height bound; "
                f"each coordinate/power must be within {MAX_CANONICAL_RATIONAL_DIGITS//2} digits and result within {MAX_CANONICAL_RATIONAL_DIGITS} digits"
            )
        return self


class CircleInversionResult(CircleInversionRequest):
    inverted_x: CanonicalRational
    inverted_y: CanonicalRational
    complete: Literal[True] = True
    method: Literal["EXACT_RATIONAL_INVERSION"] = "EXACT_RATIONAL_INVERSION"

    @model_validator(mode="after")
    def bind_inversion(self) -> Self:
        from jacobian.math.geometry.inversion._operations import invert_point

        cx, cy = self.center_x.as_fraction(), self.center_y.as_fraction()
        s = self.power.as_fraction()
        px, py = self.point_x.as_fraction(), self.point_y.as_fraction()

        result = invert_point(cx, cy, s, px, py)
        expected_x = CanonicalRational.from_fraction(result[0])
        expected_y = CanonicalRational.from_fraction(result[1])
        if self.inverted_x != expected_x:
            raise ValueError("inverted_x must be the exact inversion result")
        if self.inverted_y != expected_y:
            raise ValueError("inverted_y must be the exact inversion result")
        return self


__all__ = [
    "CircleInversionRequest",
    "CircleInversionResult",
    "RationalPoint2D",
]
