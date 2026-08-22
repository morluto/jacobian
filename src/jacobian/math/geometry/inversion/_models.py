"""Typed wire contracts for circle inversion of rational planar points."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.geometry._models import RationalPoint2D


def _inversion_height_bound_ok(
    center: RationalPoint2D,
    power: CanonicalRational,
    point: RationalPoint2D,
) -> bool:
    """Conservative admission for inversion growth.

    Uses RationalHeight to estimate result digits of
    q = c + (s/||p-c||^2)*(p-c).  Requires that both inverted coordinates
    stay within MAX_CANONICAL_RATIONAL_DIGITS. Also enforces half-digit
    input bound so domain is symmetric under involution.
    """
    # Closure under involution: an admitted input's image must also be
    # admissible. The estimator's worst-case output height grows like
    # 4H + slack, so inputs are capped at a quarter of the canonical
    # limit; requests whose estimated output still exceeds the limit are
    # rejected by the explicit hx/hy check below.
    half = MAX_CANONICAL_RATIONAL_DIGITS // 4
    for value in (center.x, center.y, point.x, point.y, power):
        if RationalHeight.from_canonical(value).exceeds(half):
            return False

    # Estimate heights
    def _disp(a: CanonicalRational, b: CanonicalRational) -> RationalHeight:
        return sum_heights(
            (RationalHeight.from_canonical(a), RationalHeight.from_canonical(b))
        )

    dx = _disp(point.x, center.x)
    dy = _disp(point.y, center.y)
    norm2 = sum_heights((dx.product(dx), dy.product(dy)))
    scale = RationalHeight.from_canonical(power).quotient(norm2)
    hx = sum_heights((RationalHeight.from_canonical(center.x), scale.product(dx)))
    hy = sum_heights((RationalHeight.from_canonical(center.y), scale.product(dy)))
    return not hx.exceeds(MAX_CANONICAL_RATIONAL_DIGITS) and not hy.exceeds(
        MAX_CANONICAL_RATIONAL_DIGITS
    )


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p - c||²) * (p - c).
    Points use the geometry domain's canonical ``RationalPoint2D`` value so
    serialized geometry results compose here unchanged.
    """

    center: RationalPoint2D = Field(description="The inversion center.")
    power: CanonicalRational = Field(
        description="Positive rational inversion power (squared radius)"
    )
    point: RationalPoint2D = Field(description="The point to invert.")

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
        if not _inversion_height_bound_ok(self.center, self.power, self.point):
            raise ValueError(
                "circle inversion inputs exceed the conservative height bound; "
                f"each coordinate/power must be within {MAX_CANONICAL_RATIONAL_DIGITS // 2} digits and result within {MAX_CANONICAL_RATIONAL_DIGITS} digits"
            )
        return self


class CircleInversionResult(CircleInversionRequest):
    inverted: RationalPoint2D
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
        expected = RationalPoint2D(
            x=CanonicalRational.from_fraction(result[0]),
            y=CanonicalRational.from_fraction(result[1]),
        )
        if self.inverted != expected:
            raise ValueError("inverted must be the exact inversion result")
        return self


__all__ = [
    "CircleInversionRequest",
    "CircleInversionResult",
]
