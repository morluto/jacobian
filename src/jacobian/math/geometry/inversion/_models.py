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


def _displacement_height(
    left: CanonicalRational, right: CanonicalRational
) -> RationalHeight:
    return sum_heights(
        (RationalHeight.from_canonical(left), RationalHeight.from_canonical(right))
    )


def _inversion_result_heights(
    center_x: CanonicalRational,
    center_y: CanonicalRational,
    power: CanonicalRational,
    point_x: CanonicalRational,
    point_y: CanonicalRational,
) -> tuple[RationalHeight, RationalHeight]:
    """Conservative height of I(p) = c + s(p - c)/||p - c||^2 before reduction.

    The admitted domain must be symmetric under unit inversion so every
    accepted result can be consumed unchanged.  For the origin-centered unit
    inversion, I(I(p)) == p exactly, and each application squares
    numerator/denominator digit counts; bounding the input height by half the
    canonical limit makes two successive accepted invocations stay within one
    canonical limit, which the squaring growth dominates.
    """

    dx = _displacement_height(point_x, center_x)
    dy = _displacement_height(point_y, center_y)
    norm_squared = sum_heights((dx.product(dx), dy.product(dy)))
    scale = RationalHeight.from_canonical(power).quotient(norm_squared)
    inverted_x = sum_heights(
        (RationalHeight.from_canonical(center_x), scale.product(dx))
    )
    inverted_y = sum_heights(
        (RationalHeight.from_canonical(center_y), scale.product(dy))
    )
    return inverted_x, inverted_y


_HALF_CANONICAL_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS // 2


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p - c||²) * (p - c).
    """

    center_x: CanonicalRational = Field(
        description="x-coordinate of the inversion center"
    )
    center_y: CanonicalRational = Field(
        description="y-coordinate of the inversion center"
    )
    power: CanonicalRational = Field(
        description="Positive rational inversion power (squared radius)"
    )
    point_x: CanonicalRational = Field(
        description="x-coordinate of the point to invert"
    )
    point_y: CanonicalRational = Field(
        description="y-coordinate of the point to invert"
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
            self.point_x.as_fraction() == self.center_x.as_fraction()
            and self.point_y.as_fraction() == self.center_y.as_fraction()
        ):
            raise ValueError("the inversion center cannot be inverted")

        # Admit only inputs whose own height is at most half the canonical
        # limit.  Unit inversion squares digit counts, so I(I(p)) for an
        # admitted p is again admitted: the domain is symmetric under the
        # advertised involution and every accepted result can be fed back.
        for rational in (
            self.center_x,
            self.center_y,
            self.point_x,
            self.point_y,
            self.power,
        ):
            height = RationalHeight.from_canonical(rational)
            if height.exceeds(_HALF_CANONICAL_DIGITS):
                raise ValueError(
                    "circle inversion inputs must stay within the "
                    f"{_HALF_CANONICAL_DIGITS}-digit symmetric admission bound"
                )

        inverted_x, inverted_y = _inversion_result_heights(
            self.center_x, self.center_y, self.power, self.point_x, self.point_y
        )
        # Outputs are checked against the same reusable admission bound as
        # inputs, not the full canonical limit: every coordinate supplied to
        # a subsequent inversion is rejected above half that limit, so a
        # result admitted here must be feedable back into the advertised
        # involution unchanged.
        if inverted_x.exceeds(_HALF_CANONICAL_DIGITS) or inverted_y.exceeds(
            _HALF_CANONICAL_DIGITS
        ):
            raise ValueError(
                "circle inversion result exceeds the "
                f"{_HALF_CANONICAL_DIGITS}-digit reusable admission bound; "
                "the exact output would be rejected as input to the next inversion"
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
