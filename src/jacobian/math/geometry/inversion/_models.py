"""Typed wire contracts for circle inversion of rational planar points."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.geometry._models import RationalPoint2D


def _displacement_height(
    left: CanonicalRational, right: CanonicalRational
) -> RationalHeight:
    return sum_heights(
        (RationalHeight.from_canonical(left), RationalHeight.from_canonical(right))
    )


def _inversion_result_heights(
    center: RationalPoint2D,
    power: CanonicalRational,
    point: RationalPoint2D,
) -> tuple[RationalHeight, RationalHeight]:
    """Conservative height of I(p) = c + s(p - c)/||p - c||^2 before reduction.

    The admitted domain must be symmetric under unit inversion so every
    accepted result can be consumed unchanged: inputs are admitted only within
    half the canonical limit and estimated result heights must stay inside the
    same half-limit domain, so I(I(p)) for an admitted p is again admitted.
    """

    dx = _displacement_height(point.x, center.x)
    dy = _displacement_height(point.y, center.y)
    norm_squared = sum_heights((dx.product(dx), dy.product(dy)))
    scale = RationalHeight.from_canonical(power).quotient(norm_squared)
    inverted_x = sum_heights(
        (RationalHeight.from_canonical(center.x), scale.product(dx))
    )
    inverted_y = sum_heights(
        (RationalHeight.from_canonical(center.y), scale.product(dy))
    )
    return inverted_x, inverted_y


_HALF_CANONICAL_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS // 2


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p - c||²) * (p - c).
    """

    center: RationalPoint2D = Field(description="the inversion center")
    power: CanonicalRational = Field(description="Positive rational inversion power (squared radius)")
    point: RationalPoint2D = Field(description="the point to invert")

    @model_validator(mode="after")
    def require_admissible_request(self) -> Self:
        if self.power.num == "0" or self.power.num.startswith("-"):
            raise ValueError("inversion power must be positive")
        # The contract requires p != c; inverting the center would divide by
        # the zero displacement, so reject it at this typed boundary.
        if self.point == self.center:
            raise ValueError("the inversion center cannot be inverted")

        # Admit only inputs whose own height is at most half the canonical
        # limit.  Unit inversion squares digit counts, so I(I(p)) for an
        # admitted p is again admitted: the domain is symmetric under the
        # advertised involution and every accepted result can be fed back.
        for rational in (
            *(
                component
                for value in (self.center, self.point)
                for component in (value.x, value.y)
            ),
            self.power,
        ):
            height = RationalHeight.from_canonical(rational)
            if height.exceeds(_HALF_CANONICAL_DIGITS):
                raise ValueError(
                    "circle inversion inputs must stay within the "
                    f"{_HALF_CANONICAL_DIGITS}-digit symmetric admission bound"
                )

        inverted_x, inverted_y = _inversion_result_heights(
            self.center, self.power, self.point
        )
        if inverted_x.exceeds(_HALF_CANONICAL_DIGITS) or inverted_y.exceeds(
            _HALF_CANONICAL_DIGITS
        ):
            raise ValueError(
                "circle inversion results must stay within the "
                f"{_HALF_CANONICAL_DIGITS}-digit reusable admission domain"
            )
        return self


class CircleInversionResult(CircleInversionRequest):
    inverted_point: RationalPoint2D = Field(description="the exact inverted point")
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
        expected_x = CanonicalRational.from_fraction(result[0])
        expected_y = CanonicalRational.from_fraction(result[1])
        if self.inverted_point.x != expected_x or self.inverted_point.y != expected_y:
            raise ValueError(
                "inverted_point must be the exact circle-inversion image"
            )
        return self


__all__ = [
    "CircleInversionRequest",
    "CircleInversionResult",
]
