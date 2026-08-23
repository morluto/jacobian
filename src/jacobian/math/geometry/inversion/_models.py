"""Typed wire contracts for circle inversion of rational planar points."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math._rational_height import RationalHeight
from jacobian.math.geometry._models import RationalPoint2D

# Admission keeps every accepted request inside the domain that its own
# output can re-enter: inputs and the EXACT inverted coordinates must stay
# within half the canonical limit, so feeding an accepted result back is
# itself an admissible request and the advertised involution I(I(p)) = p
# holds inside the admitted domain. Because the check evaluates the exact
# image instead of a cancellation-blind height estimate, a point such as
# (1/(10^1400+1), 1/(10^1400+3)) whose inverse (q1*q2^2/(q1^2+q2^2), ...)
# only looks tall before reduction is admitted together with its exact
# inverse.
_HALF_CANONICAL_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS // 2


def _require_reusable_height(rational: CanonicalRational) -> None:
    if RationalHeight.from_canonical(rational).exceeds(_HALF_CANONICAL_DIGITS):
        raise ValueError(
            "circle inversion values must stay within the "
            f"{_HALF_CANONICAL_DIGITS}-digit reusable admission bound"
        )


def _require_reusable_fraction_height(value) -> None:
    """Bound an exact Fraction without constructing its canonical value."""
    height = RationalHeight(
        len(format_canonical_integer(value.numerator).lstrip("-")),
        len(format_canonical_integer(value.denominator)),
    )
    if height.exceeds(_HALF_CANONICAL_DIGITS):
        raise ValueError(
            "circle inversion values must stay within the "
            f"{_HALF_CANONICAL_DIGITS}-digit reusable admission bound"
        )


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p-c||²) * (p - c).
    """

    center: RationalPoint2D
    power: CanonicalRational = Field(
        description="Positive rational inversion power (squared radius)"
    )
    point: RationalPoint2D

    @model_validator(mode="after")
    def require_admissible_request(self) -> Self:
        if self.power.num == "0" or self.power.num.startswith("-"):
            raise ValueError("inversion power must be positive")
        # The contract requires p != c; inverting the center would divide by
        # the zero displacement, so reject it at this typed boundary.
        if (
            self.point.x.as_fraction() == self.center.x.as_fraction()
            and self.point.y.as_fraction() == self.center.y.as_fraction()
        ):
            raise ValueError("the inversion center cannot be inverted")

        for rational in (
            self.center.x,
            self.center.y,
            self.point.x,
            self.point.y,
            self.power,
        ):
            _require_reusable_height(rational)

        # Closure under the defining involution is checked on the EXACT
        # image: the inputs above are bounded, so computing it exactly is
        # bounded work, and no cancellation-blind estimate can reject a
        # feed-back of this result.
        from jacobian.math.geometry.inversion._operations import invert_point

        image_x, image_y = invert_point(
            self.center.x.as_fraction(),
            self.center.y.as_fraction(),
            self.power.as_fraction(),
            self.point.x.as_fraction(),
            self.point.y.as_fraction(),
        )
        _require_reusable_fraction_height(image_x)
        _require_reusable_fraction_height(image_y)
        return self

    @property
    def _image(self) -> tuple[CanonicalRational, CanonicalRational]:
        from jacobian.math.geometry.inversion._operations import invert_point

        image_x, image_y = invert_point(
            self.center.x.as_fraction(),
            self.center.y.as_fraction(),
            self.power.as_fraction(),
            self.point.x.as_fraction(),
            self.point.y.as_fraction(),
        )
        return (
            CanonicalRational.from_fraction(image_x),
            CanonicalRational.from_fraction(image_y),
        )


class CircleInversionResult(CircleInversionRequest):
    """The exact image of ``point`` under I_{center, power}.

    The image carries the geometry-domain canonical point value so the
    serialized result composes unchanged into every planar-geometry
    operation that accepts a RationalPoint2D.
    """

    image: RationalPoint2D
    complete: Literal[True] = True
    method: Literal["EXACT_RATIONAL_INVERSION"] = "EXACT_RATIONAL_INVERSION"

    @model_validator(mode="after")
    def bind_inversion(self) -> Self:
        expected_x, expected_y = self._image
        if self.image.x != expected_x or self.image.y != expected_y:
            raise ValueError("image must be the exact inversion of the point")
        return self


__all__ = [
    "CircleInversionRequest",
    "CircleInversionResult",
]
