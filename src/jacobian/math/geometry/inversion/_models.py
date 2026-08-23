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
from jacobian.math.geometry._models import RationalPoint2D

_MAX_INVERSION_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS // 2


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p - c||²) * (p - c).

    The request uses the geometry-owned ``RationalPoint2D`` so that points
    produced by any planar-geometry operation can enter inversion unchanged
    and vice versa.
    """

    center: RationalPoint2D = Field(description="Inversion center c")
    power: CanonicalRational = Field(
        description="Positive rational inversion power (squared radius)"
    )
    point: RationalPoint2D = Field(description="Point p != c to invert")

    @model_validator(mode="after")
    def require_admissible_request(self) -> Self:
        if self.power.num == "0":
            raise ValueError("inversion power must be positive")
        if self.power.num.startswith("-"):
            raise ValueError("inversion power must be positive")
        if self.point == self.center:
            raise ValueError("the inversion center cannot be inverted")
        for value, label in (
            (self.center.x, "center.x"),
            (self.center.y, "center.y"),
            (self.power, "power"),
            (self.point.x, "point.x"),
            (self.point.y, "point.y"),
        ):
            require_bounded_rational(
                value, max_digits=_MAX_INVERSION_DIGITS, label=label
            )
        # Keep the domain closed under the advertised involution: every
        # accepted result must itself be admissible, so compute the exact
        # inversion and reject if the derived coordinates would exceed the
        # same admission bound.

        cx, cy = self.center.x.as_fraction(), self.center.y.as_fraction()
        s = self.power.as_fraction()
        px, py = self.point.x.as_fraction(), self.point.y.as_fraction()
        dx = px - cx
        dy = py - cy
        norm_sq = dx * dx + dy * dy
        qx = cx + s * dx / norm_sq
        qy = cy + s * dy / norm_sq
        for frac, name in ((qx, "inverted point x"), (qy, "inverted point y")):
            num_digits = len(format_canonical_integer(frac.numerator))
            den_digits = len(format_canonical_integer(frac.denominator))
            if max(num_digits, den_digits) > _MAX_INVERSION_DIGITS:
                raise ValueError(
                    f"derived {name} exceeds the {_MAX_INVERSION_DIGITS}-digit "
                    "symmetric admission bound; the input is outside the "
                    "closed inversion domain"
                )
        return self


class CircleInversionResult(CircleInversionRequest):
    """The exact inverted point as the domain-canonical geometry point value."""

    inverted_point: RationalPoint2D
    complete: Literal[True] = True
    method: Literal["EXACT_RATIONAL_INVERSION"] = "EXACT_RATIONAL_INVERSION"

    @model_validator(mode="after")
    def bind_inversion(self) -> Self:
        from jacobian.math.geometry.inversion._operations import invert_point

        cx, cy = self.center.x.as_fraction(), self.center.y.as_fraction()
        s = self.power.as_fraction()
        px, py = self.point.x.as_fraction(), self.point.y.as_fraction()

        result = invert_point(cx, cy, s, px, py)
        expected = RationalPoint2D(
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
