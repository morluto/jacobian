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

MAX_INVERSION_COMPONENT_DIGITS = 4_096


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
        for value, label in (
            (self.center_x, "center_x"),
            (self.center_y, "center_y"),
            (self.power, "power"),
            (self.point_x, "point_x"),
            (self.point_y, "point_y"),
        ):
            require_bounded_rational(
                value, max_digits=MAX_INVERSION_COMPONENT_DIGITS, label=label
            )
        # Conservative derived-output bound: the exact inversion result must
        # fit in CanonicalRational (<=32768 digits). The computation
        # q = c + s*(p-c)/|p-c|^2 can at most quadruple component digit size;
        # with each input capped at 4096 digits the result stays below the
        # limit, and we verify explicitly with the chunked canonical formatter,
        # which is exempt from CPython's int-to-str conversion limit.
        cx, cy = self.center_x.as_fraction(), self.center_y.as_fraction()
        s = self.power.as_fraction()
        px, py = self.point_x.as_fraction(), self.point_y.as_fraction()
        dx = px - cx
        dy = py - cy
        norm_sq = dx * dx + dy * dy
        # norm_sq >0 already (p != c)
        qx = cx + s * dx / norm_sq
        qy = cy + s * dy / norm_sq
        for frac, name in ((qx, "inverted_x"), (qy, "inverted_y")):
            num_digits = len(format_canonical_integer(frac.numerator))
            den_digits = len(format_canonical_integer(frac.denominator))
            if max(num_digits, den_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
                raise ValueError(f"derived {name} exceeds the canonical digit limit")
        return self


class CircleInversionResult(CircleInversionRequest):
    """The exact inverted point as the domain-canonical geometry point value.

    ``point`` is the shared ``RationalPoint2D`` so the result composes with
    every other planar-geometry operation unchanged.
    """

    point: RationalPoint2D
    complete: Literal[True] = True
    method: Literal["EXACT_RATIONAL_INVERSION"] = "EXACT_RATIONAL_INVERSION"

    @model_validator(mode="after")
    def bind_inversion(self) -> Self:
        from jacobian.math.geometry.inversion._operations import invert_point

        cx, cy = self.center_x.as_fraction(), self.center_y.as_fraction()
        s = self.power.as_fraction()
        px, py = self.point_x.as_fraction(), self.point_y.as_fraction()

        result = invert_point(cx, cy, s, px, py)
        expected = RationalPoint2D(
            x=CanonicalRational.from_fraction(result[0]),
            y=CanonicalRational.from_fraction(result[1]),
        )
        if self.point != expected:
            raise ValueError("point must be the exact inversion result")
        return self


__all__ = [
    "CircleInversionRequest",
    "CircleInversionResult",
]
