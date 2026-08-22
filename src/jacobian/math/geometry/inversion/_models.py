"""Typed wire contracts for circle inversion of rational planar points."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_INVERSION_COORDINATE_DIGITS = 8192


class RationalPoint2D(StrictModel):
    x: CanonicalRational
    y: CanonicalRational


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
        for label, value in (
            ("center_x", self.center_x),
            ("center_y", self.center_y),
            ("power", self.power),
            ("point_x", self.point_x),
            ("point_y", self.point_y),
        ):
            require_bounded_rational(
                value, max_digits=MAX_INVERSION_COORDINATE_DIGITS, label=label
            )
        # Conservatively bound the exact inversion output: for 20k-digit N,
        # power N and point 1/N produce output N^2 with ~40k digits, exceeding
        # CanonicalRational's 32,768-digit limit and leaking a host
        # validation exception. Checking the closed-form here keeps every
        # accepted request inside the declared result type.
        from jacobian.math.geometry.inversion._operations import invert_point

        try:
            qx, qy = invert_point(
                self.center_x.as_fraction(),
                self.center_y.as_fraction(),
                self.power.as_fraction(),
                self.point_x.as_fraction(),
                self.point_y.as_fraction(),
            )
            CanonicalRational.from_fraction(qx)
            CanonicalRational.from_fraction(qy)
        except ValueError as exc:
            raise ValueError(f"inversion output exceeds the canonical digit bound: {exc}") from exc
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
