"""Typed wire contracts for circle inversion of rational planar points."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.geometry._models import RationalPoint2D as GeometryPoint

# Inversion q = c + (s / ||p-c||^2) * (p - c) grows rational components by a
# derivable factor. With every input numerator and denominator bounded at D
# digits: a coordinate difference has <= 2D+1 digits over 2D; the squared
# norm <= 8D+3 over 8D; the scale <= 9D digits over 9D+3; the scaled
# displacement <= 11D+2 over 11D+3; and each output component stays within
# 12D+4 digits. Requiring 12*2730+4 = 32,764 <= 32,768 keeps every accepted
# inversion inside CanonicalRational's canonical limit, so admission here
# cannot turn into a result-construction failure.
MAX_INVERSION_INPUT_DIGITS = 2_730


class CircleInversionRequest(StrictModel):
    """Compute the exact circle inversion I_{c,s}(p) of a rational planar point.

    Given center c, positive rational inversion power s (squared inversion
    radius), and point p ≠ c, returns q = c + (s / ||p-c||²) * (p - c).
    Each coordinate's numerator and denominator carries at most
    2_730 decimal digits; this conservative bound guarantees the exact
    inverted coordinates stay within the canonical 32,768-digit limit.
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
        for value in (
            self.center.x,
            self.center.y,
            self.power,
            self.point.x,
            self.point.y,
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_INVERSION_INPUT_DIGITS,
                label="inversion coordinate",
            )
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
