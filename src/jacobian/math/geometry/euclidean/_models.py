"""Typed wire contracts for exact Euclidean geometry operations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.geometry._models import RationalPoint2D


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


class Triangle(StrictModel):
    """A triangle as three rational points."""

    a: RationalPoint2D
    b: RationalPoint2D
    c: RationalPoint2D

    @model_validator(mode="after")
    def require_non_degenerate(self) -> Self:

        ax, ay = self.a.x.as_fraction(), self.a.y.as_fraction()
        bx, by = self.b.x.as_fraction(), self.b.y.as_fraction()
        cx, cy = self.c.x.as_fraction(), self.c.y.as_fraction()
        # Cross product (b-a) x (c-a) != 0 for non-degenerate
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross == 0:
            raise _validation_error(
                "triangle_non_degenerate", "triangle must be non-degenerate"
            )
        return self


class SegmentRatioRequest(StrictModel):
    """Check if two segments have a given squared length ratio."""

    segment1: tuple[RationalPoint2D, RationalPoint2D]
    segment2: tuple[RationalPoint2D, RationalPoint2D]


class SegmentRatioResult(StrictModel):
    """Ratio of squared lengths and whether it matches a target."""

    squared_ratio: str
    ratio_numerator: str
    ratio_denominator: str


class AngleEqualityRequest(StrictModel):
    """Check if two angles are equal (via cross-product and dot-product ratios)."""

    vertex1: RationalPoint2D
    ray1_a: RationalPoint2D
    ray1_b: RationalPoint2D
    vertex2: RationalPoint2D
    ray2_a: RationalPoint2D
    ray2_b: RationalPoint2D


class AngleEqualityResult(StrictModel):
    """Whether the two angles are equal."""

    equal: bool


class TriangleSimilarityRequest(StrictModel):
    """Check if two triangles are similar."""

    triangle1: Triangle
    triangle2: Triangle


class TriangleSimilarityResult(StrictModel):
    """Whether the two triangles are similar."""

    similar: bool


__all__ = [
    "AngleEqualityRequest",
    "AngleEqualityResult",
    "RationalPoint2D",
    "SegmentRatioRequest",
    "SegmentRatioResult",
    "Triangle",
    "TriangleSimilarityRequest",
    "TriangleSimilarityResult",
]
