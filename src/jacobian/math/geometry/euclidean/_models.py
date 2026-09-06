"""Typed wire contracts for exact Euclidean geometry operations."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.math.geometry._models import RationalPoint2D


class Triangle(StrictModel):
    """A triangle as three rational points."""

    a: RationalPoint2D
    b: RationalPoint2D
    c: RationalPoint2D


class SegmentRatioRequest(StrictModel):
    """Check if two segments have a given squared length ratio."""

    segment1: tuple[RationalPoint2D, RationalPoint2D]
    segment2: tuple[RationalPoint2D, RationalPoint2D]


class AngleEqualityRequest(StrictModel):
    """Check if two angles are equal (via cross-product and dot-product ratios)."""

    vertex1: RationalPoint2D
    ray1_a: RationalPoint2D
    ray1_b: RationalPoint2D
    vertex2: RationalPoint2D
    ray2_a: RationalPoint2D
    ray2_b: RationalPoint2D


class AngleEqualityResult(StrictModel):
    """A source-bound claim about equality of two angles."""

    vertex1: RationalPoint2D
    ray1_a: RationalPoint2D
    ray1_b: RationalPoint2D
    vertex2: RationalPoint2D
    ray2_a: RationalPoint2D
    ray2_b: RationalPoint2D
    equal: bool


class TriangleSimilarityRequest(StrictModel):
    """Check if two triangles are similar."""

    triangle1: Triangle
    triangle2: Triangle


class TriangleSimilarityResult(StrictModel):
    """A source-bound claim about similarity of two triangles."""

    triangle1: Triangle
    triangle2: Triangle
    similar: bool


__all__ = [
    "AngleEqualityRequest",
    "AngleEqualityResult",
    "RationalPoint2D",
    "SegmentRatioRequest",
    "Triangle",
    "TriangleSimilarityRequest",
    "TriangleSimilarityResult",
]
