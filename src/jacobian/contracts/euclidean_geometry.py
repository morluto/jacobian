"""Typed wire contracts for exact Euclidean geometry operations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class RationalPoint2D(ContractModel):
    """A point in the rational plane."""

    x: CanonicalRational
    y: CanonicalRational


class Triangle(ContractModel):
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
            raise ValueError("triangle must be non-degenerate")
        return self


class SegmentRatioRequest(ContractModel):
    """Check if two segments have a given squared length ratio."""

    segment1: tuple[RationalPoint2D, RationalPoint2D]
    segment2: tuple[RationalPoint2D, RationalPoint2D]

    @model_validator(mode="after")
    def require_nonzero_second_segment(self) -> Self:
        start, end = self.segment2
        if (
            start.x.as_fraction() == end.x.as_fraction()
            and start.y.as_fraction() == end.y.as_fraction()
        ):
            raise ValueError("second segment must be nonzero")
        return self


class SegmentRatioResult(ContractModel):
    """Ratio of squared lengths and whether it matches a target."""

    squared_ratio: str
    ratio_numerator: str
    ratio_denominator: str


class AngleEqualityRequest(ContractModel):
    """Check if two angles are equal (via cross-product and dot-product ratios)."""

    vertex1: RationalPoint2D
    ray1_a: RationalPoint2D
    ray1_b: RationalPoint2D
    vertex2: RationalPoint2D
    ray2_a: RationalPoint2D
    ray2_b: RationalPoint2D

    @model_validator(mode="after")
    def require_nonzero_rays(self) -> Self:
        for vertex, first, second in (
            (self.vertex1, self.ray1_a, self.ray1_b),
            (self.vertex2, self.ray2_a, self.ray2_b),
        ):
            for endpoint in (first, second):
                if (
                    endpoint.x.as_fraction() == vertex.x.as_fraction()
                    and endpoint.y.as_fraction() == vertex.y.as_fraction()
                ):
                    raise ValueError("angle rays must be nonzero")
        return self


class AngleEqualityResult(ContractModel):
    """Whether the two angles are equal."""

    equal: bool


class TriangleSimilarityRequest(ContractModel):
    """Check if two triangles are similar."""

    triangle1: Triangle
    triangle2: Triangle


class TriangleSimilarityResult(ContractModel):
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
