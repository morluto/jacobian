"""Domain-owned exact Euclidean geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import format_canonical_rational
from jacobian.math.geometry.euclidean._models import (
    AngleEqualityRequest,
    AngleEqualityResult,
    RationalPoint2D,
    SegmentRatioRequest,
    SegmentRatioResult,
    TriangleSimilarityRequest,
    TriangleSimilarityResult,
)
from jacobian.math.geometry.euclidean.operations import (
    angles_equal,
    squared_segment_ratio,
    triangles_similar,
)


def _squared_dist_sq(p: RationalPoint2D, q: RationalPoint2D) -> Fraction:
    """Squared distance between two points."""
    dx = q.x.as_fraction() - p.x.as_fraction()
    dy = q.y.as_fraction() - p.y.as_fraction()
    return dx * dx + dy * dy


def compute_segment_ratio(request: SegmentRatioRequest) -> SegmentRatioResult:
    """Compute the ratio of squared lengths of two segments."""
    d1 = _squared_dist_sq(request.segment1[0], request.segment1[1])
    d2 = _squared_dist_sq(request.segment2[0], request.segment2[1])
    ratio = squared_segment_ratio(request.segment1, request.segment2)
    return SegmentRatioResult(
        squared_ratio=format_canonical_rational(ratio),
        ratio_numerator=format_canonical_rational(d1),
        ratio_denominator=format_canonical_rational(d2),
    )


def compute_angle_equality(request: AngleEqualityRequest) -> AngleEqualityResult:
    """Check if two angles are equal using cross/dot product ratios.

    Two angles are equal iff the cross products and dot products are
    proportional: cross1/dot1 = cross2/dot2.
    """
    return AngleEqualityResult(
        equal=angles_equal(
            request.vertex1,
            request.ray1_a,
            request.ray1_b,
            request.vertex2,
            request.ray2_a,
            request.ray2_b,
        )
    )


def compute_triangle_similarity(
    request: TriangleSimilarityRequest,
) -> TriangleSimilarityResult:
    """Check if two triangles are similar.

    Two triangles are similar iff their corresponding sides are proportional.
    We compute all three squared side lengths for each triangle and check
    if they are proportional up to a common factor.
    """
    return TriangleSimilarityResult(
        similar=triangles_similar(request.triangle1, request.triangle2)
    )


__all__ = [
    "compute_angle_equality",
    "compute_segment_ratio",
    "compute_triangle_similarity",
]
