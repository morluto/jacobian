"""Domain adapter for exact Euclidean geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.euclidean_geometry import (
    AngleEqualityRequest,
    AngleEqualityResult,
    RationalPoint2D,
    SegmentRatioRequest,
    SegmentRatioResult,
    TriangleSimilarityRequest,
    TriangleSimilarityResult,
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
    ratio = d1 / d2
    return SegmentRatioResult(
        squared_ratio=_format_rational(ratio),
        ratio_numerator=_format_rational(d1),
        ratio_denominator=_format_rational(d2),
    )


def _format_rational(value: Fraction) -> str:
    if value.denominator == 1:
        return format_canonical_integer(value.numerator)
    return (
        f"{format_canonical_integer(value.numerator)}/"
        f"{format_canonical_integer(value.denominator)}"
    )


def _dot(v1: tuple[Fraction, Fraction], v2: tuple[Fraction, Fraction]) -> Fraction:
    return v1[0] * v2[0] + v1[1] * v2[1]


def _cross(v1: tuple[Fraction, Fraction], v2: tuple[Fraction, Fraction]) -> Fraction:
    return v1[0] * v2[1] - v1[1] * v2[0]


def _vec(p: RationalPoint2D, q: RationalPoint2D) -> tuple[Fraction, Fraction]:
    return (
        q.x.as_fraction() - p.x.as_fraction(),
        q.y.as_fraction() - p.y.as_fraction(),
    )


def compute_angle_equality(request: AngleEqualityRequest) -> AngleEqualityResult:
    """Check if two angles are equal using cross/dot product ratios.

    Two angles are equal iff the cross products and dot products are
    proportional: cross1/dot1 = cross2/dot2.
    """
    v1a = _vec(request.vertex1, request.ray1_a)
    v1b = _vec(request.vertex1, request.ray1_b)
    v2a = _vec(request.vertex2, request.ray2_a)
    v2b = _vec(request.vertex2, request.ray2_b)

    cross1 = _cross(v1a, v1b)
    dot1 = _dot(v1a, v1b)
    cross2 = _cross(v2a, v2b)
    dot2 = _dot(v2a, v2b)

    abs_cross1 = abs(cross1)
    abs_cross2 = abs(cross2)
    equal = dot1 * abs_cross2 == dot2 * abs_cross1 and (
        dot1 == 0 or dot2 == 0 or (dot1 > 0) == (dot2 > 0)
    )
    return AngleEqualityResult(equal=equal)


def compute_triangle_similarity(
    request: TriangleSimilarityRequest,
) -> TriangleSimilarityResult:
    """Check if two triangles are similar.

    Two triangles are similar iff their corresponding sides are proportional.
    We compute all three squared side lengths for each triangle and check
    if they are proportional up to a common factor.
    """
    t1 = request.triangle1
    t2 = request.triangle2

    sides1 = sorted(
        [
            _squared_dist_sq(t1.a, t1.b),
            _squared_dist_sq(t1.b, t1.c),
            _squared_dist_sq(t1.a, t1.c),
        ]
    )
    sides2 = sorted(
        [
            _squared_dist_sq(t2.a, t2.b),
            _squared_dist_sq(t2.b, t2.c),
            _squared_dist_sq(t2.a, t2.c),
        ]
    )

    # Check if sides1[i] / sides2[i] is constant
    if sides2[0] == 0 or sides1[0] == 0:
        return TriangleSimilarityResult(similar=False)

    # All ratios must be equal
    ratios = [s1 / s2 for s1, s2 in zip(sides1, sides2, strict=True)]
    return TriangleSimilarityResult(similar=all(r == ratios[0] for r in ratios))


__all__ = [
    "compute_angle_equality",
    "compute_segment_ratio",
    "compute_triangle_similarity",
]
