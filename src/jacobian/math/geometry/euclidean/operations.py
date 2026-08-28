"""Exact native Euclidean-geometry operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry.euclidean._models import Triangle


def _vector(
    vertex: RationalPoint2D,
    endpoint: RationalPoint2D,
) -> tuple[Fraction, Fraction]:
    return (
        endpoint.x.as_fraction() - vertex.x.as_fraction(),
        endpoint.y.as_fraction() - vertex.y.as_fraction(),
    )


def _squared_distance(left: RationalPoint2D, right: RationalPoint2D) -> Fraction:
    dx, dy = _vector(left, right)
    return dx * dx + dy * dy


def squared_segment_ratio(
    first: tuple[RationalPoint2D, RationalPoint2D],
    second: tuple[RationalPoint2D, RationalPoint2D],
) -> Fraction:
    """Return the exact ratio of the two segments' squared lengths."""

    denominator = _squared_distance(*second)
    if denominator == 0:
        raise OperationDomainValidationError(
            location=("second",),
            code="geometry.second_segment_nonzero",
            message="second segment must be nonzero",
        )
    return _squared_distance(*first) / denominator


def angles_equal(
    vertex1: RationalPoint2D,
    ray1_a: RationalPoint2D,
    ray1_b: RationalPoint2D,
    vertex2: RationalPoint2D,
    ray2_a: RationalPoint2D,
    ray2_b: RationalPoint2D,
) -> bool:
    """Decide equality of the two unoriented angles in ``[0, pi]``."""

    vectors = (
        _vector(vertex1, ray1_a),
        _vector(vertex1, ray1_b),
        _vector(vertex2, ray2_a),
        _vector(vertex2, ray2_b),
    )
    if any(not any(vector) for vector in vectors):
        raise OperationDomainValidationError(
            location=("rays",),
            code="geometry.angle_rays_nonzero",
            message="angle rays must be nonzero",
        )

    first_a, first_b, second_a, second_b = vectors
    cross1 = first_a[0] * first_b[1] - first_a[1] * first_b[0]
    dot1 = first_a[0] * first_b[0] + first_a[1] * first_b[1]
    cross2 = second_a[0] * second_b[1] - second_a[1] * second_b[0]
    dot2 = second_a[0] * second_b[0] + second_a[1] * second_b[1]
    return dot1 * abs(cross2) == dot2 * abs(cross1) and (
        dot1 == 0 or dot2 == 0 or (dot1 > 0) == (dot2 > 0)
    )


def triangles_similar(first: Triangle, second: Triangle) -> bool:
    """Decide whether two nondegenerate triangles are similar."""

    first_sides = sorted(
        (
            _squared_distance(first.a, first.b),
            _squared_distance(first.b, first.c),
            _squared_distance(first.a, first.c),
        )
    )
    second_sides = sorted(
        (
            _squared_distance(second.a, second.b),
            _squared_distance(second.b, second.c),
            _squared_distance(second.a, second.c),
        )
    )
    return all(
        left * second_sides[0] == first_sides[0] * right
        for left, right in zip(first_sides, second_sides, strict=True)
    )


__all__ = ["angles_equal", "squared_segment_ratio", "triangles_similar"]
