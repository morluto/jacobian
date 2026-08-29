"""Euclidean-geometry operation declarations."""

from jacobian._exact import format_canonical_rational
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTools
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry.euclidean._models import (
    AngleEqualityRequest,
    AngleEqualityResult,
    SegmentRatioRequest,
    SegmentRatioResult,
    TriangleSimilarityRequest,
    TriangleSimilarityResult,
)
from jacobian.math.geometry.euclidean.operations import (
    _squared_segment_ratio_data,
    angles_equal,
    triangles_similar,
)


def compute_segment_ratio(request: SegmentRatioRequest) -> SegmentRatioResult:
    numerator, denominator, ratio = _squared_segment_ratio_data(
        request.segment1,
        request.segment2,
        denominator_location=("segment2",),
    )
    return SegmentRatioResult(
        squared_ratio=format_canonical_rational(ratio),
        ratio_numerator=format_canonical_rational(numerator),
        ratio_denominator=format_canonical_rational(denominator),
    )


def compute_angle_equality(request: AngleEqualityRequest) -> AngleEqualityResult:
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
    return TriangleSimilarityResult(
        similar=triangles_similar(request.triangle1, request.triangle2)
    )


_ORIGIN = {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}}
_UNIT_X = {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}}
_UNIT_Y = {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}}

TOOLS: MathTools = (
    geometry_operation(
        "geometry.euclidean.segment_ratio.compute",
        "Compute an exact squared segment-length ratio",
        "Return |A-B|^2 / |C-D|^2 exactly for two rational planar segments; "
        "the denominator segment must be nonzero.",
        SegmentRatioRequest,
        SegmentRatioResult,
        compute_segment_ratio,
        "geometry",
        "euclidean",
        "segment",
        "ratio",
        examples=(
            example(
                "unit_segments",
                "Compare two unit segments.",
                {"segment1": [_ORIGIN, _UNIT_X], "segment2": [_ORIGIN, _UNIT_Y]},
            ),
        ),
    ),
    geometry_operation(
        "geometry.euclidean.angle_equality.compute",
        "Decide exact equality of two planar angles",
        "Decide whether two unoriented rational planar angles in [0, pi] are "
        "equal, using exact dot and cross products.",
        AngleEqualityRequest,
        AngleEqualityResult,
        compute_angle_equality,
        "geometry",
        "euclidean",
        "angle",
        "predicate",
        examples=(
            example(
                "right_angles",
                "Compare two right angles.",
                {
                    "vertex1": _ORIGIN,
                    "ray1_a": _UNIT_X,
                    "ray1_b": _UNIT_Y,
                    "vertex2": _ORIGIN,
                    "ray2_a": _UNIT_Y,
                    "ray2_b": {
                        "x": {"num": "-1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.euclidean.triangle_similarity.compute",
        "Decide exact similarity of two triangles",
        "Decide whether two nondegenerate rational planar triangles have "
        "proportional corresponding side lengths.",
        TriangleSimilarityRequest,
        TriangleSimilarityResult,
        compute_triangle_similarity,
        "geometry",
        "euclidean",
        "triangle",
        "similarity",
        "predicate",
        examples=(
            example(
                "scaled_right_triangles",
                "Compare a unit right triangle with its scale-two image.",
                {
                    "triangle1": {"a": _ORIGIN, "b": _UNIT_X, "c": _UNIT_Y},
                    "triangle2": {
                        "a": _ORIGIN,
                        "b": {
                            "x": {"num": "2", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "c": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "2", "den": "1"},
                        },
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
