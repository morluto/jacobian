"""Euclidean-geometry operation declarations."""

from jacobian._exact import format_canonical_rational
from jacobian.catalog.models import MathTool, MathTools, OperationExample
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
    MathTool(
        operation_id="geometry.euclidean.segment_ratio.compute",
        title="Compute an exact squared segment-length ratio",
        description="Return |A-B|^2 / |C-D|^2 exactly for two rational planar segments; "
        "the denominator segment must be nonzero.",
        request_type=SegmentRatioRequest,
        result_type=SegmentRatioResult,
        run=compute_segment_ratio,
        tags=("geometry", "euclidean", "segment", "ratio"),
        examples=(
            OperationExample(
                name="unit_segments",
                description="Compare two unit segments.",
                input={"segment1": [_ORIGIN, _UNIT_X], "segment2": [_ORIGIN, _UNIT_Y]},
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.euclidean.angle_equality.compute",
        title="Decide exact equality of two planar angles",
        description="Decide whether two unoriented rational planar angles in [0, pi] are "
        "equal, using exact dot and cross products.",
        request_type=AngleEqualityRequest,
        result_type=AngleEqualityResult,
        run=compute_angle_equality,
        tags=("geometry", "euclidean", "angle", "predicate"),
        examples=(
            OperationExample(
                name="right_angles",
                description="Compare two right angles.",
                input={
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
    MathTool(
        operation_id="geometry.euclidean.triangle_similarity.compute",
        title="Decide exact similarity of two triangles",
        description="Decide whether two nondegenerate rational planar triangles have "
        "proportional corresponding side lengths.",
        request_type=TriangleSimilarityRequest,
        result_type=TriangleSimilarityResult,
        run=compute_triangle_similarity,
        tags=("geometry", "euclidean", "triangle", "similarity", "predicate"),
        examples=(
            OperationExample(
                name="scaled_right_triangles",
                description="Compare a unit right triangle with its scale-two image.",
                input={
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
