"""Euclidean geometry operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.euclidean._models import (
    AngleEqualityRequest,
    AngleEqualityResult,
    SegmentRatioRequest,
    SegmentRatioResult,
    TriangleSimilarityRequest,
    TriangleSimilarityResult,
)
from jacobian.math.geometry.euclidean._operations import (
    compute_angle_equality,
    compute_segment_ratio,
    compute_triangle_similarity,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


EUCLIDEAN_GEOMETRY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "geometry.euclidean.segment_ratio.compute",
        "Compute squared-length ratio of two segments",
        "Compute the ratio of squared lengths of two rational segments.",
        SegmentRatioRequest,
        SegmentRatioResult,
        compute_segment_ratio,
        "geometry",
        "segment-ratio",
        "exact",
        examples=(
            example(
                "unit_segments",
                "Ratio of unit horizontal segment to unit vertical segment.",
                {
                    "segment1": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
                    ],
                    "segment2": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
    _op(
        "geometry.euclidean.angle_equality.compute",
        "Check angle equality",
        "Check if two angles are equal using exact cross/dot product ratios.",
        AngleEqualityRequest,
        AngleEqualityResult,
        compute_angle_equality,
        "geometry",
        "angle",
        "exact",
        examples=(
            example(
                "right_angle",
                "Check two right angles are equal.",
                {
                    "vertex1": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "ray1_a": {
                        "x": {"num": "1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "ray1_b": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "1", "den": "1"},
                    },
                    "vertex2": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "ray2_a": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "1", "den": "1"},
                    },
                    "ray2_b": {
                        "x": {"num": "-1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                },
            ),
        ),
    ),
    _op(
        "geometry.euclidean.triangle_similarity.compute",
        "Check triangle similarity",
        "Check if two triangles are similar by comparing side-length ratios.",
        TriangleSimilarityRequest,
        TriangleSimilarityResult,
        compute_triangle_similarity,
        "geometry",
        "triangle-similarity",
        "exact",
        examples=(
            example(
                "similar_triangles",
                "Two similar triangles.",
                {
                    "triangle1": {
                        "a": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "b": {
                            "x": {"num": "1", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "c": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "1", "den": "1"},
                        },
                    },
                    "triangle2": {
                        "a": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
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


TOOLS = EUCLIDEAN_GEOMETRY_OPERATIONS

__all__ = ["TOOLS"]
