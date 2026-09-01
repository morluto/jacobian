"""Segment-owned exact geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry._models import (
    SegmentIntersectionRequest,
    SegmentIntersectionResult,
)
from jacobian.math.geometry._tools import segment_intersection

SEGMENT_OPERATIONS: MathTools = (
    MathTool(
        operation_id="geometry.segments.intersection.compute",
        title="Intersect two closed rational segments",
        description=(
            "Classify the exact intersection of two closed rational segments "
            "as disjoint, one typed contact point, or one maximal overlap."
        ),
        request_type=SegmentIntersectionRequest,
        result_type=SegmentIntersectionResult,
        run=segment_intersection,
        tags=("geometry", "intersection"),
        examples=(
            OperationExample(
                name="crossing_segments",
                description="Intersect two closed rational segments at one proper crossing.",
                input={
                    "first": {
                        "start": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "end": {
                            "x": {"num": "2", "den": "1"},
                            "y": {"num": "2", "den": "1"},
                        },
                    },
                    "second": {
                        "start": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "2", "den": "1"},
                        },
                        "end": {
                            "x": {"num": "2", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                    },
                },
            ),
        ),
    ),
)
