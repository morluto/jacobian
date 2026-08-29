"""Segment-owned exact geometry operations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTools
from jacobian.math.geometry._models import (
    SegmentIntersectionRequest,
    SegmentIntersectionResult,
)
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry._tools import segment_intersection

SEGMENT_OPERATIONS: MathTools = (
    geometry_operation(
        "geometry.segments.intersection.compute",
        "Intersect two closed rational segments",
        (
            "Classify the exact intersection of two closed rational segments "
            "as disjoint, one typed contact point, or one maximal overlap."
        ),
        SegmentIntersectionRequest,
        SegmentIntersectionResult,
        segment_intersection,
        "geometry",
        "intersection",
        examples=(
            example(
                "crossing_segments",
                "Intersect two closed rational segments at one proper crossing.",
                {
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
