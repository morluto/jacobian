"""Segment-owned exact geometry capabilities."""

from jacobian.contracts.geometry import (
    GeometryPointResult,
    PointPairRequest,
    SegmentIntersectionRequest,
    SegmentIntersectionResult,
)
from jacobian.domains._examples import example
from jacobian.domains.geometry._support import geometry_operation
from jacobian.domains.geometry.operations import midpoint, segment_intersection

SEGMENT_CAPABILITIES = (
    geometry_operation(
        "geometry.segment.compute.midpoint",
        "Construct segment midpoint",
        "Construct the exact midpoint of two rational endpoints.",
        PointPairRequest,
        GeometryPointResult,
        midpoint,
        "geometry",
        "construction",
        invocation_examples=(
            example(
                "segment_midpoint",
                "Construct the midpoint of a unit segment.",
                {
                    "first": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "second": {
                        "x": {"num": "1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                },
            ),
        ),
    ),
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
        relation_id="geometry.segments.intersection.relation",
        invocation_examples=(
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
