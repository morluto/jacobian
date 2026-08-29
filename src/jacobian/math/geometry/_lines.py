"""Line-owned exact geometry operations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTools
from jacobian.math.geometry._models import (
    GeometryLineIntersectionResult,
    GeometryPointResult,
    LinePairRequest,
    PointLineRequest,
)
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry._tools import (
    line_intersection,
    projection,
)

LINE_OPERATIONS: MathTools = (
    geometry_operation(
        "geometry.lines.compute.intersection",
        "Intersect exact lines",
        "Return the exact point, parallel status, or coincident status for two lines.",
        LinePairRequest,
        GeometryLineIntersectionResult,
        line_intersection,
        "geometry",
        "intersection",
        examples=(
            example(
                "crossing_diagonals",
                "Intersect two crossing lines.",
                {
                    "first_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "2", "den": "1"},
                            "y": {"num": "2", "den": "1"},
                        },
                    },
                    "second_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "2", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "2", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                    },
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.line.compute.projection",
        "Project point onto line",
        "Construct the exact orthogonal projection of a rational point onto a line.",
        PointLineRequest,
        GeometryPointResult,
        projection,
        "geometry",
        "construction",
        examples=(
            example(
                "projection_to_x_axis",
                "Project (1,2) onto the x-axis.",
                {
                    "point": {
                        "x": {"num": "1", "den": "1"},
                        "y": {"num": "2", "den": "1"},
                    },
                    "line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "1", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                    },
                },
            ),
        ),
    ),
)
