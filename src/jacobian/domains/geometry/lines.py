"""Line-owned exact geometry operations."""

from jacobian.contracts.geometry import (
    GeometryBooleanResult,
    GeometryLineIntersectionResult,
    GeometryPointResult,
    LinePairRequest,
    PointLineRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.geometry._support import geometry_operation
from jacobian.domains.geometry.operations import (
    line_intersection,
    line_predicate,
    projection,
)

LINE_OPERATIONS = (
    geometry_operation(
        "geometry.lines.decide.parallel",
        "Decide parallel lines",
        "Decide whether two exact lines are parallel.",
        LinePairRequest,
        GeometryBooleanResult,
        line_predicate(lambda first, second: bool(first.is_parallel(second))),
        "geometry",
        "line",
        examples=(
            example(
                "parallel_horizontal_lines",
                "Check two horizontal parallel lines.",
                {
                    "first_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "1", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                    },
                    "second_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "1", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "1", "den": "1"},
                            "y": {"num": "1", "den": "1"},
                        },
                    },
                },
            ),
            example(
                "parallel_diagonal_lines",
                "Decide whether two diagonal lines are parallel; each line needs two distinct points.",
                {
                    "first_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "1", "den": "1"},
                            "y": {"num": "1", "den": "1"},
                        },
                    },
                    "second_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "1", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "1", "den": "1"},
                            "y": {"num": "2", "den": "1"},
                        },
                    },
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.lines.decide.perpendicular",
        "Decide perpendicular lines",
        "Decide whether two exact lines are perpendicular.",
        LinePairRequest,
        GeometryBooleanResult,
        line_predicate(lambda first, second: bool(first.is_perpendicular(second))),
        "geometry",
        "line",
        examples=(
            example(
                "perpendicular_axes",
                "Check perpendicular coordinate axes.",
                {
                    "first_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "1", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                    },
                    "second_line": {
                        "first": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "0", "den": "1"},
                        },
                        "second": {
                            "x": {"num": "0", "den": "1"},
                            "y": {"num": "1", "den": "1"},
                        },
                    },
                },
            ),
        ),
    ),
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
