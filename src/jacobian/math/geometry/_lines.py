"""Line-owned exact geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry._models import (
    GeometryLineIntersectionResult,
    GeometryPointResult,
    LinePairRequest,
    PointLineRequest,
)
from jacobian.math.geometry._tools import (
    line_intersection,
    projection,
)

LINE_OPERATIONS: MathTools = (
    MathTool(
        operation_id="geometry.lines.compute.intersection",
        title="Intersect exact lines",
        description="Return the exact point, parallel status, or coincident status for two lines.",
        request_type=LinePairRequest,
        result_type=GeometryLineIntersectionResult,
        run=line_intersection,
        tags=("geometry", "intersection"),
        examples=(
            OperationExample(
                name="crossing_diagonals",
                description="Intersect two crossing lines.",
                input={
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
    MathTool(
        operation_id="geometry.line.compute.projection",
        title="Project point onto line",
        description="Construct the exact orthogonal projection of a rational point onto a line.",
        request_type=PointLineRequest,
        result_type=GeometryPointResult,
        run=projection,
        tags=("geometry", "construction"),
        examples=(
            OperationExample(
                name="projection_to_x_axis",
                description="Project (1,2) onto the x-axis.",
                input={
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
