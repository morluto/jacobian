"""Triangle-owned exact geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry._models import (
    CircumcircleRequest,
    GeometryCircleResult,
)
from jacobian.math.geometry._tools import circumcircle

TRIANGLE_OPERATIONS: MathTools = (
    MathTool(
        operation_id="geometry.triangle.compute.circumcircle",
        title="Construct triangle circumcircle",
        description="Construct the exact circumcenter and squared radius of a nondegenerate rational triangle.",
        request_type=CircumcircleRequest,
        result_type=GeometryCircleResult,
        run=circumcircle,
        tags=("geometry", "circle"),
        examples=(
            OperationExample(
                name="right_triangle_circumcircle",
                description="Construct the circumcircle of a right triangle.",
                input={
                    "first": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "second": {
                        "x": {"num": "2", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "third": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "2", "den": "1"},
                    },
                },
            ),
        ),
    ),
)
