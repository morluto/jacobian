"""Triangle-owned exact geometry operations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTools
from jacobian.math.geometry._models import (
    CircumcircleRequest,
    GeometryCircleResult,
)
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry._tools import circumcircle

TRIANGLE_OPERATIONS: MathTools = (
    geometry_operation(
        "geometry.triangle.compute.circumcircle",
        "Construct triangle circumcircle",
        "Construct the exact circumcenter and squared radius of a nondegenerate rational triangle.",
        CircumcircleRequest,
        GeometryCircleResult,
        circumcircle,
        "geometry",
        "circle",
        examples=(
            example(
                "right_triangle_circumcircle",
                "Construct the circumcircle of a right triangle.",
                {
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
