"""Triangle-owned exact geometry operations."""

from jacobian.catalog._examples import example
from jacobian.math.geometry._models import (
    CircumcircleRequest,
    GeometryCircleResult,
    GeometryOrientationResult,
    GeometryPointResult,
    PointTripleRequest,
)
from jacobian.math.geometry._operations import centroid, circumcircle, orientation
from jacobian.math.geometry._support import geometry_operation

TRIANGLE_OPERATIONS = (
    geometry_operation(
        "geometry.triangle.compute.orientation",
        "Compute triangle orientation",
        "Return clockwise, collinear, or counterclockwise orientation as -1, 0, or 1.",
        PointTripleRequest,
        GeometryOrientationResult,
        orientation,
        "geometry",
        "orientation",
        examples=(
            example(
                "counterclockwise_triangle",
                "Compute orientation of a counterclockwise triangle.",
                {
                    "first": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "second": {
                        "x": {"num": "1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "third": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "1", "den": "1"},
                    },
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.triangle.compute.centroid",
        "Construct triangle centroid",
        "Construct the exact centroid of three rational points.",
        PointTripleRequest,
        GeometryPointResult,
        centroid,
        "geometry",
        "construction",
        examples=(
            example(
                "right_triangle_centroid",
                "Construct the centroid of a right triangle.",
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
