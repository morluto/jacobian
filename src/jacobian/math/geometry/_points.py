"""Point-owned exact geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry._models import (
    CircleInversionRequest,
    GeometryBooleanResult,
    GeometryConvexHullResult,
    GeometryPointResult,
    GeometryRationalResult,
    PointPairRequest,
    PointQuadrupleRequest,
    PointSetRequest,
)
from jacobian.math.geometry._tools import (
    circle_inversion,
    concyclic,
    convex_hull_points,
    squared_distance,
)

POINT_OPERATIONS: MathTools = (
    MathTool(
        operation_id="geometry.points.compute.squared_distance",
        title="Compute squared distance",
        description="Compute exact squared Euclidean distance between two rational points.",
        request_type=PointPairRequest,
        result_type=GeometryRationalResult,
        run=squared_distance,
        tags=("geometry", "distance"),
        examples=(
            OperationExample(
                name="diagonal_squared_distance",
                description="Compute the squared distance from (0,0) to (2,2).",
                input={
                    "first": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "second": {
                        "x": {"num": "2", "den": "1"},
                        "y": {"num": "2", "den": "1"},
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.points.decide.concyclic",
        title="Decide concyclicity",
        description="Decide whether four rational points lie on one circle.",
        request_type=PointQuadrupleRequest,
        result_type=GeometryBooleanResult,
        run=concyclic,
        tags=("geometry", "circle"),
        examples=(
            OperationExample(
                name="unit_circle_points",
                description="Check four points on the unit circle.",
                input={
                    "first": {
                        "x": {"num": "1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "second": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "1", "den": "1"},
                    },
                    "third": {
                        "x": {"num": "-1", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "fourth": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "-1", "den": "1"},
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.points.compute.convex_hull",
        title="Construct planar convex hull",
        description="Construct the exact convex hull vertices of a finite rational point set.",
        request_type=PointSetRequest,
        result_type=GeometryConvexHullResult,
        run=convex_hull_points,
        tags=("geometry", "convexity"),
        examples=(
            OperationExample(
                name="square_convex_hull",
                description="Construct the hull of a rational square.",
                input={
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "2", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "2", "den": "1"}},
                        {"x": {"num": "2", "den": "1"}, "y": {"num": "2", "den": "1"}},
                    ]
                },
            ),
            OperationExample(
                name="triangle_convex_hull",
                description="Construct the convex hull of a triangle; the input points must be unique.",
                input={
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "3", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "3", "den": "1"}},
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.points.compute.circle_inversion",
        title="Invert a point in a circle",
        description="Given a rational planar center c, a positive rational inversion power "
        "s (the squared inversion radius), and a rational planar point p != c, "
        "return the exact inverted point I_{c,s}(p) = c + (s / ||p - c||^2) * "
        "(p - c) using Jacobian's canonical rational planar point value.",
        request_type=CircleInversionRequest,
        result_type=GeometryPointResult,
        run=circle_inversion,
        tags=("geometry", "inversion", "circle"),
        examples=(
            OperationExample(
                name="unit_inversion_of_two_zero",
                description=(
                    "Unit inversion (s=1) centered at the origin maps (2,0) to "
                    "(1/2, 0). The point must differ from the center."
                ),
                input={
                    "center": {
                        "x": {"num": "0", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                    "power": {"num": "1", "den": "1"},
                    "point": {
                        "x": {"num": "2", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                },
            ),
        ),
    ),
)
