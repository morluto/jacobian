"""Point-owned exact geometry operations."""

from jacobian.catalog._examples import example
from jacobian.math.geometry._models import (
    CircleInversionRequest,
    GeometryBooleanResult,
    GeometryConvexHullResult,
    GeometryPointResult,
    GeometryRationalResult,
    PointPairRequest,
    PointQuadrupleRequest,
    PointSetRequest,
    PointTripleRequest,
)
from jacobian.math.geometry._operations import (
    circle_inversion,
    collinear,
    concyclic,
    convex_hull_points,
    squared_distance,
)
from jacobian.math.geometry._support import geometry_operation

POINT_OPERATIONS = (
    geometry_operation(
        "geometry.points.compute.squared_distance",
        "Compute squared distance",
        "Compute exact squared Euclidean distance between two rational points.",
        PointPairRequest,
        GeometryRationalResult,
        squared_distance,
        "geometry",
        "distance",
        examples=(
            example(
                "diagonal_squared_distance",
                "Compute the squared distance from (0,0) to (2,2).",
                {
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
    geometry_operation(
        "geometry.points.decide.collinear",
        "Decide collinearity",
        "Decide exact collinearity of three rational points.",
        PointTripleRequest,
        GeometryBooleanResult,
        collinear,
        "geometry",
        "incidence",
        examples=(
            example(
                "collinear_x_axis",
                "Check three points on the x-axis.",
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
                        "x": {"num": "2", "den": "1"},
                        "y": {"num": "0", "den": "1"},
                    },
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.points.decide.concyclic",
        "Decide concyclicity",
        "Decide whether four rational points lie on one circle.",
        PointQuadrupleRequest,
        GeometryBooleanResult,
        concyclic,
        "geometry",
        "circle",
        examples=(
            example(
                "unit_circle_points",
                "Check four points on the unit circle.",
                {
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
    geometry_operation(
        "geometry.points.compute.convex_hull",
        "Construct planar convex hull",
        "Construct the exact convex hull vertices of a finite rational point set.",
        PointSetRequest,
        GeometryConvexHullResult,
        convex_hull_points,
        "geometry",
        "convexity",
        examples=(
            example(
                "square_convex_hull",
                "Construct the hull of a rational square.",
                {
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "2", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "2", "den": "1"}},
                        {"x": {"num": "2", "den": "1"}, "y": {"num": "2", "den": "1"}},
                    ]
                },
            ),
            example(
                "triangle_convex_hull",
                "Construct the convex hull of a triangle; the input points must be unique.",
                {
                    "points": [
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "3", "den": "1"}, "y": {"num": "0", "den": "1"}},
                        {"x": {"num": "0", "den": "1"}, "y": {"num": "3", "den": "1"}},
                    ]
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.points.compute.circle_inversion",
        "Invert a point in a circle",
        "Given a rational planar center c, a positive rational inversion power "
        "s (the squared inversion radius), and a rational planar point p != c, "
        "return the exact inverted point I_{c,s}(p) = c + (s / ||p - c||^2) * "
        "(p - c) using Jacobian's canonical rational planar point value.",
        CircleInversionRequest,
        GeometryPointResult,
        circle_inversion,
        "geometry",
        "inversion",
        "circle",
        examples=(
            example(
                "unit_inversion_of_two_zero",
                (
                    "Unit inversion (s=1) centered at the origin maps (2,0) to "
                    "(1/2, 0). The point must differ from the center."
                ),
                {
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
